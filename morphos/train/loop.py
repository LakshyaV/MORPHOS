"""The single, phase-parameterised training loop.

One loop handles all three upstream regimes via two step thresholds:

    pool_from_step=None                  -> Growing     (fresh seed every step)
    pool_from_step=k, damage_from_step=None -> Persistent
    pool_from_step=k, damage_from_step=k    -> Regenerating

`train_step` is the whole scientific content and does nothing else: no logging,
no checkpointing, no filesystem, no printing. `train` owns all of that.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import Tensor

from morphos.config import Config, config_dict
from morphos.eval.metrics import alive_count, body_mask, iou
from morphos.io.runlog import RunLog
from morphos.seeding import RNG, make_rng
from morphos.substrate.nca import ALIVE_CHANNEL, NCAOrganism, SENDER_LAYOUT
from morphos.train.losses import normalize_grads_, per_sample_rgba_mse, rgba_mse


class PoolLike(Protocol):
    """Structural type so loop.py does not depend on pool.py existing yet."""

    def sample(self, batch: int, *, generator: torch.Generator) -> tuple[Tensor, Tensor]: ...
    def commit(self, idx: Tensor, states: Tensor) -> None: ...


class DeathReset(Exception):
    """Raised inside train_step when the organism has collapsed to all-dead."""


@dataclass
class DeathGuard:
    """Detects the unrecoverable all-dead collapse.

    Why this is mandatory rather than nice-to-have: once every cell is dead the
    state is identically zero, and the alive mask is a hard threshold comparison
    with no gradient. So the gradient into the weights is *exactly* zero, and with
    per-tensor normalisation 0/(0+eps) stays zero while Adam's moments decay. The
    model is frozen forever. This is a trap with a measure-zero escape, not a local
    minimum that annealing escapes.

    It cannot fire spuriously. fc2 is zero-initialised, so an untrained rule
    returns the seed exactly, whose centre alpha is 1.0 -- far above the 0.1
    threshold. `amax` asks "is anything alive at all", not "how big", so a
    legitimately-not-yet-grown organism always passes.

    Never trigger on `alive_count < K`: that fires on step 1 of every run.
    """

    threshold: float = 0.1
    patience: int = 3
    strikes: int = 0

    def observe(self, out: Tensor) -> bool:
        dead_now = out[:, ALIVE_CHANNEL].amax().item() < self.threshold
        self.strikes = self.strikes + 1 if dead_now else 0
        return self.strikes >= self.patience


@dataclass
class TrainState:
    model: NCAOrganism
    opt: torch.optim.Optimizer
    sched: torch.optim.lr_scheduler.LRScheduler
    seed_state: Tensor
    pool: PoolLike | None = None
    guard: DeathGuard = field(default_factory=DeathGuard)
    rng: RNG | None = None
    step: int = 0
    n_resets: int = 0


def build_state(cfg: Config, *, device: torch.device, weight_seed: int) -> TrainState:
    """Fresh model, optimizer and scheduler. Called again on a death reset."""
    torch.manual_seed(weight_seed)
    model = NCAOrganism(
        layout=SENDER_LAYOUT,
        hidden=cfg.nca.hidden,
        grid=cfg.nca.grid,
        fire_rate=cfg.nca.fire_rate,
        alive_threshold=cfg.nca.alive_threshold,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=list(cfg.train.lr_milestones), gamma=cfg.train.lr_gamma
    )
    return TrainState(
        model=model,
        opt=opt,
        sched=sched,
        seed_state=model.seed_state(1, device=device),
        guard=DeathGuard(
            threshold=cfg.guard.death_threshold, patience=cfg.guard.death_patience
        ),
    )


def draw_rollout_len(cfg: Config, rng: RNG) -> int:
    """ONE scalar per training step, not per sample -- matching upstream."""
    lo, hi = cfg.train.rollout_min, cfg.train.rollout_max
    u = torch.rand((), generator=rng.task, device=rng.device).item()
    return int(lo + (hi - lo + 1) * u) if hi > lo else lo


def _pool_active(cfg: Config, step: int) -> bool:
    k = cfg.train.pool_from_step
    return k is not None and step >= k


def _damage_active(cfg: Config, step: int) -> bool:
    k = cfg.train.damage_from_step
    return k is not None and step >= k


def train_step(st: TrainState, target: Tensor, rng: RNG, cfg: Config) -> dict[str, float]:
    """One optimisation step. Raises DeathReset if the organism has collapsed."""
    B = cfg.train.batch
    n_steps = draw_rollout_len(cfg, rng)
    use_pool = _pool_active(cfg, st.step) and st.pool is not None

    if use_pool:
        idx, x0 = st.pool.sample(B, generator=rng.pool)
        # Descending by loss => index 0 is the WORST. Permute idx and states
        # TOGETHER; permuting only states writes every result back to the wrong
        # pool slot, which merely looks like noise and costs days to find.
        order = per_sample_rgba_mse(x0, target).argsort(descending=True)
        idx, x0 = idx[order], x0[order]
        x0[: cfg.train.reseed_worst] = st.seed_state

        if _damage_active(cfg, st.step) and cfg.train.damage_best > 0:
            from morphos.damage.ops import sample_training_masks

            keep = sample_training_masks(
                cfg.train.damage_best,
                cfg.nca.grid,
                generator=rng.damage,
                device=x0.device,
            )
            # Damage the BEST (lowest-loss) samples; damaging a struggling pattern
            # would not test regeneration. Applied strictly before the rollout.
            x0[-cfg.train.damage_best :] = x0[-cfg.train.damage_best :] * keep
    else:
        idx = None
        x0 = st.seed_state.expand(B, -1, -1, -1).clone()

    out = st.model.rollout(
        x0,
        n_steps,
        generator=rng.update,
        checkpoint_every=cfg.train.checkpoint_every,
    )

    if st.guard.observe(out):
        raise DeathReset(f"all cells dead for {st.guard.patience} consecutive steps")

    loss = rgba_mse(out, target)
    st.opt.zero_grad(set_to_none=True)
    loss.backward()
    if cfg.train.grad_norm_per_tensor:
        normalize_grads_(st.model)
    st.opt.step()
    st.sched.step()

    if use_pool:
        st.pool.commit(idx, out.detach())

    with torch.no_grad():
        counts = alive_count(out).to(torch.float32)
        batch_iou = iou(body_mask(out), body_mask(target))
        frac_dead = (out[:, ALIVE_CHANNEL].amax(dim=(1, 2)) < st.guard.threshold)

    return {
        "loss": loss.item(),
        "lr": st.sched.get_last_lr()[0],
        "n_steps": float(n_steps),
        "alive_min": counts.min().item(),
        "alive_med": counts.median().item(),
        "alive_max": counts.max().item(),
        "frac_dead": frac_dead.to(torch.float32).mean().item(),
        "iou": batch_iou.mean().item(),
        "pool": float(use_pool),
        "damage": float(_damage_active(cfg, st.step) and use_pool),
    }


def _memory_mb(device: torch.device) -> float:
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 1e6
    return float("nan")


def _save_last(st: TrainState, cfg: Config, run_dir: Path, target: Tensor) -> None:
    from morphos.task.targets import target_fingerprint
    from morphos.train.checkpoint import save

    save(
        Path(run_dir) / "ckpt" / "last.pt",
        step=st.step,
        model=st.model,
        optimizer=st.opt,
        scheduler=st.sched,
        rng=st.rng,
        pool=st.pool,
        config=config_dict(cfg),
        target_fingerprint=target_fingerprint(target),
        extra={"n_resets": st.n_resets},
    )


def train(
    cfg: Config,
    target: Tensor,
    *,
    run_dir: str | Path,
    log: RunLog | None = None,
    gate_fn: Any = None,
    resume: str | Path | None = None,
) -> TrainState:
    """Full training run with death-reset handling and periodic gate checks."""
    from morphos.task.targets import target_fingerprint
    from morphos.train.checkpoint import load as load_ckpt
    from morphos.train.checkpoint import save as save_ckpt

    device = torch.device(cfg.device)
    target = target.to(device)
    run_dir = Path(run_dir)
    owns_log = log is None
    log = log or RunLog(run_dir, config_dict(cfg), resume=resume is not None)

    rng = make_rng(cfg.seed, device)
    st = build_state(cfg, device=device, weight_seed=cfg.seed * 1000 + 97)
    st.rng = rng

    if resume is not None:
        ckpt = load_ckpt(
            resume,
            model=st.model,
            optimizer=st.opt,
            scheduler=st.sched,
            rng=st.rng,
            pool=st.pool,
            map_location=device,
            expect_target_fingerprint=target_fingerprint(target),
        )
        st.step = int(ckpt["step"])
        log.event("resume", step=st.step, path=str(resume))

    ema: float | None = None
    t0 = time.perf_counter()
    consecutive_gate_passes = 0

    while st.step < cfg.train.steps:
        try:
            rec = train_step(st, target, st.rng, cfg)
        except DeathReset as e:
            st.n_resets += 1
            log.event(
                "death_reset",
                step=st.step,
                n_resets=st.n_resets,
                loss_ema=ema,
                reason=str(e),
            )
            if st.n_resets > cfg.guard.max_resets:
                log.event("abort", reason="max death resets exceeded")
                raise RuntimeError(
                    f"organism collapsed {st.n_resets} times; aborting rather than looping"
                ) from e
            # Rebuild EVERYTHING: stale Adam moments would immediately re-kill it,
            # and resuming at the post-milestone 2e-4 LR nearly guarantees a repeat.
            st_new = build_state(
                cfg, device=device, weight_seed=cfg.seed * 1000 + 97 * (st.n_resets + 1)
            )
            st_new.n_resets = st.n_resets
            st_new.rng = make_rng(cfg.seed + st.n_resets, device)
            st = st_new
            ema = None
            continue

        ema = rec["loss"] if ema is None else 0.98 * ema + 0.02 * rec["loss"]
        st.step += 1

        if st.step % cfg.train.log_interval == 0 or st.step == 1:
            log.log(
                step=st.step,
                loss_ema=ema,
                wall_s=time.perf_counter() - t0,
                mem_mb=_memory_mb(device),
                **rec,
            )

        if st.step % cfg.train.ckpt_interval == 0:
            _save_last(st, cfg, run_dir, target)
            save_ckpt(
                run_dir / "ckpt" / f"step_{st.step:06d}.pt",
                step=st.step,
                model=st.model,
                rng=st.rng,
                target_fingerprint=target_fingerprint(target),
            )

        if gate_fn is not None and st.step % cfg.train.gate_interval == 0:
            results = gate_fn(st.model, target, cfg, device=device)
            for r in results:
                log.event("gate", step=st.step, **r)
            all_pass = bool(results) and all(r.get("passed") for r in results)
            consecutive_gate_passes = consecutive_gate_passes + 1 if all_pass else 0
            if cfg.train.early_stop_on_gates and consecutive_gate_passes >= 2:
                log.event("early_stop", step=st.step, reason="gates passed twice")
                break

    _save_last(st, cfg, run_dir, target)
    if owns_log:
        log.close()
    return st
