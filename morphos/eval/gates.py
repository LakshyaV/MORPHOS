"""Pre-registered pass/fail gates.

Every threshold is a number a script checks, so nothing advances on a subjective
read of a loss curve.

G1 (growth + persistence) gates on FULL-GRID per-element RMSE, which is exactly
`sqrt(training loss)` -- so G1 is predictable from the loss curve alone.
docs/PROTOCOL.md §1.1 defines RMSE over the union mask; gating on that at 0.05 is
a ~2x harsher bar that would fail a perfectly good organism, so the union form is
kept for the *recovery* metric where empty background is definitionally
irrelevant, and reported alongside here.

Gates must never train, never mutate the model, and never share randomness with
the training loop -- otherwise an early stop and the finally-reported number are
the same draw.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from morphos.config import Config
from morphos.eval.metrics import alive_count, body_mask, iou, rmse, union_mask
from morphos.seeding import make_rng
from morphos.substrate.nca import NCAOrganism

GATE_RNG_SALT = 0xE0A1


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    metrics: dict[str, float]
    thresholds: dict[str, str]
    detail: dict[str, Any] = field(default_factory=dict)

    def report(self) -> str:
        head = f"{'PASS' if self.passed else 'FAIL'}  {self.name}"
        lines = [head]
        for k, v in self.metrics.items():
            thr = self.thresholds.get(k)
            mark = "" if thr is None else f"   (need {thr})"
            lines.append(f"    {k:<28} {v:>9.4f}{mark}")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            **{f"m_{k}": v for k, v in self.metrics.items()},
        }


def _grow(
    model: NCAOrganism, n: int, steps: int, *, generator, device
) -> Tensor:
    x = model.seed_state(n, device=device)
    return model.rollout(x, steps, generator=generator)


@torch.no_grad()
def check_g1(
    model: NCAOrganism,
    target: Tensor,
    *,
    device: torch.device,
    seed: int = 0,
    n_rollouts: int = 128,
    batch: int = 32,
    t_grow: int = 64,
    t_persist: int = 256,
) -> GateResult:
    """Growth and persistence.

    Rolls CONTINUOUSLY to t_persist rather than restarting, snapshotting at
    t_grow. The persistence clause catches the classic NCA failure where the
    organism looks perfect at the training horizon and then explodes or dissolves
    past it.
    """
    was_training = model.training
    model.eval()
    rng = make_rng(seed ^ GATE_RNG_SALT, device)
    tgt_mask = body_mask(target)

    iou64, iou256, rmse64, rmse64_union, area_ratio = [], [], [], [], []

    for _ in range(max(1, n_rollouts // batch)):
        x = model.seed_state(batch, device=device)
        x = model.rollout(x, t_grow, generator=rng.update)

        iou64.append(iou(body_mask(x), tgt_mask))
        rmse64.append(rmse(x, target))
        rmse64_union.append(rmse(x, target, mask=union_mask(body_mask(x), tgt_mask)))
        a64 = alive_count(x).cpu().double()

        x = model.rollout(x, t_persist - t_grow, generator=rng.update)
        iou256.append(iou(body_mask(x), tgt_mask))
        a256 = alive_count(x).cpu().double()
        area_ratio.append(a256 / a64.clamp(min=1))

    iou64_t = torch.cat(iou64)
    iou256_t = torch.cat(iou256)
    ratio_t = torch.cat(area_ratio)

    m = {
        "iou_64_mean": iou64_t.mean().item(),
        "iou_64_p5": torch.quantile(iou64_t, 0.05).item(),
        "rmse_64_mean": torch.cat(rmse64).mean().item(),
        "rmse_64_union": torch.cat(rmse64_union).mean().item(),
        "iou_256_mean": iou256_t.mean().item(),
        "area_ratio_mean": ratio_t.mean().item(),
        "area_ratio_in_band_frac": ((ratio_t >= 0.9) & (ratio_t <= 1.1)).double().mean().item(),
    }
    passed = (
        m["iou_64_mean"] >= 0.90
        and m["iou_64_p5"] >= 0.85
        and m["rmse_64_mean"] <= 0.05
        and m["iou_256_mean"] >= 0.85
        and 0.9 <= m["area_ratio_mean"] <= 1.1
        and m["area_ratio_in_band_frac"] >= 0.90
    )
    if was_training:
        model.train()
    return GateResult(
        name="G1_growth_persistence",
        passed=passed,
        metrics=m,
        thresholds={
            "iou_64_mean": ">= 0.90",
            "iou_64_p5": ">= 0.85",
            "rmse_64_mean": "<= 0.05",
            "iou_256_mean": ">= 0.85",
            "area_ratio_mean": "in [0.9, 1.1]",
            "area_ratio_in_band_frac": ">= 0.90",
        },
    )


@torch.no_grad()
def check_g2(
    model: NCAOrganism,
    target: Tensor,
    *,
    device: torch.device,
    seed: int = 0,
    n_trials: int = 256,
    batch: int = 32,
    severity: float = 0.30,
    tolerance: float = 0.02,
    t_mature: int = 64,
    t_recover: int = 128,
) -> GateResult:
    """Regeneration after a disk containing `severity` of alive cells is killed."""
    from morphos.damage.ops import kill_fraction

    was_training = model.training
    model.eval()
    rng = make_rng(seed ^ GATE_RNG_SALT ^ 0x2, device)
    tgt_mask = body_mask(target)

    best_iou, sustained, achieved, covers_centre, tau = [], [], [], [], []

    for _ in range(max(1, n_trials // batch)):
        x = model.seed_state(batch, device=device)
        x = model.rollout(x, t_mature, generator=rng.update)

        x, info = kill_fraction(x, severity, generator=rng.damage, tol=tolerance)
        achieved.append(info["achieved_f"].cpu().double())
        covers_centre.append(info["covers_centre"].cpu())

        curve = torch.empty(x.shape[0], t_recover, dtype=torch.float64)
        for t in range(t_recover):
            x = model.rollout(x, 1, generator=rng.update)
            curve[:, t] = iou(body_mask(x), tgt_mask)

        best_iou.append(curve.max(dim=1).values)
        # Sustained-for-20 variant: the form that goes in the paper.
        ok = curve >= 0.90
        window = ok.unfold(1, 20, 1).all(dim=2).any(dim=1) if t_recover >= 20 else ok.any(dim=1)
        sustained.append(window)
        first = torch.where(ok.any(dim=1), ok.double().argmax(dim=1).double(), torch.tensor(float("inf")))
        tau.append(first)

    best = torch.cat(best_iou)
    ach = torch.cat(achieved)
    cov = torch.cat(covers_centre)
    recovered = best >= 0.90
    tau_t = torch.cat(tau)
    finite_tau = tau_t[torch.isfinite(tau_t)]

    m = {
        "recovered_frac": recovered.double().mean().item(),
        "sustained20_frac": torch.cat(sustained).double().mean().item(),
        "best_iou_mean": best.mean().item(),
        "severity_within_tol_frac": ((ach - severity).abs() <= tolerance).double().mean().item(),
        "severity_mean": ach.mean().item(),
        "tau_recover_median": finite_tau.median().item() if finite_tau.numel() else float("inf"),
        "recovered_frac_centre_covered": (
            recovered[cov].double().mean().item() if cov.any() else float("nan")
        ),
        "recovered_frac_centre_spared": (
            recovered[~cov].double().mean().item() if (~cov).any() else float("nan")
        ),
    }
    # The literal pre-registered gate; the sustained variant is reported, not gated.
    passed = m["recovered_frac"] >= 0.90 and m["severity_within_tol_frac"] >= 0.99

    if was_training:
        model.train()
    return GateResult(
        name="G2_regeneration",
        passed=passed,
        metrics=m,
        thresholds={
            "recovered_frac": ">= 0.90",
            "severity_within_tol_frac": ">= 0.99 (assertion)",
        },
    )


def gate_report(
    model: NCAOrganism, target: Tensor, cfg: Config, *, device: torch.device
) -> list[dict[str, Any]]:
    """Callback shape the training loop expects: a list of plain dicts."""
    results = [
        check_g1(
            model, target, device=device, seed=cfg.seed,
            n_rollouts=cfg.eval.g1_rollouts, batch=cfg.train.batch,
            t_grow=cfg.eval.t_grow, t_persist=cfg.eval.t_persist,
        )
    ]
    if cfg.train.damage_from_step is not None:
        results.append(
            check_g2(
                model, target, device=device, seed=cfg.seed,
                n_trials=cfg.eval.g2_trials, batch=cfg.train.batch,
                severity=cfg.eval.g2_severity, tolerance=cfg.eval.g2_tolerance,
                t_mature=cfg.eval.t_grow, t_recover=cfg.eval.g2_recover_steps,
            )
        )
    return [r.as_dict() for r in results]
