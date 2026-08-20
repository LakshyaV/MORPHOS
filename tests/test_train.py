"""Tests for losses, the death guard, and the training loop's plumbing."""

from __future__ import annotations

import json

import pytest
import torch

from morphos.config import load_config
from morphos.io.runlog import RunLog, read_metrics
from morphos.seeding import make_rng
from morphos.substrate.nca import ALIVE_CHANNEL, NCAOrganism, SENDER_LAYOUT
from morphos.train.loop import (
    DeathGuard,
    DeathReset,
    build_state,
    draw_rollout_len,
    train_step,
)
from morphos.train.losses import normalize_grads_, per_sample_rgba_mse, rgba_mse
from morphos.task.targets import disk_target


def tiny_cfg(*overrides: str):
    base = [
        "device=cpu",
        "nca.grid=12",
        "nca.hidden=16",
        "target.radius=4.0",
        "train.batch=4",
        "train.reseed_worst=1",
        "train.damage_best=1",
        "train.rollout_min=6",
        "train.rollout_max=8",
        "train.pool_from_step=null",
        "train.damage_from_step=null",
        "train.checkpoint_every=null",
    ]
    return load_config("configs/base.yaml", overrides=base + list(overrides))


# --- losses -------------------------------------------------------------------


def test_rgba_mse_matches_sqrt_relationship():
    """Gate G1 gates on full-grid RMSE precisely because it is sqrt(loss)."""
    t = disk_target(12, 4.0)
    x = torch.zeros(3, SENDER_LAYOUT.total, 12, 12)
    x[:, :4] = t
    assert rgba_mse(x, t).item() == pytest.approx(0.0, abs=1e-12)

    x[:, :4] = t + 0.2
    assert rgba_mse(x, t).item() == pytest.approx(0.04, rel=1e-6)


def test_per_sample_loss_is_per_sample():
    t = disk_target(12, 4.0)
    x = torch.zeros(3, SENDER_LAYOUT.total, 12, 12)
    x[0, :4] = t
    x[1, :4] = t + 0.1
    x[2, :4] = t + 0.2

    per = per_sample_rgba_mse(x, t)
    assert per.shape == (3,)
    assert per[0] < per[1] < per[2]
    assert per.mean().item() == pytest.approx(rgba_mse(x, t).item(), rel=1e-6)


def test_grad_norm_is_per_tensor_and_excludes_buffers():
    """Each parameter tensor is normalised by its OWN norm; the Sobel kernel is a
    buffer and must never be touched."""
    org = NCAOrganism(layout=SENDER_LAYOUT, hidden=16, grid=8)
    torch.nn.init.normal_(org.fc2.weight, std=0.3)

    x = org.seed_state(2)
    out = org.rollout(x, 4, generator=torch.Generator().manual_seed(0))
    out.square().mean().backward()

    normalize_grads_(org)
    grads = [p.grad for p in org.parameters() if p.grad is not None]
    assert len(grads) == 3, "expected fc1.weight, fc1.bias, fc2.weight"
    for g in grads:
        assert g.norm().item() == pytest.approx(1.0, abs=1e-5)

    names = {n for n, _ in org.named_parameters()}
    assert "kernel" not in names, "perception kernel must be a buffer, not a parameter"
    assert org.kernel.grad is None


def test_grad_norm_is_nan_safe_on_zero_grad():
    org = NCAOrganism(layout=SENDER_LAYOUT, hidden=8, grid=8)
    for p in org.parameters():
        p.grad = torch.zeros_like(p)
    normalize_grads_(org)
    for p in org.parameters():
        assert torch.isfinite(p.grad).all()
        assert p.grad.abs().sum() == 0


# --- death guard --------------------------------------------------------------


def test_death_guard_ignores_a_single_live_cell():
    """A freshly seeded organism has alive_count == 1 but is perfectly healthy.
    Triggering on alive_count would fire on step 1 of every run."""
    org = NCAOrganism(layout=SENDER_LAYOUT, hidden=8, grid=8)
    seed = org.seed_state(2)
    guard = DeathGuard(patience=1)
    assert guard.observe(seed) is False


def test_death_guard_fires_only_after_patience():
    guard = DeathGuard(patience=3)
    dead = torch.zeros(2, SENDER_LAYOUT.total, 8, 8)
    assert guard.observe(dead) is False
    assert guard.observe(dead) is False
    assert guard.observe(dead) is True


def test_death_guard_strikes_reset_on_recovery():
    guard = DeathGuard(patience=3)
    dead = torch.zeros(2, SENDER_LAYOUT.total, 8, 8)
    alive = dead.clone()
    alive[0, ALIVE_CHANNEL, 4, 4] = 1.0

    guard.observe(dead)
    guard.observe(dead)
    assert guard.observe(alive) is False
    assert guard.strikes == 0
    assert guard.observe(dead) is False, "one strike after recovery must not fire"


def test_death_guard_partially_alive_batch_does_not_fire():
    guard = DeathGuard(patience=1)
    batch = torch.zeros(4, SENDER_LAYOUT.total, 8, 8)
    batch[2, ALIVE_CHANNEL, 3, 3] = 1.0  # only one member alive
    assert guard.observe(batch) is False


def test_train_step_raises_death_reset_on_collapse():
    cfg = tiny_cfg()
    device = torch.device("cpu")
    st = build_state(cfg, device=device, weight_seed=0)
    st.guard.patience = 1
    # A rule that annihilates everything: force alpha strongly negative.
    with torch.no_grad():
        st.model.fc2.weight.zero_()
        st.model.fc2.weight[ALIVE_CHANNEL] = -50.0

    target = disk_target(cfg.nca.grid, cfg.target.radius)
    with pytest.raises(DeathReset):
        train_step(st, target, make_rng(0, device), cfg)


# --- loop plumbing ------------------------------------------------------------


def test_rollout_length_is_one_scalar_in_range_and_reproducible():
    cfg = tiny_cfg("train.rollout_min=64", "train.rollout_max=96")
    a = [draw_rollout_len(cfg, make_rng(7, torch.device("cpu"))) for _ in range(3)]
    b = [draw_rollout_len(cfg, make_rng(7, torch.device("cpu"))) for _ in range(3)]
    assert a == b, "same seed must give the same rollout length"
    for n in a:
        assert 64 <= n <= 96 and isinstance(n, int)

    rng = make_rng(0, torch.device("cpu"))
    draws = {draw_rollout_len(cfg, rng) for _ in range(50)}
    assert len(draws) > 5, "rollout length should actually vary across steps"


def test_train_step_reduces_loss_over_a_few_steps():
    cfg = tiny_cfg()
    device = torch.device("cpu")
    st = build_state(cfg, device=device, weight_seed=0)
    rng = make_rng(0, device)
    target = disk_target(cfg.nca.grid, cfg.target.radius)

    first = train_step(st, target, rng, cfg)["loss"]
    for _ in range(30):
        st.step += 1
        rec = train_step(st, target, rng, cfg)
    assert rec["loss"] < first, f"loss did not fall: {first:.4f} -> {rec['loss']:.4f}"


def test_train_step_records_expected_fields():
    cfg = tiny_cfg()
    device = torch.device("cpu")
    st = build_state(cfg, device=device, weight_seed=0)
    rec = train_step(st, disk_target(cfg.nca.grid, cfg.target.radius), make_rng(0, device), cfg)
    for key in ("loss", "lr", "n_steps", "alive_min", "alive_med", "alive_max",
                "frac_dead", "iou", "pool", "damage"):
        assert key in rec, f"missing metric {key}"
    assert rec["pool"] == 0.0, "Growing regime must not report pool usage"


def test_comm_step_carries_vote_losses_for_both_organisms():
    """The comm loss must include the per-cell vote terms -- their absence is
    exactly what dissolved the code in the first comm run (agreement 0.95 -> 0.40
    in 250 steps) -- and the channel-SNR telemetry must be logged."""
    cfg = load_config("configs/comm.yaml", overrides=[
        "device=cpu", "init_from=null", "nca.grid=12", "nca.hidden=16",
        "target.radius=4.0", "train.batch=4", "train.reseed_worst=1",
        "train.damage_best=1", "train.rollout_min=6", "train.rollout_max=8",
        "train.pool_from_step=null", "train.damage_from_step=null",
        "train.checkpoint_every=null", "train.lambda_vote=1.0",
    ])
    device = torch.device("cpu")
    st = build_state(cfg, device=device, weight_seed=1)
    st.step = cfg.train.log_interval - 1  # exercise the gradient diagnostic path
    rec = train_step(st, disk_target(cfg.nca.grid, cfg.target.radius), make_rng(0, device), cfg)

    for key in ("vote_sender", "vote_receiver", "p_top", "agree_recv",
                "grad_vote_task", "grad_morph_task"):
        assert key in rec, f"missing metric {key}"
    assert rec["vote_sender"] > 0 and rec["vote_receiver"] > 0
    assert 0.0 < rec["p_top"] <= 1.0
    assert rec["grad_vote_task"] > 0, "vote gradient must actually reach the sender"
    import math
    assert math.isfinite(rec["loss"])


def test_lr_schedule_steps_at_the_milestone():
    cfg = tiny_cfg("train.lr=2e-3", "train.lr_milestones=[3]", "train.lr_gamma=0.1")
    device = torch.device("cpu")
    st = build_state(cfg, device=device, weight_seed=0)
    target = disk_target(cfg.nca.grid, cfg.target.radius)
    rng = make_rng(0, device)

    seen = []
    for _ in range(5):
        seen.append(train_step(st, target, rng, cfg)["lr"])
        st.step += 1
    # lr is read after sched.step(), so the drop appears once the milestone passes.
    assert seen[0] == pytest.approx(2e-3)
    assert seen[-1] == pytest.approx(2e-4)


# --- runlog -------------------------------------------------------------------


def test_runlog_writes_readable_jsonl(tmp_path):
    with RunLog(tmp_path / "r", {"name": "t", "seed": 0}) as log:
        log.log(step=1, loss=0.5)
        log.log(step=2, loss=0.25)
        log.event("gate", name="G1", passed=False)

    rows = read_metrics(tmp_path / "r")
    assert [r["step"] for r in rows] == [1, 2]
    assert (tmp_path / "r" / "config.yaml").exists()
    assert (tmp_path / "r" / "ckpt").is_dir()

    meta = json.loads((tmp_path / "r" / "meta.json").read_text())
    assert "torch" in meta and "git_sha" in meta

    events = (tmp_path / "r" / "events.jsonl").read_text().strip().splitlines()
    assert json.loads(events[0])["kind"] == "gate"


def test_read_metrics_tolerates_truncated_final_line(tmp_path):
    d = tmp_path / "r"
    with RunLog(d, {"name": "t"}) as log:
        log.log(step=1, loss=1.0)
    with open(d / "metrics.jsonl", "a") as fh:
        fh.write('{"step": 2, "loss": 0.5')  # killed mid-write
    rows = read_metrics(d)
    assert len(rows) == 1 and rows[0]["step"] == 1
