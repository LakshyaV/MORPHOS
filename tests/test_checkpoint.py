"""Checkpoint and resume tests.

`test_resume_is_bit_exact` is the strongest determinism assertion in the project:
it validates the model, optimizer, scheduler and all four RNG streams in one shot.
If resume drifts, every multi-day training run becomes unreproducible.
"""

from __future__ import annotations

import pytest
import torch

from morphos.config import load_config
from morphos.io.runlog import RunLog
from morphos.seeding import make_rng
from morphos.task.targets import build_target, target_fingerprint
from morphos.train.checkpoint import config_from, latest, load, save
from morphos.train.loop import build_state, train


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
        "train.log_interval=1000",
        "train.early_stop_on_gates=false",
    ]
    return load_config("configs/base.yaml", overrides=base + list(overrides))


def _target(cfg):
    return build_target(cfg.target.shape, cfg.nca.grid, radius=cfg.target.radius)


def test_save_load_roundtrip(tmp_path):
    cfg = tiny_cfg()
    device = torch.device("cpu")
    st = build_state(cfg, device=device, weight_seed=1)
    rng = make_rng(0, device)

    path = save(
        tmp_path / "c.pt",
        step=7,
        model=st.model,
        optimizer=st.opt,
        scheduler=st.sched,
        rng=rng,
        config={"name": "x"},
        target_fingerprint="abc123",
    )
    assert path.exists()
    assert not path.with_suffix(".pt.tmp").exists(), "temp file must be renamed away"

    fresh = build_state(cfg, device=device, weight_seed=999)
    assert not torch.equal(fresh.model.fc1.weight, st.model.fc1.weight)

    ckpt = load(path, model=fresh.model, optimizer=fresh.opt, scheduler=fresh.sched)
    assert ckpt["step"] == 7
    assert torch.equal(fresh.model.fc1.weight, st.model.fc1.weight)
    assert config_from(ckpt) == {"name": "x"}


def test_load_rejects_a_different_target(tmp_path):
    cfg = tiny_cfg()
    st = build_state(cfg, device=torch.device("cpu"), weight_seed=1)
    save(tmp_path / "c.pt", step=1, model=st.model, target_fingerprint="deadbeef")

    with pytest.raises(ValueError, match="refusing to resume"):
        load(tmp_path / "c.pt", model=st.model, expect_target_fingerprint="different")


def test_rng_state_survives_the_roundtrip(tmp_path):
    device = torch.device("cpu")
    rng = make_rng(3, device)
    torch.rand(5, generator=rng.update)  # advance it off its initial state

    st = build_state(tiny_cfg(), device=device, weight_seed=0)
    save(tmp_path / "c.pt", step=0, model=st.model, rng=rng)

    expected = torch.rand(4, generator=rng.update)

    restored = make_rng(3, device)
    load(tmp_path / "c.pt", rng=restored)
    assert torch.equal(torch.rand(4, generator=restored.update), expected)


def test_latest_finds_last_checkpoint(tmp_path):
    assert latest(tmp_path) is None
    (tmp_path / "ckpt").mkdir()
    (tmp_path / "ckpt" / "last.pt").write_bytes(b"x")
    assert latest(tmp_path) == tmp_path / "ckpt" / "last.pt"


@pytest.mark.slow
def test_resume_is_bit_exact(tmp_path):
    """train(20) must equal train(10) + save + load + train(10), parameter for parameter.

    This exercises the model, the optimizer moments, the LR scheduler and all four
    RNG streams simultaneously. Anything that silently reseeds shows up here.
    """
    target = _target(tiny_cfg())

    straight = train(
        tiny_cfg("train.steps=20"), target, run_dir=tmp_path / "straight"
    )

    first = train(tiny_cfg("train.steps=10"), target, run_dir=tmp_path / "split")
    ckpt = latest(tmp_path / "split")
    assert ckpt is not None, "training must leave a last.pt behind"

    second = train(
        tiny_cfg("train.steps=20"),
        target,
        run_dir=tmp_path / "split",
        resume=ckpt,
    )

    assert first.step == 10
    assert second.step == straight.step == 20

    for (na, pa), (nb, pb) in zip(
        straight.model.named_parameters(), second.model.named_parameters(), strict=True
    ):
        assert na == nb
        assert torch.equal(pa, pb), (
            f"{na} diverged after resume; max delta "
            f"{(pa - pb).abs().max().item():.3e}"
        )


@pytest.mark.slow
def test_training_writes_numbered_and_last_checkpoints(tmp_path):
    cfg = tiny_cfg("train.steps=6", "train.ckpt_interval=3")
    target = _target(cfg)
    with RunLog(tmp_path / "r", {"name": "t"}) as log:
        train(cfg, target, run_dir=tmp_path / "r", log=log)

    ckpts = sorted(p.name for p in (tmp_path / "r" / "ckpt").glob("*.pt"))
    assert "last.pt" in ckpts
    assert "step_000003.pt" in ckpts and "step_000006.pt" in ckpts

    # Numbered checkpoints must stay small: no pool, no optimizer state.
    numbered = (tmp_path / "r" / "ckpt" / "step_000003.pt").stat().st_size
    assert numbered < 2_000_000, f"numbered checkpoint unexpectedly large: {numbered} bytes"

    loaded = load(tmp_path / "r" / "ckpt" / "last.pt")
    assert loaded["step"] == 6
    assert "optimizer" in loaded and "scheduler" in loaded and "rng" in loaded
    assert target_fingerprint(target) == loaded["target_fingerprint"]
