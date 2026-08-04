"""Tests for the sample pool, damage operators, and the gates."""

from __future__ import annotations

import pytest
import torch

from morphos.damage.ops import (
    apply_damage,
    disk_keep_mask,
    kill_fraction,
    sample_on_body,
    sample_training_masks,
    solve_radius_for_fraction,
)
from morphos.eval.gates import GateResult, check_g1
from morphos.eval.metrics import body_mask
from morphos.substrate.nca import ALIVE_CHANNEL, SENDER_LAYOUT, NCAOrganism
from morphos.substrate.pool import SamplePool
from morphos.task.targets import disk_target
from morphos.train.losses import per_sample_rgba_mse

C = SENDER_LAYOUT.total


def make_pool(size=64, grid=12):
    seed = torch.zeros(1, C, grid, grid)
    seed[:, ALIVE_CHANNEL:, grid // 2, grid // 2] = 1.0
    return SamplePool(seed, size)


# --- pool ---------------------------------------------------------------------


def test_pool_initialises_every_slot_to_the_seed():
    p = make_pool(size=8)
    idx, states = p.sample(8, generator=torch.Generator().manual_seed(0))
    assert states.shape == (8, C, 12, 12)
    for i in range(1, 8):
        assert torch.equal(states[0], states[i])


def test_pool_samples_without_replacement():
    """A duplicate index makes the write-back an order-unspecified scatter, which
    silently breaks same-device reproducibility. Drawing 32-of-1024 with
    replacement collides ~38% of the time."""
    p = make_pool(size=1024)
    g = torch.Generator().manual_seed(0)
    for _ in range(50):
        idx, _ = p.sample(32, generator=g)
        assert idx.unique().numel() == 32, "pool returned duplicate indices"


def test_pool_sampling_is_reproducible():
    p = make_pool(size=64)
    a, _ = p.sample(8, generator=torch.Generator().manual_seed(3))
    b, _ = p.sample(8, generator=torch.Generator().manual_seed(3))
    assert torch.equal(a, b)


def test_pool_commit_writes_to_the_right_slots_without_aliasing():
    p = make_pool(size=16)
    idx, states = p.sample(4, generator=torch.Generator().manual_seed(0))
    states[:, 0] = torch.arange(1, 5, dtype=torch.float32).view(4, 1, 1)
    p.commit(idx, states)

    all_idx, all_states = p.sample(16, generator=torch.Generator().manual_seed(1))
    lookup = {int(i): s for i, s in zip(all_idx, all_states, strict=True)}
    for k, i in enumerate(idx.tolist()):
        assert lookup[i][0].mean().item() == pytest.approx(k + 1)
    untouched = [i for i in range(16) if i not in idx.tolist()]
    for i in untouched:
        assert lookup[i][0].abs().sum() == 0, "commit corrupted an unrelated slot"


def test_pool_sample_returns_a_copy_not_a_view():
    p = make_pool(size=8)
    idx, states = p.sample(4, generator=torch.Generator().manual_seed(0))
    states.mul_(99.0)
    _, again = p.sample(8, generator=torch.Generator().manual_seed(1))
    assert again.abs().max().item() <= 1.0, "sample() aliased pool storage"


def test_pool_commit_rejects_duplicate_indices():
    p = make_pool(size=8)
    with pytest.raises(ValueError, match="duplicate indices"):
        p.commit(torch.tensor([1, 1]), torch.zeros(2, C, 12, 12))


def test_pool_state_dict_roundtrip():
    p = make_pool(size=8)
    idx, states = p.sample(4, generator=torch.Generator().manual_seed(0))
    p.commit(idx, states + 5.0)
    sd = p.state_dict()

    q = make_pool(size=8)
    q.load_state_dict(sd)
    _, a = p.sample(8, generator=torch.Generator().manual_seed(2))
    _, b = q.sample(8, generator=torch.Generator().manual_seed(2))
    assert torch.equal(a, b)


def test_pool_is_float32():
    """fp16 would flip alpha values near the 0.1 threshold, making the alive mask
    and therefore IoU non-deterministic across round-trips."""
    p = make_pool()
    _, s = p.sample(2, generator=torch.Generator().manual_seed(0))
    assert s.dtype == torch.float32


def test_ranking_permutes_idx_and_states_together():
    """The bug this guards: permuting only `states` writes every rolled-out result
    back to the WRONG pool slot. The pool then just looks noisy, the loss plateaus,
    and it costs days to find.
    """
    grid = 12
    target = disk_target(grid, 4.0)
    p = make_pool(size=16, grid=grid)
    idx, x0 = p.sample(4, generator=torch.Generator().manual_seed(0))
    # Give each sample a distinct, known loss.
    for k in range(4):
        x0[k, :4] = target * (1.0 - 0.25 * k)
    x0_marked = x0.clone()
    for k in range(4):
        x0_marked[k, ALIVE_CHANNEL + 1] = float(k)  # identity tag

    losses = per_sample_rgba_mse(x0_marked, target)
    order = losses.argsort(descending=True)
    idx_p, x_p = idx[order], x0_marked[order]

    assert losses[order][0] == losses.max(), "slot 0 must be the WORST sample"
    assert losses[order][-1] == losses.min(), "last slot must be the BEST sample"
    # The identity tag must travel with the index.
    for pos, orig in enumerate(order.tolist()):
        assert x_p[pos, ALIVE_CHANNEL + 1].mean().item() == pytest.approx(float(orig))
        assert idx_p[pos] == idx[orig]


# --- damage -------------------------------------------------------------------


def test_disk_keep_mask_geometry():
    keep = disk_keep_mask(16, torch.tensor([[8.0, 8.0]]), torch.tensor([3.0]))
    assert keep.shape == (1, 1, 16, 16)
    assert keep[0, 0, 8, 8] == 0.0  # centre destroyed
    assert keep[0, 0, 0, 0] == 1.0  # far corner kept
    assert keep[0, 0, 8, 12] == 1.0  # outside radius 3


def test_apply_damage_zeroes_all_channels_and_spares_the_rest():
    x = torch.rand(2, C, 16, 16) + 1.0
    keep = disk_keep_mask(16, torch.tensor([[8.0, 8.0], [4.0, 4.0]]), torch.tensor([3.0, 2.0]))
    before = x.clone()
    out = apply_damage(x, keep)

    assert torch.equal(x, before), "apply_damage mutated its input"
    killed = keep[0, 0] == 0
    assert out[0, :, killed].abs().sum() == 0, "killed cells must be zero in EVERY channel"
    assert torch.equal(out[0][:, ~killed], before[0][:, ~killed]), "survivors changed"


def test_training_masks_are_in_the_upstream_range():
    g = torch.Generator().manual_seed(0)
    keep = sample_training_masks(64, 32, generator=g)
    assert keep.shape == (64, 1, 32, 32)
    destroyed = (1 - keep).flatten(1).sum(dim=1)
    assert (destroyed > 0).all(), "every training mask should destroy something"
    # radius in [0.1,0.4]*16 = [1.6, 6.4] cells -> area between ~8 and ~129 cells
    assert destroyed.max().item() < 200


def test_sample_on_body_lands_on_alive_cells():
    alive = torch.zeros(4, 1, 16, 16, dtype=torch.bool)
    alive[:, :, 10:14, 10:14] = True
    centres = sample_on_body(alive, generator=torch.Generator().manual_seed(0))
    assert centres.shape == (4, 2)
    for cy, cx in centres.tolist():
        assert alive[0, 0, int(cy), int(cx)], f"centre ({cy},{cx}) is not on the body"


def test_sample_on_body_survives_an_empty_body():
    alive = torch.zeros(2, 1, 16, 16, dtype=torch.bool)
    centres = sample_on_body(alive, generator=torch.Generator().manual_seed(0))
    assert torch.equal(centres, torch.tensor([[8.0, 8.0], [8.0, 8.0]]))


def test_kill_fraction_hits_the_requested_severity():
    """Severity is a fraction of ALIVE cells, not a radius: a fixed radius would be
    a different severity at every body size and regeneration state."""
    t = disk_target(32, 10.0)
    x = torch.zeros(16, C, 32, 32)
    x[:, :4] = t

    for f in (0.10, 0.30, 0.60):
        damaged, info = kill_fraction(x, f, generator=torch.Generator().manual_seed(0))
        achieved = info["achieved_f"]
        assert (achieved - f).abs().max().item() <= 0.02, (
            f"target {f}, achieved {achieved.tolist()}"
        )
        # Recompute independently of the solver.
        n_before = body_mask(x).flatten(1).sum(1).float()
        n_after = body_mask(damaged).flatten(1).sum(1).float()
        indep = (n_before - n_after) / n_before
        assert torch.allclose(indep.double(), achieved.double(), atol=0.02)


def test_kill_fraction_reports_centre_coverage():
    t = disk_target(32, 10.0)
    x = torch.zeros(8, C, 32, 32)
    x[:, :4] = t
    _, info = kill_fraction(x, 0.5, generator=torch.Generator().manual_seed(0))
    assert info["covers_centre"].dtype == torch.bool
    assert info["covers_centre"].shape == (8,)


def test_solve_radius_is_monotone_in_target():
    t = disk_target(32, 10.0)
    x = torch.zeros(4, C, 32, 32)
    x[:, :4] = t
    alive = body_mask(x)
    centres = torch.full((4, 2), 16.0)

    r_small, _ = solve_radius_for_fraction(alive, centres, 0.1)
    r_big, _ = solve_radius_for_fraction(alive, centres, 0.5)
    assert (r_big > r_small).all()


# --- gates --------------------------------------------------------------------


def test_gate_result_reports_readably():
    r = GateResult("G_test", False, {"iou": 0.5}, {"iou": ">= 0.9"})
    text = r.report()
    assert "FAIL" in text and "iou" in text and ">= 0.9" in text
    assert r.as_dict()["passed"] is False


def test_g1_fails_an_untrained_model_and_does_not_train_it():
    """An untrained NCA has a zero-init output layer, so it stays a single seed
    cell: G1 must fail it, and must leave its weights untouched."""
    org = NCAOrganism(layout=SENDER_LAYOUT, hidden=16, grid=16)
    target = disk_target(16, 5.0)
    before = org.fc1.weight.clone()

    res = check_g1(
        org, target, device=torch.device("cpu"),
        n_rollouts=8, batch=8, t_grow=8, t_persist=16,
    )
    assert isinstance(res, GateResult)
    assert res.passed is False
    assert res.metrics["iou_64_mean"] < 0.9
    assert torch.equal(org.fc1.weight, before), "gate mutated the model"
