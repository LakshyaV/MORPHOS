"""Consensus readout tests.

`test_readout_is_parameter_free` and `test_no_cell_can_dominate` guard the
project's central anti-centralisation claim. If either fails, the "distributed
computation" framing is unsupported and the paper's premise goes with it.
"""

from __future__ import annotations

import pytest
import torch

from morphos.substrate.nca import SENDER_LAYOUT
from morphos.task.readout import (
    VotePool,
    agreement,
    cell_argmax,
    consensus,
    per_cell_vote_loss,
    quorum_fraction,
)

C = SENDER_LAYOUT.total
POOL = VotePool(channels=SENDER_LAYOUT.vote, n_out=8)


def state_with(votes: torch.Tensor, grid: int = 8) -> torch.Tensor:
    """Build a state whose vote channels are `votes` (B, 8, H, W)."""
    b = votes.shape[0]
    x = torch.zeros(b, C, grid, grid)
    x[:, SENDER_LAYOUT.vote] = votes
    return x


def all_alive(b: int, grid: int = 8) -> torch.Tensor:
    return torch.ones(b, 1, grid, grid)


def test_readout_is_parameter_free():
    """Zero learned parameters is the whole anti-centralisation argument."""
    assert POOL.n_params == 0
    assert not isinstance(POOL, torch.nn.Module)
    assert not any(isinstance(v, torch.nn.Parameter) for v in vars(POOL).values())


def test_cell_votes_lie_on_the_simplex():
    v = torch.randn(3, 8, 8, 8) * 5
    votes = POOL.cell_votes(state_with(v))
    assert torch.allclose(votes.sum(dim=1), torch.ones(3, 8, 8), atol=1e-5)
    assert (votes >= 0).all()


def test_no_cell_can_dominate():
    """A single cell shouting arbitrarily loudly must not swing the result.

    Without per-cell softmax this is exactly the failure mode gradient descent
    prefers: the 9 sensor cells emit an enormous vote and the rest of the body is
    computationally irrelevant.
    """
    grid = 8
    v = torch.zeros(1, 8, grid, grid)
    v[0, 3] = 1.0  # every cell mildly prefers class 3
    v[0, 5, 0, 0] = 1e6  # one cell screams for class 5

    pooled = POOL(state_with(v, grid), all_alive(1, grid))
    assert pooled.argmax().item() == 3, "one cell overrode the entire body"

    # The precise guarantee: because each cell's vote lies on the simplex, one
    # cell contributes at most 1/n to any class, so silencing the loudest cell can
    # move the pooled distribution by at most 1/n. (The 0.117 mass on class 5 is
    # the other 63 cells' softmax tails, not this cell's doing.)
    quiet = v.clone()
    quiet[0, 5, 0, 0] = 0.0
    pooled_quiet = POOL(state_with(quiet, grid), all_alive(1, grid))
    shift = (pooled - pooled_quiet).abs().max().item()
    assert shift <= 1.0 / (grid * grid) + 1e-6, (
        f"one cell moved the pooled result by {shift:.4f}, exceeding the 1/n bound"
    )


def test_raw_logit_pooling_is_vulnerable_by_contrast():
    """The same input DOES break a raw-logit mean -- documenting why the softmax
    variant is the default rather than an optional refinement."""
    grid = 8
    v = torch.zeros(1, 8, grid, grid)
    v[0, 3] = 1.0
    v[0, 5, 0, 0] = 1e6

    raw = VotePool(channels=SENDER_LAYOUT.vote, n_out=8, per_cell_softmax=False)
    assert raw(state_with(v, grid), all_alive(1, grid)).argmax().item() == 5


def test_pooling_ignores_dead_cells():
    grid = 8
    v = torch.zeros(2, 8, grid, grid)
    v[:, 2] = 10.0  # alive region votes class 2
    v[:, 7, 4:, :] = 10.0  # dead region would vote class 7

    alive = torch.ones(2, 1, grid, grid)
    alive[:, :, 4:, :] = 0.0

    assert POOL(state_with(v, grid), alive).argmax(dim=-1).tolist() == [2, 2]


def test_pooled_result_is_a_distribution():
    v = torch.randn(4, 8, 8, 8)
    p = POOL(state_with(v), all_alive(4))
    assert p.shape == (4, 8)
    assert torch.allclose(p.sum(-1), torch.ones(4), atol=1e-5)


def test_decision_is_deterministic():
    """Eval must use argmax, never a stochastic draw: otherwise a drop in
    confidence looks identical to a change in meaning."""
    v = torch.randn(4, 8, 8, 8)
    s, a = state_with(v), all_alive(4)
    assert torch.equal(POOL.decision(s, a), POOL.decision(s, a))


def test_dropout_changes_the_electorate_but_keeps_a_valid_result():
    v = torch.randn(4, 8, 8, 8)
    s, a = state_with(v), all_alive(4)
    g = torch.Generator().manual_seed(0)
    p = POOL(s, a, dropout=0.5, generator=g)
    assert p.shape == (4, 8)
    assert torch.allclose(p.sum(-1), torch.ones(4), atol=1e-5)
    assert torch.isfinite(p).all()


def test_dropout_never_divides_by_an_empty_electorate():
    v = torch.randn(2, 8, 4, 4)
    p = POOL(state_with(v, 4), all_alive(2, 4), dropout=1.0,
             generator=torch.Generator().manual_seed(0))
    assert torch.isfinite(p).all()
    assert torch.allclose(p.sum(-1), torch.ones(2), atol=1e-5)


def test_per_cell_loss_is_zero_at_perfect_consensus():
    grid = 8
    v = torch.full((2, 8, grid, grid), -20.0)
    v[0, 3] = 20.0
    v[1, 6] = 20.0
    loss = per_cell_vote_loss(POOL, state_with(v, grid), all_alive(2, grid),
                              torch.tensor([3, 6]))
    assert loss.item() < 1e-4


def test_per_cell_loss_punishes_a_correct_pool_with_dissenting_cells():
    """The point of per-cell supervision: pooling to the right answer is not
    enough, every cell must individually know it. That is what forces the referent
    to actually reach distant cells."""
    grid = 8
    v = torch.zeros(2, 8, grid, grid)
    v[:, 3] = 20.0
    v[:, 3, 4:, :] = -20.0  # half the body dissents
    v[:, 7, 4:, :] = 20.0

    pooled_ok = POOL(state_with(v, grid), all_alive(2, grid)).argmax(-1)
    loss = per_cell_vote_loss(POOL, state_with(v, grid), all_alive(2, grid),
                              torch.tensor([3, 3]))
    assert pooled_ok.tolist() == [3, 3], "pool should still be correct"
    assert loss.item() > 0.4, "but the loss must see the dissent"


def test_per_cell_loss_ignores_dead_cells():
    grid = 8
    v = torch.full((1, 8, grid, grid), -20.0)
    v[0, 2] = 20.0
    v[0, 2, 4:, :] = -20.0  # wrong, but dead
    v[0, 5, 4:, :] = 20.0

    alive = torch.ones(1, 1, grid, grid)
    alive[:, :, 4:, :] = 0.0
    loss = per_cell_vote_loss(POOL, state_with(v, grid), alive, torch.tensor([2]))
    assert loss.item() < 1e-4


def test_consensus_and_agreement():
    grid = 8
    v = torch.full((1, 8, grid, grid), -20.0)
    v[0, 1] = 20.0
    v[0, 1, :2, :] = -20.0  # 16 of 64 cells dissent
    v[0, 4, :2, :] = 20.0

    s, a = state_with(v, grid), all_alive(1, grid)
    assert consensus(POOL, s, a).item() == pytest.approx(48 / 64)
    assert agreement(POOL, s, a).item() == pytest.approx(48 / 64)
    assert cell_argmax(POOL, s)[0, 0, 0].item() == 4


def test_quorum_is_high_when_distributed_and_low_for_a_megaphone():
    grid = 8
    n = grid * grid

    uniform = torch.full((1, 8, grid, grid), -20.0)
    uniform[0, 2] = 20.0
    q_dist = quorum_fraction(POOL, state_with(uniform, grid), all_alive(1, grid)).item()

    # One cell decides; the rest are indifferent.
    mega = torch.zeros(1, 8, grid, grid)
    mega[0, 5, 0, 0] = 20.0
    q_mega = quorum_fraction(POOL, state_with(mega, grid), all_alive(1, grid)).item()

    assert q_dist > 0.4, f"uniform consensus should need a large quorum, got {q_dist}"
    assert q_mega <= 2 / n + 1e-9, f"a single decisive cell should give ~1/n, got {q_mega}"
    assert q_mega < q_dist
