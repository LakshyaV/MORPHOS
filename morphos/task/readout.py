"""Parameter-free consensus readout.

This module carries the project's central anti-centralisation claim, so its
design choices are worth stating explicitly.

`softmax(mean over cells)` is parameter-free but NOT decentralised: a global mean
has a global receptive field and instantaneous propagation, and gradient descent
actively prefers the centralised solution. Getting a referent to distant cells
costs ~64 steps of attenuating backprop; having the 9 sensor cells emit an
enormous vote costs two. The shortest gradient path is a megaphone.

Three mechanisms push against that, in increasing order of force:

1. **Per-cell softmax before pooling.** Each cell emits a point on the simplex, so
   the pooled result is a bounded average of distributions and no single cell's
   influence can exceed 1/|A|. This kills the megaphone *mechanically* rather than
   hoping a diagnostic catches it afterwards.
2. **Per-cell loss.** Every alive cell is supervised toward the correct answer, so
   consensus becomes an emergent, measured property rather than an imposed
   aggregation. This is the actual design of Randazzo et al.'s self-classifying
   MNIST, and it is also -- per the Milestone 2 probe -- the only thing that
   creates pressure for the referent to reach distant cells at all.
3. **Readout cell-dropout.** Pooling over a random subset during training is the
   one mechanism that supplies gradient pressure toward electorate-invariance,
   which is precisely the property the damage experiment depends on.

The loss is L2 against a one-hot target, never cross-entropy. Randazzo et al.
found CE on per-cell logit channels drives unbounded logit growth, so the residual
update never decays and the automaton never settles into a quiescent state.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

EPS = 1e-8


@dataclass(frozen=True)
class VotePool:
    """Parameter-free pooled consensus over designated vote channels."""

    channels: slice
    n_out: int
    temperature: float = 1.0
    per_cell_softmax: bool = True

    n_params: int = 0  # asserted in tests; the central claim depends on it

    def cell_votes(self, state: Tensor) -> Tensor:
        """(B,C,H,W) -> (B,n_out,H,W). Each cell's opinion, on the simplex."""
        z = state[:, self.channels]
        if not self.per_cell_softmax:
            return z
        return F.softmax(z / self.temperature, dim=1)

    def __call__(
        self,
        state: Tensor,
        alive: Tensor,
        *,
        dropout: float | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """-> (B, n_out) pooled probability vector over alive cells.

        `dropout` pools over a random fraction of alive cells instead of all of
        them, which is what trains electorate-invariance.
        """
        votes = self.cell_votes(state)
        w = alive.to(votes.dtype)

        if dropout is not None and dropout > 0:
            keep = (
                torch.rand(w.shape, generator=generator, device=w.device) >= dropout
            ).to(w.dtype)
            w = w * keep
            # If dropout emptied a sample entirely, fall back to the full electorate
            # rather than dividing by zero.
            empty = w.flatten(1).sum(1) == 0
            if empty.any():
                w[empty] = alive.to(votes.dtype)[empty]

        num = (votes * w).flatten(2).sum(-1)
        den = w.flatten(2).sum(-1).clamp(min=EPS)
        return num / den

    def logits(self, state: Tensor, alive: Tensor, **kw) -> Tensor:
        """Pooled result as logits, for a Gumbel channel."""
        return (self(state, alive, **kw) + EPS).log()

    def decision(self, state: Tensor, alive: Tensor) -> Tensor:
        """(B,) deterministic argmax. Eval uses this -- never a stochastic draw --
        so that a change in confidence cannot masquerade as a change in meaning."""
        return self(state, alive).argmax(dim=-1)


def per_cell_vote_loss(
    pool: VotePool, state: Tensor, alive: Tensor, target_idx: Tensor
) -> Tensor:
    """Mean L2 between each alive cell's vote and the one-hot target.

    L2 rather than cross-entropy: CE on per-cell logits grows them without bound,
    which stops the residual update decaying and prevents the automaton settling.
    """
    votes = pool.cell_votes(state)
    onehot = F.one_hot(target_idx, pool.n_out).to(votes.dtype)
    err = (votes - onehot[:, :, None, None]).pow(2).sum(dim=1, keepdim=True)
    w = alive.to(votes.dtype)
    return (err * w).flatten(1).sum(1).div(w.flatten(1).sum(1).clamp(min=EPS)).mean()


# --- consensus diagnostics ----------------------------------------------------


def cell_argmax(pool: VotePool, state: Tensor) -> Tensor:
    """(B,H,W) each cell's private opinion."""
    return pool.cell_votes(state).argmax(dim=1)


def consensus(pool: VotePool, state: Tensor, alive: Tensor) -> Tensor:
    """(B,) plurality fraction: how much of the body agrees with the majority.

    Randazzo et al. report 88.1% of samples reaching FULL agreement; this is the
    comparable quantity.
    """
    a = alive[:, 0].to(torch.bool)
    op = cell_argmax(pool, state)
    b = op.shape[0]
    out = torch.zeros(b, dtype=torch.float64)
    for i in range(b):
        sel = op[i][a[i]]
        if sel.numel() == 0:
            continue
        out[i] = (torch.bincount(sel, minlength=pool.n_out).max().float() / sel.numel()).cpu().double()
    return out


def agreement(pool: VotePool, state: Tensor, alive: Tensor) -> Tensor:
    """(B,) fraction of alive cells whose own argmax matches the pooled decision."""
    a = alive[:, 0].to(torch.bool)
    op = cell_argmax(pool, state)
    dec = pool.decision(state, alive)
    b = op.shape[0]
    out = torch.zeros(b, dtype=torch.float64)
    for i in range(b):
        sel = op[i][a[i]]
        if sel.numel() == 0:
            continue
        out[i] = (sel == dec[i]).float().mean().cpu().double()
    return out


def quorum_fraction(pool: VotePool, state: Tensor, alive: Tensor) -> Tensor:
    """(B,) smallest fraction of cells whose removal flips the pooled decision.

    Computed exactly rather than by search: for a mean pool, dropping the cells
    that most favour the winner is the worst case, so sorting by per-cell margin
    and walking down gives the answer in one pass.

    ~0.5 means genuinely distributed. ~0.02 means a megaphone. This is one number
    that belongs in the abstract.
    """
    votes = pool.cell_votes(state)
    a = alive[:, 0].to(torch.bool)
    dec = pool.decision(state, alive)
    b = votes.shape[0]
    out = torch.zeros(b, dtype=torch.float64)

    for i in range(b):
        v = votes[i][:, a[i]]  # (n_out, n_alive)
        n = v.shape[1]
        if n == 0:
            continue
        win = dec[i]
        rival = v.sum(dim=1)
        rival[win] = -float("inf")
        r = int(rival.argmax())
        margin = (v[win] - v[r]).sort(descending=True).values  # drop best first
        remaining = margin.sum() - margin.cumsum(0)
        flipped = (remaining <= 0).nonzero()
        k = int(flipped[0].item()) + 1 if flipped.numel() else n
        out[i] = float(k) / float(n)
    return out
