"""The Lewis referential game between two organisms.

Sender sees a referent, Receiver sees nothing but one symbol from a vocabulary of
V, and must name the referent. They share one loss.

Communication is necessary *by construction*: the Receiver has no other input
path. Gate G4 verifies it -- with the message removed, accuracy must fall to
chance. That is what separates a real protocol from a receiver exploiting a prior.

Both agents are NCAs with separate parameters and no shared state. The Sender's
symbol is a referendum over its living cells rather than a network output, and the
Receiver's symbol arrives at a patch of cells and must physically propagate. That
is what makes damage meaningful: it removes electorate, not a module.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from morphos.substrate.nca import NCAOrganism
from morphos.task.broadcast import make_inject_fn, patch_slices, referent_codes
from morphos.task.channel import GumbelChannel
from morphos.task.readout import VotePool


@dataclass
class EpisodeOutput:
    sender_state: Tensor
    receiver_state: Tensor
    symbol_logits: Tensor  # (B, V) sender's pooled vote
    symbol: Tensor  # (B, V) one-hot actually transmitted
    symbol_idx: Tensor  # (B,) integer symbol
    answer_probs: Tensor  # (B, N) receiver's pooled vote
    answer_idx: Tensor  # (B,) receiver's decision
    referents: Tensor  # (B,) ground truth


def make_ear_inject_fn(
    symbol: Tensor, layout, grid: int, t_inject: int, *, patch: int = 3
):
    """Write the one-hot symbol into the Receiver's sensor ('ear') patch.

    Same convention as the sender's sensor: writing simply stops at `t_inject`
    rather than being zeroed, so removal is the absence of a write and not a
    second event the organism could detect.
    """
    ys, xs = patch_slices(grid, patch)

    def inject(x: Tensor, t: int) -> Tensor:
        if t >= t_inject:
            return x
        x = x.clone()
        x[:, layout.sensor, ys, xs] = symbol.view(symbol.shape[0], -1, 1, 1)
        return x

    return inject


def run_episode(
    sender: NCAOrganism,
    receiver: NCAOrganism,
    *,
    referents: Tensor,
    sender_state: Tensor,
    receiver_state: Tensor,
    sender_readout: VotePool,
    receiver_readout: VotePool,
    channel: GumbelChannel,
    rng,
    n_referents: int,
    t_sender: int,
    t_receiver: int,
    t_inject: int,
    grid: int,
    patch: int = 3,
    step: int = 0,
    checkpoint_every: int | None = None,
    hard: bool = True,
    noise: bool = True,
    dropout: float | None = None,
    override_symbol: Tensor | None = None,
) -> EpisodeOutput:
    """One full sender -> symbol -> receiver episode.

    `override_symbol` replaces the transmitted symbol, which is how every causal
    intervention is implemented: remove the message, randomise it, permute the
    vocabulary, or replay one from another episode.
    """
    codes = referent_codes(n_referents, sender.layout.n_sensor, device=referents.device)
    s_inject = make_inject_fn(
        codes[referents], sender.layout, grid, t_inject, patch=patch
    )
    s_out = sender.rollout(
        sender_state, t_sender, generator=rng.update,
        inject_fn=s_inject, checkpoint_every=checkpoint_every,
    )

    s_alive = sender.alive_mask(s_out)
    symbol_logits = sender_readout.logits(
        s_out, s_alive, dropout=dropout, generator=rng.task
    )
    symbol = channel(symbol_logits, generator=rng.task, step=step, hard=hard, noise=noise)
    if override_symbol is not None:
        symbol = override_symbol

    r_inject = make_ear_inject_fn(symbol, receiver.layout, grid, t_inject, patch=patch)
    r_out = receiver.rollout(
        receiver_state, t_receiver, generator=rng.update,
        inject_fn=r_inject, checkpoint_every=checkpoint_every,
    )

    r_alive = receiver.alive_mask(r_out)
    answer_probs = receiver_readout(r_out, r_alive, dropout=dropout, generator=rng.task)

    return EpisodeOutput(
        sender_state=s_out,
        receiver_state=r_out,
        symbol_logits=symbol_logits,
        symbol=symbol,
        symbol_idx=symbol.argmax(-1),
        answer_probs=answer_probs,
        answer_idx=answer_probs.argmax(-1),
        referents=referents,
    )


def task_loss(ep: EpisodeOutput, n_referents: int) -> Tensor:
    """L2 between the receiver's pooled vote and the one-hot referent.

    L2 rather than cross-entropy, for the same reason as the per-cell vote loss:
    CE drives the underlying per-cell logits up without bound, so the residual
    update never decays and the automaton never settles.
    """
    onehot = torch.zeros_like(ep.answer_probs).scatter_(
        -1, ep.referents.unsqueeze(-1), 1.0
    )
    return (ep.answer_probs - onehot).pow(2).sum(-1).mean()


def accuracy(ep: EpisodeOutput) -> float:
    return (ep.answer_idx == ep.referents).float().mean().item()
