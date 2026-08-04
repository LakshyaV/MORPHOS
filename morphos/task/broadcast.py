"""The broadcast task: get a referent from the sensor patch to every cell.

This is a single-organism precursor to the Lewis game, and it exists because of
the Milestone 2 probe. That probe showed a morphology-trained organism relays an
injected signal only ~5 cells against a body radius of 9 -- and that its loss
never once referenced the sensor channels, so it had no reason to relay anything.

The broadcast task supplies exactly the missing pressure: a referent is injected
at the centre, and *every alive cell* is supervised to vote for it. A cell can
only be right if the referent reached it, so the loss is a direct demand for
propagation.

It is deliberately cheaper than the full Lewis game -- one organism, no discrete
channel, no receiver -- because it isolates the one question that decides whether
Milestone 3 is viable: can this substrate be taught to move information across
itself at all?

Referent codes are bipolar {-1,+1}. With {0,1} the all-zero code injects nothing
and is indistinguishable from no injection.
"""

from __future__ import annotations

import torch
from torch import Tensor

from morphos.substrate.nca import ChannelLayout


def sample_referents(
    batch: int, n_referents: int, *, generator: torch.Generator, device
) -> Tensor:
    """(B,) uniform referent indices."""
    u = torch.rand(batch, generator=generator, device=device)
    return (u * n_referents).long().clamp(max=n_referents - 1)


def referent_codes(n_referents: int, n_bits: int, *, device) -> Tensor:
    """(n_referents, n_bits) bipolar codes, one row per referent."""
    idx = torch.arange(n_referents, device=device).unsqueeze(1)
    bits = (idx >> torch.arange(n_bits, device=device)) & 1
    return bits.float() * 2.0 - 1.0


def patch_slices(grid: int, patch: int = 3) -> tuple[slice, slice]:
    c, r = grid // 2, patch // 2
    return slice(c - r, c + r + 1), slice(c - r, c + r + 1)


def make_inject_fn(
    codes: Tensor,
    layout: ChannelLayout,
    grid: int,
    t_inject: int,
    *,
    patch: int = 3,
    channels: slice | None = None,
):
    """Build an `inject_fn(x, t)` that hard-writes `codes` into the sensor patch.

    Writing stops at `t_inject` rather than zeroing: stopping *is* removal, while
    zeroing would be a second injection event the organism could detect. After
    that the organism must hold the referent in its own state, which is what makes
    damage able to destroy a memory rather than merely interrupt a wire.
    """
    ys, xs = patch_slices(grid, patch)
    ch = channels if channels is not None else layout.sensor

    def inject(x: Tensor, t: int) -> Tensor:
        if t >= t_inject:
            return x
        x = x.clone()
        x[:, ch, ys, xs] = codes.view(codes.shape[0], -1, 1, 1)
        return x

    return inject


def build_episode_inject(
    referents: Tensor, layout: ChannelLayout, grid: int, t_inject: int, **kw
):
    """Convenience: referent indices -> injection function."""
    codes = referent_codes(
        int(referents.max().item()) + 1 if referents.numel() else 1,
        layout.n_sensor,
        device=referents.device,
    )
    n_ref = codes.shape[0]
    all_codes = referent_codes(max(n_ref, int(referents.max().item()) + 1),
                               layout.n_sensor, device=referents.device)
    return make_inject_fn(all_codes[referents], layout, grid, t_inject, **kw)
