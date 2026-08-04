"""Growing Neural Cellular Automaton substrate (Mordvintsev et al., Distill 2020).

Channel layout is explicit and role-tagged rather than positional magic numbers,
because sender and receiver carry different sensor/vote widths:

    sender    RGBA(4) + sensor(3  referent bits) + vote(8 symbols) + hidden(13) = 28
    receiver  RGBA(4) + sensor(8  symbol onehot) + vote(8 answers) + hidden(8)  = 28

Two details in here are easy to get wrong and fail *silently*:

1. ``groups=C`` conv assigns output channels to groups in contiguous blocks of
   3, so the perception layout is [id_0, sx_0, sy_0, id_1, sx_1, sy_1, ...] and
   NOT [id_0..id_{C-1}, sx_0..., sy_0...]. Wrong ordering still trains, just
   worse. ``test_sobel_on_ramp`` is what catches it.
2. Stochastic fire masks are pre-generated for the whole rollout rather than
   drawn inside ``step``. Gradient checkpointing recomputes the forward pass,
   which would otherwise advance our explicit generator and silently desynchronise
   the recomputed masks from the originals.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

# Fixed (non-learned) perception filters. conv2d is cross-correlation, so no flip.
_IDENTITY = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
_SOBEL_Y = _SOBEL_X.T.contiguous()

ALIVE_CHANNEL = 3
ALIVE_THRESHOLD = 0.1


@dataclass(frozen=True)
class ChannelLayout:
    """Role-tagged channel budget. RGBA is always the first four channels."""

    n_sensor: int
    n_vote: int
    n_hidden: int
    n_rgba: int = 4

    @property
    def total(self) -> int:
        return self.n_rgba + self.n_sensor + self.n_vote + self.n_hidden

    @property
    def rgba(self) -> slice:
        return slice(0, self.n_rgba)

    @property
    def sensor(self) -> slice:
        return slice(self.n_rgba, self.n_rgba + self.n_sensor)

    @property
    def vote(self) -> slice:
        lo = self.n_rgba + self.n_sensor
        return slice(lo, lo + self.n_vote)

    @property
    def hidden(self) -> slice:
        lo = self.n_rgba + self.n_sensor + self.n_vote
        return slice(lo, lo + self.n_hidden)


# N=8 referents as 3 bits, V=8 symbols.
SENDER_LAYOUT = ChannelLayout(n_sensor=3, n_vote=8, n_hidden=13)
RECEIVER_LAYOUT = ChannelLayout(n_sensor=8, n_vote=8, n_hidden=8)


def build_perception_kernel(channels: int) -> Tensor:
    """-> (3C, 1, 3, 3) laid out for ``groups=C`` as [id_c, sx_c, sy_c] per channel."""
    base = torch.stack([_IDENTITY, _SOBEL_X, _SOBEL_Y]).unsqueeze(1)  # (3, 1, 3, 3)
    return base.repeat(channels, 1, 1, 1).contiguous()  # (3C, 1, 3, 3)


class NCAOrganism(nn.Module):
    """A cell-autonomous substrate: shared local update rule, no central module."""

    has_morphology = True

    def __init__(
        self,
        layout: ChannelLayout = SENDER_LAYOUT,
        hidden: int = 128,
        grid: int = 32,
        fire_rate: float = 0.5,
        alive_threshold: float = ALIVE_THRESHOLD,
    ) -> None:
        super().__init__()
        self.layout = layout
        self.channels = layout.total
        self.grid = grid
        self.fire_rate = fire_rate
        self.alive_threshold = alive_threshold
        self.state_shape = (self.channels, grid, grid)

        self.register_buffer("kernel", build_perception_kernel(self.channels))
        self.fc1 = nn.Conv2d(3 * self.channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, self.channels, kernel_size=1, bias=False)
        nn.init.zeros_(self.fc2.weight)  # start as a do-nothing update rule

    # -- pieces -------------------------------------------------------------

    def perceive(self, x: Tensor) -> Tensor:
        """(B,C,H,W) -> (B,3C,H,W). Circular-free: zero padding, as in the paper."""
        return F.conv2d(x, self.kernel, padding=1, groups=self.channels)

    def alive_mask(self, x: Tensor) -> Tensor:
        """(B,C,H,W) -> (B,1,H,W) float. A cell is alive if any 3x3 neighbour has alpha>thr."""
        alpha = x[:, ALIVE_CHANNEL : ALIVE_CHANNEL + 1]
        pooled = F.max_pool2d(alpha, kernel_size=3, stride=1, padding=1)
        return (pooled > self.alive_threshold).to(x.dtype)

    def rgba(self, x: Tensor) -> Tensor:
        return x[:, self.layout.rgba]

    def sample_fire_mask(
        self, batch: int, generator: torch.Generator, device: torch.device
    ) -> Tensor:
        """Per-cell stochastic update mask, decoupling cells from a global clock."""
        r = torch.rand(batch, 1, self.grid, self.grid, generator=generator, device=device)
        return (r <= self.fire_rate).float()

    # -- dynamics -----------------------------------------------------------

    def step(self, x: Tensor, fire_mask: Tensor) -> Tensor:
        """One synchronous-in-form, stochastic-in-effect residual update."""
        pre_alive = self.alive_mask(x)
        dx = self.fc2(F.relu(self.fc1(self.perceive(x)))) * fire_mask
        x = x + dx
        return x * (pre_alive * self.alive_mask(x))

    def rollout(
        self,
        x: Tensor,
        n_steps: int,
        *,
        generator: torch.Generator,
        inject_fn=None,
        checkpoint_every: int | None = None,
        fire_masks: list[Tensor] | None = None,
    ) -> Tensor:
        """Run ``n_steps`` updates.

        ``inject_fn(x, t) -> x`` is applied before each step (sensor/ear writes).
        ``checkpoint_every`` trades ~1.33x compute for ~4x less activation memory;
        required on 8 GB for rollouts beyond ~24 steps.

        ``fire_masks`` supplies the stochastic update masks explicitly instead of
        drawing them. The probe needs this: to isolate the effect of an injected
        code from update noise, every code must experience the *same* fire mask,
        otherwise between-code variance is contaminated by mask differences.
        """
        if fire_masks is not None:
            if len(fire_masks) < n_steps:
                raise ValueError(
                    f"need {n_steps} fire masks, got {len(fire_masks)}"
                )
            masks = fire_masks
        else:
            masks = [
                self.sample_fire_mask(x.shape[0], generator, x.device)
                for _ in range(n_steps)
            ]

        if checkpoint_every is None:
            for t in range(n_steps):
                if inject_fn is not None:
                    x = inject_fn(x, t)
                x = self.step(x, masks[t])
            return x

        for lo in range(0, n_steps, checkpoint_every):
            chunk = list(range(lo, min(lo + checkpoint_every, n_steps)))

            def run_chunk(state: Tensor, _chunk=chunk) -> Tensor:
                for t in _chunk:
                    if inject_fn is not None:
                        state = inject_fn(state, t)
                    state = self.step(state, masks[t])
                return state

            x = checkpoint(run_chunk, x, use_reentrant=False)
        return x

    # -- initial conditions -------------------------------------------------

    def seed_state(self, batch: int, *, device: torch.device | str = "cpu") -> Tensor:
        """A single alive cell at the grid centre; everything else is void."""
        x = torch.zeros(batch, *self.state_shape, device=device)
        c = self.grid // 2
        x[:, ALIVE_CHANNEL:, c, c] = 1.0  # alpha + sensor + vote + hidden
        return x
