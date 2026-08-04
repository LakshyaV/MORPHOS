"""Sample pool for persistence and regeneration training.

The pool is deliberately dumb storage. The ranking / reseed / damage *policy*
lives in the training loop, because that policy is scientific content and this is
a buffer.

Two decisions here are load-bearing:

1. **float32, not float16.** fp16 would save 59 MB of a ~2700 MB budget (<2.5%)
   while costing bit-exact resume, and -- worse -- alpha values near the 0.1
   threshold can flip on quantisation, which makes the alive mask and therefore
   IoU non-deterministic. States round-trip through the pool roughly every
   `pool_size/batch` steps, so quantisation error would compound ~190 times over
   a 6000-step run, producing drift indistinguishable from a dynamics bug. If
   memory ever binds, keep the pool on CPU instead (~1 ms/step to transfer).

2. **Sampling without replacement.** Drawing 32 of 1024 with replacement collides
   ~38% of the time, and a duplicate index makes the write-back an
   order-unspecified scatter -- last-writer-wins with no defined order on MPS.
   That silently breaks the same-device reproducibility this project asserts.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


class SamplePool:
    """A fixed-size buffer of organism states, all initialised to the seed."""

    def __init__(
        self,
        seed_state: Tensor,
        size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if seed_state.dim() != 4 or seed_state.shape[0] != 1:
            raise ValueError(f"seed_state must be (1,C,H,W), got {tuple(seed_state.shape)}")
        self.size = int(size)
        self.device = torch.device(device) if device is not None else seed_state.device
        self.dtype = dtype
        self._seed = seed_state.detach().to(self.device, dtype).clone()
        self._x = self._seed.repeat(self.size, 1, 1, 1).contiguous()

    def __len__(self) -> int:
        return self.size

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._x.shape)

    def sample(self, batch: int, *, generator: torch.Generator) -> tuple[Tensor, Tensor]:
        """-> (idx (B,) long, states (B,C,H,W) fresh copy), sampled WITHOUT replacement."""
        if batch > self.size:
            raise ValueError(f"batch {batch} exceeds pool size {self.size}")
        # argsort(rand) is a permutation, so indices are unique by construction.
        # Uses only the rand primitive already proven reproducible on MPS.
        idx = torch.argsort(
            torch.rand(self.size, generator=generator, device=self.device)
        )[:batch]
        return idx, self._x[idx].clone()

    @torch.no_grad()
    def commit(self, idx: Tensor, states: Tensor) -> None:
        """Write rolled-out states back. Detaches: holding graph tensors here would
        grow memory without bound across training steps."""
        if idx.shape[0] != states.shape[0]:
            raise ValueError(
                f"idx has {idx.shape[0]} entries but states has {states.shape[0]}"
            )
        if idx.unique().numel() != idx.numel():
            raise ValueError(
                "duplicate indices in commit: the scatter would be order-unspecified"
            )
        self._x[idx] = states.detach().to(self.device, self.dtype)

    @torch.no_grad()
    def reset(self) -> None:
        self._x = self._seed.repeat(self.size, 1, 1, 1).contiguous()

    def state_dict(self) -> dict[str, Any]:
        return {"x": self._x.cpu(), "seed": self._seed.cpu(), "size": self.size}

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self._x = d["x"].to(self.device, self.dtype)
        self._seed = d["seed"].to(self.device, self.dtype)
        self.size = int(d["size"])
