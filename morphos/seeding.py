"""Deterministic RNG management.

Four *separate* generator streams. With a single global stream, adding an eval
probe mid-training shifts every subsequent NCA fire mask and the run stops
reproducing. Model code must never call the global ``torch.rand``.

CPU and MPS produce different streams from the same seed, so bit-identical
results across devices are impossible by construction. Device is therefore part
of run identity; determinism is asserted same-device only.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class RNG:
    """Explicit generator bundle, one stream per source of randomness."""

    update: torch.Generator  # NCA stochastic fire mask
    task: torch.Generator  # referent sampling, Gumbel noise
    damage: torch.Generator  # damage placement
    pool: torch.Generator  # sample-pool index sampling

    @property
    def device(self) -> torch.device:
        return self.update.device


_STREAMS = ("update", "task", "damage", "pool")


def make_rng(seed: int, device: torch.device | str = "cpu") -> RNG:
    """Build the four-stream bundle. Offsets keep the streams disjoint."""
    device = torch.device(device)
    gens = {
        name: torch.Generator(device=device).manual_seed(seed * 1000 + i)
        for i, name in enumerate(_STREAMS)
    }
    return RNG(**gens)


def seed_everything(seed: int) -> None:
    """Seed the global interpreter-level RNGs (library calls we do not control)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rng_state(rng: RNG) -> dict[str, torch.Tensor]:
    """Capture generator states for checkpointing."""
    return {name: getattr(rng, name).get_state() for name in _STREAMS}


def load_rng_state(rng: RNG, state: dict[str, torch.Tensor]) -> None:
    """Restore generator states from a checkpoint, in place."""
    for name in _STREAMS:
        getattr(rng, name).set_state(state[name])
