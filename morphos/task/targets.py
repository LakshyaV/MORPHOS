"""Procedural target morphologies.

The target body is **referent-independent**: one fixed shape, identical for every
referent. That is what makes the morphology-vs-semantics dissociation well-posed
-- if the body varied with the referent, morphological and semantic recovery
would be the same variable and hypotheses H1/H3 would be vacuous.

The primary target is a filled disk rather than the emoji used upstream, so that
geodesic distance from the centre sensor patch is analytic and damage location
parameterises cleanly as (radius, angle). Verified for grid=32, radius=10:
341 alive cells, max Chebyshev geodesic from centre exactly 10.

Targets are RGB-premultiplied-by-alpha, matching upstream. This matters: the loss
compares the raw state against this target over the whole grid, so a premultiplied
background (RGB = 0 where alpha = 0) is what supervises dead cells toward zero
colour. Do not "fix" this by masking the loss.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import torch
from torch import Tensor

DEFAULT_COLOR: tuple[float, float, float] = (0.20, 0.60, 0.86)


def _coords(grid: int, *, device, dtype) -> tuple[Tensor, Tensor, float]:
    """Index-space coordinate grids and the centre index.

    The centre is ``grid // 2`` in INDEX coordinates, not ``grid / 2`` in
    pixel-corner coordinates, because that is exactly where
    ``NCAOrganism.seed_state`` places its single live cell. A half-cell offset
    between the seed and the target centre would bias every geodesic-distance
    measurement in the propagation probe and every (radius, angle) damage
    parameterisation.
    """
    c = grid // 2
    i = torch.arange(grid, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(i, i, indexing="ij")
    return yy, xx, float(c)


def _antialiased_disk(
    yy: Tensor, xx: Tensor, cy: float, cx: float, radius: float, edge: float
) -> Tensor:
    """Coverage-style alpha for a disk: 1 inside, linear ramp over `edge` at the rim."""
    d = ((yy - cy) ** 2 + (xx - cx) ** 2).sqrt()
    return ((radius + 0.5 - d) / edge).clamp(0.0, 1.0)


def _compose(alpha: Tensor, rgb: Tensor) -> Tensor:
    """-> (1, 4, H, W) with RGB premultiplied by alpha."""
    return torch.cat([rgb * alpha, alpha[None]], dim=0)[None]


def _rgb_field(
    mode: str, yy: Tensor, xx: Tensor, grid: int, color: Sequence[float]
) -> Tensor:
    if mode == "flat":
        return torch.tensor(color, device=yy.device, dtype=yy.dtype).view(3, 1, 1).expand(
            3, grid, grid
        )
    if mode == "gradient":
        # Symmetry-breaking fallback: keeps the disk geometry (so geodesic distance
        # stays analytic) while giving the rule an orientable colour field.
        return torch.stack(
            [
                0.15 + 0.70 * xx / (grid - 1),
                0.15 + 0.70 * yy / (grid - 1),
                torch.full_like(xx, 0.55),
            ]
        )
    raise ValueError(f"unknown rgb_mode {mode!r}; expected 'flat' or 'gradient'")


def disk_target(
    grid: int = 32,
    radius: float = 10.0,
    *,
    edge: float = 1.0,
    color: Sequence[float] = DEFAULT_COLOR,
    rgb_mode: str = "flat",
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """A filled disk centred on the seed cell. -> (1, 4, grid, grid), premultiplied."""
    yy, xx, c = _coords(grid, device=device, dtype=dtype)
    alpha = _antialiased_disk(yy, xx, c, c, radius, edge)
    return _compose(alpha, _rgb_field(rgb_mode, yy, xx, grid, color))


def tadpole_target(
    grid: int = 32,
    radius: float = 8.0,
    *,
    lobe_radius: float = 4.0,
    lobe_offset: tuple[float, float] = (0.0, 9.0),
    edge: float = 1.0,
    color: Sequence[float] = DEFAULT_COLOR,
    rgb_mode: str = "flat",
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Asymmetric fallback: a body disk plus an offset lobe, so the shape has an axis.

    Use if the disk's rotational symmetry produces a rotating or breathing
    attractor. ``lobe_offset`` is (dy, dx) in cells.
    """
    yy, xx, c = _coords(grid, device=device, dtype=dtype)
    dy, dx = lobe_offset
    body = _antialiased_disk(yy, xx, c, c, radius, edge)
    lobe = _antialiased_disk(yy, xx, c + dy, c + dx, lobe_radius, edge)
    alpha = torch.maximum(body, lobe)
    return _compose(alpha, _rgb_field(rgb_mode, yy, xx, grid, color))


def build_target(
    shape: str,
    grid: int,
    *,
    radius: float = 10.0,
    edge: float = 1.0,
    color: Sequence[float] = DEFAULT_COLOR,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Dispatch on a config string. ``disk`` | ``disk_gradient`` | ``tadpole``."""
    common = dict(grid=grid, edge=edge, color=color, device=device, dtype=dtype)
    if shape == "disk":
        return disk_target(radius=radius, rgb_mode="flat", **common)
    if shape == "disk_gradient":
        return disk_target(radius=radius, rgb_mode="gradient", **common)
    if shape == "tadpole":
        return tadpole_target(radius=radius, **common)
    raise ValueError(
        f"unknown target shape {shape!r}; expected 'disk', 'disk_gradient' or 'tadpole'"
    )


def target_fingerprint(target: Tensor) -> str:
    """Short content hash, asserted on checkpoint load so a run cannot silently
    resume against a different body."""
    t = target.detach().to("cpu", torch.float32).contiguous()
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()[:12]
