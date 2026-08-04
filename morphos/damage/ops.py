"""Damage operators.

Two deliberately separate entry points, which must not be merged:

- `sample_training_masks` is upstream's radius-uniform circle **augmentation**,
  used to teach regeneration. Its parameters live in normalised [-1,1] coords
  exactly as upstream, so the relative severity transfers across grid sizes.
- `kill_fraction` is our **evaluation** severity control, parameterised by the
  fraction of currently-alive cells destroyed and solved per episode by
  bisection. Defining severity by radius instead would confound it with body size
  and regeneration state -- a damaged, half-regrown organism has fewer cells, so
  a fixed radius is a different severity at every timepoint.

Damage is applied strictly BEFORE a rollout in the morphology phase, never
mid-rollout, matching upstream.
"""

from __future__ import annotations

import torch
from torch import Tensor

from morphos.eval.metrics import body_mask


def _cell_coords(grid: int, *, device, dtype=torch.float32) -> tuple[Tensor, Tensor]:
    i = torch.arange(grid, device=device, dtype=dtype)
    return torch.meshgrid(i, i, indexing="ij")


def disk_keep_mask(
    grid: int, centres: Tensor, radii: Tensor, *, device: torch.device | str = "cpu"
) -> Tensor:
    """-> (B,1,H,W) float. 1 = KEEP, 0 = destroyed.

    `centres` is (B,2) in CELL coordinates (row, col); `radii` is (B,) in cells.
    """
    yy, xx = _cell_coords(grid, device=device)
    cy = centres[:, 0].view(-1, 1, 1)
    cx = centres[:, 1].view(-1, 1, 1)
    d2 = (yy[None] - cy) ** 2 + (xx[None] - cx) ** 2
    inside = d2 < radii.view(-1, 1, 1) ** 2
    return (~inside).to(torch.float32).unsqueeze(1)


def sample_training_masks(
    batch: int,
    grid: int,
    *,
    generator: torch.Generator,
    device: torch.device | str = "cpu",
    centre_range: float = 0.5,
    radius_range: tuple[float, float] = (0.1, 0.4),
) -> Tensor:
    """Upstream's augmentation, verbatim, in normalised [-1,1] coordinates.

    centre ~ U[-0.5, 0.5]^2 and radius ~ U[0.1, 0.4] of half the grid. On a 32
    grid that is 1.6-6.4 cells against a body radius of 10, i.e. relative severity
    0.16-0.64 -- matching upstream's 0.18-0.72 on its own geometry.
    """
    lo, hi = radius_range
    half = grid / 2.0
    u = torch.rand(batch, 3, generator=generator, device=device)
    centres_norm = (u[:, :2] - 0.5) * 2.0 * centre_range  # in [-centre_range, centre_range]
    centres = centres_norm * half + half
    radii = (lo + (hi - lo) * u[:, 2]) * half
    return disk_keep_mask(grid, centres, radii, device=device)


def apply_damage(x: Tensor, keep: Tensor) -> Tensor:
    """Zero every channel of destroyed cells. Out of place -- never mutates `x`."""
    return x * keep


def sample_on_body(
    alive: Tensor, *, generator: torch.Generator
) -> Tensor:
    """Draw one damage centre per batch element, uniformly from that body's cells.

    -> (B,2) float cell coordinates (row, col). Falls back to the grid centre for
    an empty body so a dead organism cannot crash the evaluation.
    """
    b, _, h, w = alive.shape
    flat = alive.view(b, -1).to(torch.float32)
    empty = flat.sum(dim=1) == 0
    weights = torch.where(empty.view(-1, 1), torch.ones_like(flat), flat)
    idx = torch.multinomial(weights, 1, generator=generator).squeeze(1)
    centres = torch.stack([(idx // w).to(torch.float32), (idx % w).to(torch.float32)], dim=1)
    fallback = torch.tensor([h // 2, w // 2], device=alive.device, dtype=torch.float32)
    return torch.where(empty.view(-1, 1), fallback.expand_as(centres), centres)


def _killed_fraction(alive: Tensor, keep: Tensor) -> Tensor:
    """Fraction of alive cells that `keep` destroys. -> (B,)"""
    a = alive.to(torch.float32)
    total = a.flatten(1).sum(dim=1).clamp(min=1)
    killed = (a * (1.0 - keep)).flatten(1).sum(dim=1)
    return killed / total


def solve_radius_for_fraction(
    alive: Tensor,
    centres: Tensor,
    target_f: float,
    *,
    tol: float = 0.02,
    max_iter: int = 24,
) -> tuple[Tensor, Tensor]:
    """Vectorised bisection for the radius hitting `target_f` of alive cells.

    f(rho) is a step function of rho, so bisection cannot hit the target exactly.
    After the search we take whichever bracket end is closer and RETURN the
    achieved fraction, so the caller asserts the tolerance rather than the solver
    silently lying about it.
    """
    grid = alive.shape[-1]
    device = alive.device
    b = alive.shape[0]

    lo = torch.zeros(b, device=device)
    hi = torch.full((b,), float(grid) * 1.5, device=device)

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f = _killed_fraction(alive, disk_keep_mask(grid, centres, mid, device=device))
        too_small = f < target_f
        lo = torch.where(too_small, mid, lo)
        hi = torch.where(too_small, hi, mid)

    f_lo = _killed_fraction(alive, disk_keep_mask(grid, centres, lo, device=device))
    f_hi = _killed_fraction(alive, disk_keep_mask(grid, centres, hi, device=device))
    pick_hi = (f_hi - target_f).abs() < (f_lo - target_f).abs()
    radii = torch.where(pick_hi, hi, lo)
    achieved = torch.where(pick_hi, f_hi, f_lo)
    return radii, achieved


def kill_fraction(
    x: Tensor,
    target_f: float,
    *,
    generator: torch.Generator,
    threshold: float = 0.1,
    tol: float = 0.02,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Destroy a disk containing `target_f` of the currently-alive cells.

    Returns the damaged state and diagnostics, including whether the disk covered
    the grid centre -- free to compute, and it pre-figures the sensor-targeted
    damage arm, where destroying the input pathway is a different experiment from
    destroying computation.
    """
    alive = body_mask(x, threshold)
    grid = x.shape[-1]
    centres = sample_on_body(alive, generator=generator)
    radii, achieved = solve_radius_for_fraction(alive, centres, target_f, tol=tol)
    keep = disk_keep_mask(grid, centres, radii, device=x.device)

    c = grid // 2
    covers_centre = keep[:, 0, c, c] < 0.5

    return apply_damage(x, keep), {
        "radius": radii,
        "achieved_f": achieved,
        "centre": centres,
        "covers_centre": covers_centre,
        "keep": keep,
    }
