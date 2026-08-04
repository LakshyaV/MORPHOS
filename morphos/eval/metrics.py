"""Morphology metrics.

CRITICAL CONVENTION: morphology metrics use the RAW alpha threshold
(``body_mask``), never ``NCAOrganism.alive_mask``.

``alive_mask`` 3x3-maxpools alpha, so it marks a cell alive when any *neighbour*
is alive. That dilation is correct for the NCA update rule (it is what lets the
organism grow outward) and wrong for measuring shape. Measured on the canonical
grid=32 / radius=10 disk: ``IoU(alive_mask(target), body_mask(target)) = 0.7949``.
Using the dilated mask for IoU therefore caps the achievable value at ~0.795 and
makes gate G1 (>= 0.90) **structurally unreachable** -- while the organism looks
perfect on screen and the training loss keeps falling. ``test_mask_convention``
exists to prevent exactly this.

Reductions go through float64 on CPU: MPS has no float64, and these numbers are
compared against pre-registered thresholds.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from morphos.substrate.nca import ALIVE_CHANNEL, ALIVE_THRESHOLD


def body_mask(x: Tensor, threshold: float = ALIVE_THRESHOLD) -> Tensor:
    """(B,C,H,W) -> (B,1,H,W) bool. THE morphology mask: raw alpha, no dilation."""
    return x[:, ALIVE_CHANNEL : ALIVE_CHANNEL + 1] > threshold


def alive_count(x: Tensor, threshold: float = ALIVE_THRESHOLD) -> Tensor:
    """(B,C,H,W) -> (B,) long. Number of cells in the body."""
    return body_mask(x, threshold).flatten(1).sum(dim=1)


def _as_bool(m: Tensor) -> Tensor:
    return m if m.dtype == torch.bool else m > 0.5


def iou(a: Tensor, b: Tensor) -> Tensor:
    """Intersection-over-union of two masks. -> (B,) float64 on CPU.

    ``b`` broadcasts over the batch, so a (1,1,H,W) target compares against a
    (B,1,H,W) batch of bodies. An empty union scores 0 rather than NaN, which is
    the meaningful value for a dead organism.
    """
    a, b = _as_bool(a), _as_bool(b)
    inter = (a & b).flatten(1).sum(dim=1).cpu().double()
    union = (a | b).flatten(1).sum(dim=1).cpu().double()
    return torch.where(union > 0, inter / union.clamp(min=1), torch.zeros_like(union))


def union_mask(a: Tensor, b: Tensor) -> Tensor:
    return _as_bool(a) | _as_bool(b)


def rmse(x: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    """Root-mean-square error on the RGBA channels. -> (B,) float64 on CPU.

    ``mask=None`` gives the full-grid per-element RMSE, which equals
    ``sqrt(training loss)`` and is what gate G1 uses -- so G1 is predictable from
    the loss curve alone. Pass a mask for the union-masked form used by the
    recovery metric in docs/PROTOCOL.md, where empty background is definitionally
    irrelevant.
    """
    err = (x[:, :4] - target).cpu().double().pow(2)
    if mask is None:
        return err.mean(dim=(1, 2, 3)).sqrt()
    m = _as_bool(mask).to("cpu")
    if m.shape[0] == 1 and err.shape[0] > 1:
        m = m.expand(err.shape[0], -1, -1, -1)
    counts = m.flatten(1).sum(dim=1).double().clamp(min=1) * 4
    return (err * m).sum(dim=(1, 2, 3)).div(counts).sqrt()


def geodesic_distance(
    alive: Tensor, source: Tensor, *, max_iter: int | None = None
) -> Tensor:
    """8-connected (Chebyshev) BFS distance from `source`, restricted to `alive`.

    -> (B,1,H,W) float; cells unreachable within the body are ``+inf``.

    Chebyshev is the right metric here because it is exactly the NCA light cone:
    a 3x3 neighbourhood advances information one cell in any of 8 directions per
    step. On an intact disk this equals the Euclidean distance; on a damaged body
    it correctly routes around the wound, which is the whole point.
    """
    alive_b = _as_bool(alive).float()
    reach = (_as_bool(source).float() * alive_b)
    dist = torch.where(
        reach > 0, torch.zeros_like(reach), torch.full_like(reach, float("inf"))
    )
    limit = max_iter if max_iter is not None else alive.shape[-1] * alive.shape[-2]

    for t in range(1, limit + 1):
        grown = (F.max_pool2d(reach, kernel_size=3, stride=1, padding=1) > 0).float() * alive_b
        newly = (grown > 0) & torch.isinf(dist)
        if not newly.any():
            break
        dist = torch.where(newly, torch.full_like(dist, float(t)), dist)
        reach = grown
    return dist
