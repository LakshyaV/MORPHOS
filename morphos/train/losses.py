"""Losses and gradient normalisation.

The morphology loss is a plain elementwise MSE against the premultiplied RGBA
target over the WHOLE grid. Do not alive-mask it: the premultiplied background
(RGB = 0 where alpha = 0) is precisely what supervises dead cells toward zero
colour. Masking removes that signal and the organism smears colour into the void.

Do not clamp the state before the loss either -- upstream does not, and clamping
hides exactly the runaway values the loss should be punishing. Clamp only for
display.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def rgba_mse(state: Tensor, target: Tensor) -> Tensor:
    """Scalar training loss: mean over B x 4 x H x W.

    Because this is a per-element mean, ``sqrt(loss)`` is exactly the full-grid
    RMSE that gate G1 checks -- so G1 is predictable from the loss curve alone.
    """
    return (state[:, :4] - target).pow(2).mean()


def per_sample_rgba_mse(state: Tensor, target: Tensor) -> Tensor:
    """(B,) per-sample loss, used to rank pool entries for reseeding/damage."""
    return (state[:, :4] - target).pow(2).mean(dim=(1, 2, 3))


@torch.no_grad()
def normalize_grads_(module: nn.Module, eps: float = 1e-8) -> None:
    """Per-parameter-tensor L2 gradient normalisation, in place.

    Upstream reports this was required to fix sudden loss spikes late in training.
    Each parameter tensor is normalised by its OWN norm, not a global norm -- that
    distinction is the whole point.

    Iterating ``parameters()`` automatically excludes the fixed Sobel perception
    kernel, which is a registered buffer and never receives a gradient. Do not add
    gradient clipping on top: per-tensor normalisation already fixes the scale, and
    stacking the two makes them fight.

    A dead organism yields an exactly-zero gradient; ``0 / (0 + eps) = 0``, so this
    is NaN-safe. It is also unrecoverable, which is what the DeathGuard is for.
    """
    for p in module.parameters():
        if p.grad is not None:
            p.grad.div_(p.grad.norm() + eps)
