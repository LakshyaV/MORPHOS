"""Cell state -> displayable RGB frames.

Rendering is where clamping belongs -- never in the loss. The state is unbounded
by design, so display composites the premultiplied RGBA over a light background
the same way the target was built, and clamps only at the end.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from morphos.substrate.nca import ALIVE_CHANNEL

BACKGROUND = 1.0  # white


def to_rgb(state: Tensor, *, background: float = BACKGROUND) -> Tensor:
    """(B,C,H,W) -> (B,3,H,W) float in [0,1], premultiplied-alpha composite."""
    rgb = state[:, :3]
    alpha = state[:, ALIVE_CHANNEL : ALIVE_CHANNEL + 1].clamp(0.0, 1.0)
    return (background * (1.0 - alpha) + rgb).clamp(0.0, 1.0)


def to_frames(state: Tensor, *, background: float = BACKGROUND) -> np.ndarray:
    """(B,C,H,W) -> (B,H,W,3) uint8, ready for the video writer."""
    rgb = to_rgb(state, background=background)
    return (rgb * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()


def alpha_frames(state: Tensor) -> np.ndarray:
    """Alpha channel alone as greyscale (B,H,W,3) uint8 -- shows the body outline."""
    a = state[:, ALIVE_CHANNEL : ALIVE_CHANNEL + 1].clamp(0.0, 1.0)
    g = (a * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    return np.repeat(g, 3, axis=-1)


def tile(frames: np.ndarray, ncol: int | None = None, pad: int = 1) -> np.ndarray:
    """(B,H,W,3) -> a single (H',W',3) grid, for looking at a whole batch at once."""
    b, h, w, c = frames.shape
    ncol = ncol or int(np.ceil(np.sqrt(b)))
    nrow = int(np.ceil(b / ncol))
    out = np.full(
        (nrow * (h + pad) + pad, ncol * (w + pad) + pad, c), 255, dtype=np.uint8
    )
    for i in range(b):
        r, cc = divmod(i, ncol)
        y, x = pad + r * (h + pad), pad + cc * (w + pad)
        out[y : y + h, x : x + w] = frames[i]
    return out


def side_by_side(*frames: np.ndarray, pad: int = 2) -> np.ndarray:
    """Horizontally concatenate equal-height frames with a separator."""
    h = frames[0].shape[0]
    gap = np.full((h, pad, 3), 255, dtype=np.uint8)
    out = []
    for i, f in enumerate(frames):
        if i:
            out.append(gap)
        out.append(f)
    return np.concatenate(out, axis=1)


def ascii_preview(state: Tensor, index: int = 0, threshold: float = 0.1) -> str:
    """Terminal-friendly view of one organism's body. Useful over SSH and in logs."""
    a = state[index, ALIVE_CHANNEL].detach().cpu()
    rows = []
    for row in a:
        rows.append("".join("#" if v > 0.9 else ("+" if v > threshold else ".") for v in row))
    return "\n".join(rows)
