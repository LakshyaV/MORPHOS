"""Frames -> mp4, piped straight to ffmpeg.

No intermediate PNGs on disk and no imagemagick dependency.

`flags=neighbor` on the upscale is not cosmetic: the default bicubic filter turns
a 32x32 automaton into mush, and you end up misreading your own results. Cells
must stay crisp squares.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

FFMPEG = shutil.which("ffmpeg") or "/usr/local/bin/ffmpeg"


def ffmpeg_available() -> bool:
    return Path(FFMPEG).exists() or shutil.which("ffmpeg") is not None


def write_mp4(
    frames: np.ndarray,
    path: str | Path,
    *,
    fps: int = 12,
    scale: int = 8,
    crf: int = 18,
) -> Path:
    """frames: (T,H,W,3) uint8 -> an mp4 at `path`."""
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected (T,H,W,3) uint8, got {frames.shape}")
    if frames.dtype != np.uint8:
        raise TypeError(f"expected uint8 frames, got {frames.dtype}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _, h, w, _ = frames.shape

    # H.264 with yuv420p requires even dimensions.
    out_w, out_h = w * scale, h * scale
    out_w += out_w % 2
    out_h += out_h % 2

    cmd = [
        FFMPEG, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-vf", f"scale={out_w}:{out_h}:flags=neighbor",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf),
        str(path),
    ]
    proc = subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(),
                          capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode()[:500]}")
    return path


def probe_mp4(path: str | Path) -> dict:
    """Read back dimensions and frame count, so tests can verify what was written."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return {}
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,nb_read_packets",
         "-count_packets", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    if out.returncode != 0 or not out.stdout.strip():
        return {}
    parts = out.stdout.strip().split(",")
    return {"width": int(parts[0]), "height": int(parts[1]), "frames": int(parts[2])}
