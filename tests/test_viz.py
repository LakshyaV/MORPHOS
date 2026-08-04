"""Visualisation tests.

Visualisation is built before the sample pool on purpose: persistence failures
(drift, explosion, breathing, a rotating attractor) are instantly obvious in an
mp4 and only slowly obvious in a loss curve.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from morphos.substrate.nca import ALIVE_CHANNEL, SENDER_LAYOUT, NCAOrganism
from morphos.task.targets import disk_target
from morphos.viz.frames import (
    alpha_frames,
    ascii_preview,
    side_by_side,
    tile,
    to_frames,
    to_rgb,
)
from morphos.viz.video import ffmpeg_available, probe_mp4, write_mp4


def test_to_rgb_composites_over_background():
    state = torch.zeros(1, SENDER_LAYOUT.total, 8, 8)
    rgb = to_rgb(state)
    # Fully transparent everywhere -> pure background.
    assert torch.allclose(rgb, torch.ones_like(rgb))

    # Opaque red cell (premultiplied) shows through as red.
    state[0, 0, 4, 4] = 1.0
    state[0, ALIVE_CHANNEL, 4, 4] = 1.0
    rgb = to_rgb(state)
    assert rgb[0, 0, 4, 4] == pytest.approx(1.0)
    assert rgb[0, 1, 4, 4] == pytest.approx(0.0)
    assert rgb[0, 2, 4, 4] == pytest.approx(0.0)


def test_to_rgb_clamps_runaway_state():
    """The state is unbounded by design; clamping belongs in display, not the loss."""
    state = torch.zeros(1, SENDER_LAYOUT.total, 8, 8)
    state[0, :4] = 50.0
    state[0, ALIVE_CHANNEL] = -20.0
    rgb = to_rgb(state)
    assert torch.isfinite(rgb).all()
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0


def test_to_frames_shape_and_dtype():
    t = disk_target(16, 5.0)
    state = torch.zeros(3, SENDER_LAYOUT.total, 16, 16)
    state[:, :4] = t
    f = to_frames(state)
    assert f.shape == (3, 16, 16, 3)
    assert f.dtype == np.uint8


def test_rendered_target_is_visible_against_background():
    """A rendered disk must actually differ from the empty background, otherwise
    every video in the project is a blank rectangle."""
    t = disk_target(32, 10.0)
    state = torch.zeros(1, SENDER_LAYOUT.total, 32, 32)
    state[:, :4] = t
    f = to_frames(state)[0]

    centre, corner = f[16, 16], f[0, 0]
    assert not np.array_equal(centre, corner)
    assert np.array_equal(corner, np.array([255, 255, 255], dtype=np.uint8))
    # A meaningful fraction of the frame is body, not background.
    non_bg = (f != 255).any(axis=-1).mean()
    assert 0.2 < non_bg < 0.6, f"body covers {non_bg:.2%} of the frame"


def test_alpha_frames_are_greyscale():
    t = disk_target(16, 5.0)
    state = torch.zeros(1, SENDER_LAYOUT.total, 16, 16)
    state[:, :4] = t
    g = alpha_frames(state)[0]
    assert g.shape == (16, 16, 3)
    assert np.array_equal(g[..., 0], g[..., 1]) and np.array_equal(g[..., 1], g[..., 2])
    assert g[8, 8, 0] == 255  # centre of the disk is opaque


def test_tile_and_side_by_side_shapes():
    f = np.zeros((4, 8, 8, 3), dtype=np.uint8)
    grid = tile(f, ncol=2, pad=1)
    assert grid.shape == (2 * 9 + 1, 2 * 9 + 1, 3)

    a = np.zeros((8, 8, 3), dtype=np.uint8)
    joined = side_by_side(a, a, pad=2)
    assert joined.shape == (8, 18, 3)


def test_ascii_preview_renders_a_disk():
    t = disk_target(16, 5.0)
    state = torch.zeros(1, SENDER_LAYOUT.total, 16, 16)
    state[:, :4] = t
    art = ascii_preview(state)
    lines = art.splitlines()
    assert len(lines) == 16 and len(lines[0]) == 16
    assert lines[8][8] == "#"  # centre filled
    assert lines[0][0] == "."  # corner empty


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not installed")
def test_write_mp4_produces_a_readable_file(tmp_path):
    org = NCAOrganism(layout=SENDER_LAYOUT, hidden=16, grid=16)
    torch.nn.init.normal_(org.fc2.weight, std=0.3)
    x = org.seed_state(1)
    g = torch.Generator().manual_seed(0)

    frames = []
    for _ in range(10):
        frames.append(to_frames(x)[0])
        x = org.rollout(x, 1, generator=g)

    out = write_mp4(np.stack(frames), tmp_path / "t.mp4", fps=6, scale=4)
    assert out.exists() and out.stat().st_size > 0

    info = probe_mp4(out)
    if info:  # ffprobe present
        assert info["width"] == 64 and info["height"] == 64
        assert info["width"] % 2 == 0 and info["height"] % 2 == 0
        assert info["frames"] == 10


def test_write_mp4_rejects_bad_input(tmp_path):
    with pytest.raises(ValueError):
        write_mp4(np.zeros((4, 8, 8), dtype=np.uint8), tmp_path / "a.mp4")
    with pytest.raises(TypeError):
        write_mp4(np.zeros((4, 8, 8, 3), dtype=np.float32), tmp_path / "b.mp4")
