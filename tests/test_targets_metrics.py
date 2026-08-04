"""Tests for target generation and morphology metrics.

The first three tests guard silent failures -- bugs that leave the organism
looking perfect on screen while every downstream number is wrong.
"""

from __future__ import annotations

import math

import torch

from morphos.eval.metrics import (
    alive_count,
    body_mask,
    geodesic_distance,
    iou,
    rmse,
)
from morphos.substrate.nca import NCAOrganism, SENDER_LAYOUT
from morphos.task.targets import (
    build_target,
    disk_target,
    tadpole_target,
    target_fingerprint,
)

GRID = 32
RADIUS = 10.0


def test_disk_target_shape_and_premultiply():
    t = disk_target(GRID, RADIUS)
    assert t.shape == (1, 4, GRID, GRID)
    assert t.dtype == torch.float32

    alpha, rgb = t[0, 3], t[0, :3]
    assert math.isclose(alpha.max().item(), 1.0, abs_tol=1e-6)
    # Premultiplied => colour can never exceed coverage anywhere.
    assert (rgb <= alpha.unsqueeze(0) + 1e-6).all(), "target RGB is not premultiplied by alpha"
    # Corners are far outside a radius-10 disk on a 32 grid.
    assert t[0, :, 0, 0].abs().sum() == 0
    assert t[0, :, -1, -1].abs().sum() == 0

    n = alive_count(t).item()
    assert 336 <= n <= 346, f"expected ~341 alive cells for r=10 on 32x32, got {n}"


def test_target_centre_matches_seed_cell():
    """The disk centre must be the exact cell where seed_state puts its live cell.

    A half-cell offset here biases every geodesic-distance measurement in the
    propagation probe and every (radius, angle) damage parameterisation.
    """
    t = disk_target(GRID, RADIUS)
    org = NCAOrganism(layout=SENDER_LAYOUT, grid=GRID)
    seed = org.seed_state(1)

    seed_yx = (seed[0, 3] > 0).nonzero()
    assert seed_yx.shape[0] == 1, "seed should have exactly one live cell"
    sy, sx = seed_yx[0].tolist()

    assert math.isclose(t[0, 3, sy, sx].item(), 1.0, abs_tol=1e-6)

    # Alpha centroid coincides with the seed cell.
    a = t[0, 3]
    i = torch.arange(GRID, dtype=torch.float32)
    yy, xx = torch.meshgrid(i, i, indexing="ij")
    cy = (a * yy).sum() / a.sum()
    cx = (a * xx).sum() / a.sum()
    assert math.isclose(cy.item(), sy, abs_tol=1e-4), f"centroid y {cy.item()} != seed {sy}"
    assert math.isclose(cx.item(), sx, abs_tol=1e-4), f"centroid x {cx.item()} != seed {sx}"


def test_mask_convention_dilated_vs_raw():
    """Morphology metrics must use raw alpha, never the dilated NCA alive_mask.

    NCAOrganism.alive_mask 3x3-maxpools alpha, marking a cell alive if any
    NEIGHBOUR is alive. Measured on this exact target: IoU(dilated, raw) = 0.7949.
    So using the dilated mask caps IoU at ~0.795 and makes gate G1 (>= 0.90)
    structurally unreachable, while the body looks correct and the loss still falls.
    This test is the entire defence against that.
    """
    t = disk_target(GRID, RADIUS)
    org = NCAOrganism(layout=SENDER_LAYOUT, grid=GRID)

    raw = body_mask(t)
    dilated = org.alive_mask(t) > 0.5

    assert iou(raw, raw).item() == 1.0

    overlap = iou(dilated, raw).item()
    assert 0.78 < overlap < 0.81, f"expected the measured ~0.7949, got {overlap:.4f}"
    assert overlap < 0.90, "dilated mask cannot reach the G1 threshold -- do not use it"


def test_iou_basic_properties():
    a = torch.zeros(2, 1, 8, 8, dtype=torch.bool)
    b = torch.zeros(2, 1, 8, 8, dtype=torch.bool)
    a[:, :, 0:4, 0:4] = True  # 16 cells
    b[:, :, 2:6, 0:4] = True  # 16 cells, 8 shared

    got = iou(a, b)
    assert torch.allclose(got, torch.full((2,), 8 / 24, dtype=torch.float64))

    # Disjoint -> 0; both empty -> 0 (not NaN), which is the meaningful value.
    assert iou(a, ~a).sum().item() == 0.0
    empty = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
    assert iou(empty, empty).item() == 0.0


def test_iou_broadcasts_target_over_batch():
    t = disk_target(GRID, RADIUS)
    batch = t.repeat(5, 1, 1, 1)
    got = iou(body_mask(batch), body_mask(t))
    assert got.shape == (5,)
    assert torch.allclose(got, torch.ones(5, dtype=torch.float64))


def test_rmse_full_grid_equals_sqrt_of_mse():
    """Gate G1 uses full-grid RMSE precisely because it equals sqrt(training loss)."""
    t = disk_target(GRID, RADIUS)
    x = torch.zeros(1, SENDER_LAYOUT.total, GRID, GRID)
    x[:, :4] = t + 0.1

    mse = (x[:, :4] - t).pow(2).mean()
    assert math.isclose(rmse(x, t).item(), mse.sqrt().item(), rel_tol=1e-6)

    # Masked form is a different, harsher quantity -- must not be conflated.
    masked = rmse(x, t, mask=body_mask(t)).item()
    assert masked > 0


def test_geodesic_distance_on_disk_matches_chebyshev():
    t = disk_target(GRID, RADIUS)
    alive = body_mask(t)
    c = GRID // 2
    src = torch.zeros_like(alive)
    src[0, 0, c, c] = True

    d = geodesic_distance(alive, src)
    assert d[0, 0, c, c] == 0.0

    finite = d[torch.isfinite(d)]
    assert finite.max().item() == 10.0, "max Chebyshev geodesic on r=10 disk must be 10"

    # Off-body cells are unreachable.
    assert torch.isinf(d[0, 0, 0, 0])

    # On an intact convex body, geodesic == Chebyshev distance.
    i = torch.arange(GRID)
    yy, xx = torch.meshgrid(i, i, indexing="ij")
    cheb = torch.maximum((yy - c).abs(), (xx - c).abs()).float()
    on = alive[0, 0] & torch.isfinite(d[0, 0])
    assert torch.equal(d[0, 0][on], cheb[on])


def test_geodesic_routes_around_a_wound():
    """The metric must be geodesic, not a plain distance transform.

    Split the body with a wall: cells on the far side must be reached by going
    around, so their geodesic distance strictly exceeds the Chebyshev distance.
    """
    alive = torch.zeros(1, 1, 16, 16, dtype=torch.bool)
    alive[0, 0, 2:14, 2:14] = True
    alive[0, 0, 2:12, 8] = False  # wall from the top, leaving a gap at the bottom

    src = torch.zeros_like(alive)
    src[0, 0, 3, 3] = True
    d = geodesic_distance(alive, src)

    far_y, far_x = 3, 12
    assert alive[0, 0, far_y, far_x]
    cheb = max(abs(far_y - 3), abs(far_x - 3))
    assert torch.isfinite(d[0, 0, far_y, far_x])
    assert d[0, 0, far_y, far_x].item() > cheb, "distance did not route around the wall"


def test_unreachable_region_is_infinite():
    alive = torch.zeros(1, 1, 12, 12, dtype=torch.bool)
    alive[0, 0, 1:4, 1:4] = True
    alive[0, 0, 8:11, 8:11] = True  # disconnected island
    src = torch.zeros_like(alive)
    src[0, 0, 2, 2] = True

    d = geodesic_distance(alive, src)
    assert torch.isfinite(d[0, 0, 3, 3])
    assert torch.isinf(d[0, 0, 9, 9]), "disconnected island must stay unreachable"


def test_tadpole_is_asymmetric():
    t = tadpole_target(GRID)
    a = t[0, 3]
    # Mirroring left-right must change the shape, otherwise it has no axis.
    assert not torch.allclose(a, a.flip(-1)), "tadpole should break left-right symmetry"
    assert alive_count(t).item() > 0


def test_build_target_dispatch_and_fingerprint():
    disk = build_target("disk", GRID, radius=RADIUS)
    grad = build_target("disk_gradient", GRID, radius=RADIUS)
    tad = build_target("tadpole", GRID)

    # Same geometry, different colour field.
    assert torch.equal(body_mask(disk), body_mask(grad))
    assert not torch.allclose(disk[:, :3], grad[:, :3])
    assert not torch.equal(body_mask(disk), body_mask(tad))

    assert target_fingerprint(disk) == target_fingerprint(build_target("disk", GRID, radius=RADIUS))
    assert target_fingerprint(disk) != target_fingerprint(grad)

    try:
        build_target("banana", GRID)
    except ValueError as e:
        assert "banana" in str(e)
    else:
        raise AssertionError("unknown shape must raise")
