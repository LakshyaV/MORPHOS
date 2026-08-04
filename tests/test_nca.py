"""Unit tests for the NCA substrate.

These target the failure modes that are silent rather than loud: a transposed
Sobel filter or a mis-grouped perception kernel still trains, just worse, and
would not surface for a week.
"""

from __future__ import annotations

import torch

from morphos.substrate.nca import (
    ChannelLayout,
    NCAOrganism,
    build_perception_kernel,
)


def make_org(**kw) -> NCAOrganism:
    kw.setdefault("layout", ChannelLayout(n_sensor=1, n_vote=2, n_hidden=1))
    kw.setdefault("hidden", 16)
    kw.setdefault("grid", 8)
    return NCAOrganism(**kw)


def wake_up(org: NCAOrganism, seed: int = 0, std: float = 0.5) -> NCAOrganism:
    """Randomise the zero-initialised output layer so the rule actually does something.

    Uses an explicit generator: seeding via the global RNG makes these tests
    flaky, because a bad draw can kill the organism outright.
    """
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        org.fc2.weight.normal_(mean=0.0, std=std, generator=g)
    return org


def test_zero_init_is_identity():
    """fc2 zero-init => dx == 0 => the first step is a no-op on every alive cell.

    Catches init bugs, residual-connection sign errors, and a stray output bias.
    """
    org = make_org()
    g = torch.Generator().manual_seed(0)
    x = torch.rand(2, org.channels, 8, 8, generator=g)
    x[:, 3] = 1.0  # force every cell alive
    mask = org.sample_fire_mask(2, torch.Generator().manual_seed(1), x.device)

    y = org.step(x.clone(), mask)
    assert torch.equal(x, y), (x - y).abs().max().item()


def test_perception_kernel_layout():
    """groups=C assigns 3 consecutive out-channels per in-channel: [id, sx, sy]."""
    k = build_perception_kernel(4)
    assert k.shape == (12, 1, 3, 3)
    for c in range(4):
        assert k[3 * c + 0, 0, 1, 1] == 1.0, "slot 0 of each triple must be identity"
        assert torch.equal(k[3 * c + 1], k[1]), "slot 1 must be sobel_x for every channel"
        assert torch.equal(k[3 * c + 2], k[2]), "slot 2 must be sobel_y for every channel"


def test_sobel_on_ramp():
    """A ramp increasing along x: sobel_x is constant nonzero, sobel_y is exactly 0.

    This is the test that catches the x/y transpose and the groups= interleaving,
    both of which are otherwise invisible.
    """
    org = make_org()
    C = org.channels
    # Channel c is a ramp along the width axis, scaled by (c+1).
    ramp = torch.arange(8, dtype=torch.float32).view(1, 1, 1, 8).expand(1, C, 8, 8).clone()
    ramp = ramp * torch.arange(1, C + 1, dtype=torch.float32).view(1, C, 1, 1)

    p = org.perceive(ramp)[0, :, 1:-1, 1:-1]  # drop the zero-padded border

    for c in range(C):
        scale = c + 1
        identity = p[3 * c + 0]
        sobel_x = p[3 * c + 1]
        sobel_y = p[3 * c + 2]

        expected_id = ramp[0, c, 1:-1, 1:-1]
        assert torch.allclose(identity, expected_id), f"identity wrong on channel {c}"
        assert torch.allclose(sobel_x, torch.full_like(sobel_x, float(scale)), atol=1e-6), (
            f"sobel_x should be the constant gradient {scale} on channel {c}"
        )
        assert torch.allclose(sobel_y, torch.zeros_like(sobel_y), atol=1e-6), (
            f"sobel_y must vanish on an x-ramp (channel {c}) -- filters are transposed"
        )


def test_growth_respects_light_cone():
    """Locality bound: after k steps the support fits in a (2k+1)^2 square.

    A violation means padding or the alive-mask maxpool is wrong. Information
    travelling faster than 1 cell/step would also invalidate the propagation
    argument the whole project rests on.
    """
    org = wake_up(make_org(grid=16), seed=0, std=0.5)
    x = org.seed_state(1, device="cpu")
    g = torch.Generator().manual_seed(0)

    prev = 1
    for k in range(1, 8):
        x = org.rollout(x, 1, generator=g)
        support = (x.abs().sum(dim=1) > 0).sum().item()
        assert support <= (2 * k + 1) ** 2, f"step {k}: {support} cells exceeds light cone"
        assert support >= 1, "organism died out entirely"
        prev = support
    assert prev > 1, "no growth at all -- perception kernel grouping is likely wrong"


def test_dead_cells_stay_dead():
    """A cell with no alive neighbour must remain exactly zero."""
    org = wake_up(make_org(grid=16), seed=0, std=0.5)
    x = org.seed_state(1, device="cpu")
    g = torch.Generator().manual_seed(0)
    x = org.rollout(x, 3, generator=g)

    # Corners are >3 steps away from the centre seed, so they cannot be reached.
    assert x[0, :, 0, 0].abs().sum() == 0
    assert x[0, :, -1, -1].abs().sum() == 0


def test_fire_rate_statistics_and_reproducibility():
    org = make_org(fire_rate=0.5)
    m1 = org.sample_fire_mask(64, torch.Generator().manual_seed(7), torch.device("cpu"))
    m2 = org.sample_fire_mask(64, torch.Generator().manual_seed(7), torch.device("cpu"))
    assert torch.equal(m1, m2), "same seed must give the same fire mask"
    assert 0.47 < m1.mean().item() < 0.53, m1.mean().item()


def test_rollout_is_deterministic_given_seed():
    org = wake_up(make_org(grid=16), seed=0, std=0.3)
    x0 = org.seed_state(2, device="cpu")

    a = org.rollout(x0.clone(), 12, generator=torch.Generator().manual_seed(3))
    b = org.rollout(x0.clone(), 12, generator=torch.Generator().manual_seed(3))
    assert torch.equal(a, b)


def test_checkpointed_rollout_matches_plain_rollout():
    """Gradient checkpointing must be numerically transparent.

    Fire masks are pre-generated precisely so recomputation during the backward
    pass reuses the same masks instead of advancing the generator.
    """
    org = wake_up(make_org(grid=16), seed=0, std=0.3)
    x0 = org.seed_state(2, device="cpu")

    plain = org.rollout(x0.clone(), 16, generator=torch.Generator().manual_seed(5))
    ckpt = org.rollout(
        x0.clone(), 16, generator=torch.Generator().manual_seed(5), checkpoint_every=4
    )
    assert torch.allclose(plain, ckpt, atol=1e-6), (plain - ckpt).abs().max().item()


def test_gradients_flow_through_checkpointed_rollout():
    org = wake_up(make_org(grid=16), seed=0, std=0.3)
    x0 = org.seed_state(2, device="cpu")

    out = org.rollout(x0, 8, generator=torch.Generator().manual_seed(1), checkpoint_every=4)
    out.square().mean().backward()

    assert org.fc1.weight.grad is not None
    assert org.fc1.weight.grad.abs().sum() > 0, "no gradient reached the update rule"
    assert torch.isfinite(org.fc1.weight.grad).all()


def test_alive_mask_semantics():
    org = make_org(grid=8)
    x = torch.zeros(1, org.channels, 8, 8)
    x[0, 3, 4, 4] = 1.0  # one cell with alpha above threshold

    mask = org.alive_mask(x)
    assert mask.sum().item() == 9, "a live cell should mark its 3x3 neighbourhood"
    assert mask[0, 0, 4, 4] == 1.0
    assert mask[0, 0, 0, 0] == 0.0
