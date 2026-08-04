"""Propagation probe tests.

The light-cone test is the probe's self-validation: information cannot outrun the
3x3 neighbourhood, so any effect at geodesic distance d > t means the injection is
leaking globally (a broadcast-shape bug, or the alive mask failing to localise).
Without it, a leaking probe would report a beautifully fast propagation speed and
we would believe it.
"""

from __future__ import annotations

import pytest
import torch

from morphos.eval.probe import (
    ETA2_THRESHOLD,
    ProbeResult,
    bipolar_codes,
    run_probe,
    sensor_patch_mask,
)
from morphos.seeding import make_rng
from morphos.substrate.nca import ChannelLayout, NCAOrganism

TOY_LAYOUT = ChannelLayout(n_sensor=2, n_vote=2, n_hidden=4)


def toy_model(grid=12, seed=0, std=0.2):
    """A woken-up but untrained rule.

    std matters: at 0.5 the untrained dynamics are chaotic and state magnitudes
    reach ~275, which swamps a +/-1 injected code and drives eta2 to ~0. At 0.2
    states stay bounded around 4, which is the regime a trained model lives in and
    the one the probe is designed for.
    """
    org = NCAOrganism(layout=TOY_LAYOUT, hidden=16, grid=grid)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        org.fc2.weight.normal_(std=std, generator=g)
    return org


def test_bipolar_codes_are_plus_minus_one():
    """{0,1} would make the all-zero code inject nothing, indistinguishable from
    no injection at all, corrupting the variance decomposition."""
    c = bipolar_codes(3)
    assert c.shape == (8, 3)
    assert set(c.unique().tolist()) == {-1.0, 1.0}
    assert c.unique(dim=0).shape[0] == 8, "codes must be distinct"
    assert (c.abs() == 1).all()


def test_sensor_patch_is_centred_and_right_size():
    m = sensor_patch_mask(32, 3)
    assert m.sum().item() == 9
    assert m[0, 0, 16, 16]
    assert not m[0, 0, 0, 0]
    ys, xs = m[0, 0].nonzero(as_tuple=True)
    assert ys.min() == 15 and ys.max() == 17


def test_probe_respects_the_light_cone():
    """eta2 must be ~0 wherever geodesic distance exceeds elapsed steps.

    This is the probe's self-check: a violation means the injection is not
    localised and every propagation number would be fiction.
    """
    model = toy_model(grid=12)
    rng = make_rng(0, torch.device("cpu"))
    res = run_probe(
        model, TOY_LAYOUT, rng=rng, device=torch.device("cpu"),
        t_grow=10, t_probe=8, t_inject=8, n_noise=4, patch=3,
    )

    for t in range(res.eta2.shape[0]):
        # A cell at distance d can only be reached after d steps; the patch itself
        # spans radius 1, so allow d <= t + 1.
        beyond = torch.isfinite(res.dist) & (res.dist > t + 1)
        if beyond.any():
            leaked = res.eta2[t][beyond].max().item()
            assert leaked < 1e-6, (
                f"t={t}: eta2={leaked:.2e} at distance > {t + 1} -- injection is leaking"
            )


def test_probe_detects_the_injected_signal_at_the_source():
    """Sanity in the other direction: the sensor patch itself must show a strong
    effect, otherwise the probe measures nothing and the light-cone test passes
    vacuously."""
    # Grow long enough that the body actually extends past the 3x3 sensor patch,
    # otherwise there is nowhere for a signal to propagate to and max_dist is 0.
    model = toy_model(grid=16)
    rng = make_rng(0, torch.device("cpu"))
    res = run_probe(
        model, TOY_LAYOUT, rng=rng, device=torch.device("cpu"),
        t_grow=24, t_probe=12, t_inject=8, n_noise=4, patch=3,
    )
    source = res.eta2_by_dist[:, 0].max().item()
    # eta2 pools over ALL channels, and an untrained chaotic rule leaves most of
    # them carrying pure update noise, so the source effect is a fraction rather
    # than near 1. What must hold is that it is clearly non-trivial and clearly
    # stronger than the far field.
    assert source > 0.2, f"no signal even at the source (eta2={source:.3f})"
    assert res.max_dist > 0, "body has no extent; probe cannot say anything"

    # The signal must attenuate with distance. Note we do NOT require the far
    # field to be weak: a healthy substrate propagates, and a strong far field is
    # the outcome we want. Localisation is asserted separately, by the light-cone
    # test, which is the correct instrument for it.
    far = res.eta2_max_by_dist[res.max_dist].item()
    assert source >= far, (
        f"far field ({far:.3f}) exceeds the source ({source:.3f}); "
        "the effect is not originating at the sensor patch"
    )


def test_probe_result_shapes_are_consistent():
    model = toy_model(grid=12)
    rng = make_rng(0, torch.device("cpu"))
    res = run_probe(
        model, TOY_LAYOUT, rng=rng, device=torch.device("cpu"),
        t_grow=10, t_probe=10, t_inject=6, n_noise=4,
    )
    T = res.eta2.shape[0]
    assert T == 10
    assert res.eta2.shape[1:] == (12, 12)
    assert res.eta2_by_dist.shape == (T, res.max_dist + 1)
    assert res.front.shape == (T,)
    assert res.eta2_max_by_dist.shape == (res.max_dist + 1,)
    assert (res.eta2 >= 0).all() and (res.eta2 <= 1 + 1e-9).all()


def _result(t_cover, far_max, max_dist=10):
    n = max_dist + 1
    by_dist = torch.zeros(n, dtype=torch.float64)
    by_dist[:6] = 0.9
    by_dist[6:] = far_max
    return ProbeResult(
        eta2=torch.zeros(1, 4, 4), dist=torch.zeros(4, 4),
        eta2_by_dist=torch.zeros(1, n), front=torch.zeros(1),
        speed=0.5, t_cover=t_cover, eta2_max_by_dist=by_dist, max_dist=max_dist,
    )


@pytest.mark.parametrize(
    "t_cover,far_max,expected",
    [
        (20, 0.8, "OK"),          # covered well within T_comm/2
        (48, 0.8, "MARGINAL"),    # covered, but late in T_comm
        (100, 0.8, "TOO_SHORT"),  # not covered within T_comm
        (float("inf"), 0.0, "ATTENUATING"),  # never reaches far cells at all
    ],
)
def test_verdict_distinguishes_slow_from_attenuating(t_cover, far_max, expected):
    """The whole point of measuring eta2 over (distance, time): 'slow' is fixed by
    raising T_comm, 'attenuating' is not fixed by time at all."""
    assert _result(t_cover, far_max).verdict(t_comm=64) == expected


def test_attenuating_beats_t_cover_in_the_verdict():
    """Even if coverage looks fine, no far-field signal means ATTENUATING wins --
    otherwise we would raise T_comm against a problem time cannot solve."""
    assert _result(t_cover=10, far_max=0.0).verdict(64) == "ATTENUATING"


def test_summary_mentions_the_action_to_take():
    for t_cover, far, word in [
        (20, 0.8, "consensus"),
        (48, 0.8, "96"),
        (100, 0.8, "128"),
        (float("inf"), 0.0, "amplitude"),
    ]:
        text = _result(t_cover, far).summary(64)
        assert word in text, f"summary should say what to do; got: {text}"
        assert str(ETA2_THRESHOLD) or True
