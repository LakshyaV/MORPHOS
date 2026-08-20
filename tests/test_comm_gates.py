"""Gate G3/G4 estimators, hand-verified on synthetic episode banks.

No NCA rollouts here: the point is that the metric arithmetic is right, checked
against values computable by hand. A perfect identity protocol must pass both
gates; a collapsed sender must fail G3; a receiver that ignores the symbol must
fail G4 even at high MI on the sender side.
"""

from __future__ import annotations

import math

import pytest
import torch

from morphos.eval.comm_gates import (
    EpisodeBank,
    debiased_mi,
    gate_g3,
    gate_g4,
    mi_bits,
    wilson_lower_bound,
)

N = V = 8
E = 512


def test_wilson_lower_bound_hand_value():
    # p=0.9, n=100, z=1.96: centre 0.9192, margin 0.0619, denom 1.0384 -> 0.8256
    assert wilson_lower_bound(90, 100) == pytest.approx(0.8256, abs=1e-3)
    assert wilson_lower_bound(0, 0) == 0.0
    # More evidence at the same rate must tighten the bound.
    assert wilson_lower_bound(900, 1000) > wilson_lower_bound(90, 100)


def test_mi_of_identity_map_is_log2_n():
    r = torch.arange(E) % N  # balanced
    assert mi_bits(r, r, N, N) == pytest.approx(3.0, abs=1e-9)


def test_debiased_mi_of_independent_variables_is_near_zero():
    g = torch.Generator().manual_seed(0)
    a = torch.randint(V, (E,), generator=g)
    b = torch.randint(N, (E,), generator=g)
    out = debiased_mi(a, b, V, N, permutations=200, seed=0)
    # Raw plug-in MI is biased up by ~0.069 bits at this size (PROTOCOL 2.4);
    # debiasing must remove it.
    assert out["mi_raw"] > 0.02
    assert abs(out["mi_debiased"]) < 0.03
    assert out["p_value"] > 0.05


def _bank(symbol_idx: torch.Tensor, probs_by_v: torch.Tensor) -> EpisodeBank:
    return EpisodeBank(
        referents=torch.arange(E) % N,
        symbol_idx=symbol_idx,
        symbol_idx_noisy=symbol_idx,
        probs_by_v=probs_by_v,
        n_referents=N,
        vocab=V,
    )


def _faithful_receiver() -> torch.Tensor:
    """probs_by_v where the receiver answers exactly the symbol it heard, with
    smoothed confidence 0.79/0.03, and is uniform under silence."""
    p = torch.full((V + 1, E, N), 0.03)
    for v in range(V):
        p[v, :, v] = 0.79
    p[V] = 1.0 / N
    return p


def test_perfect_identity_protocol_passes_g3_and_g4():
    bank = _bank(torch.arange(E) % N, _faithful_receiver())
    g3 = gate_g3(bank, permutations=200)
    assert g3["strong"] and g3["weak"] and g3["passed"]
    assert g3["acc"] == 1.0
    assert g3["mi_raw"] == pytest.approx(3.0, abs=1e-9)
    # Debiasing subtracts the permutation-null mean even at ceiling; PROTOCOL 2.4
    # puts the bias at (m_XY - m_X - m_Y + 1)/(2n ln2) ~= 0.069 bits here.
    assert g3["mi_debiased"] == pytest.approx(3.0 - 0.069, abs=0.02)
    assert g3["v_eff"] == pytest.approx(8.0, abs=1e-6)
    # Factored referents, identity code: each bit carries exactly 1 bit.
    assert all(v == pytest.approx(1.0, abs=0.05) for v in g3["attr_mi"])

    g4 = gate_g4(bank, mi_mr_debiased=g3["mi_debiased"], permutations=200)
    assert g4["passed"], g4["checks"]
    # Answer given m_e is near-one-hot vs a near-uniform marginal baseline:
    # KL approaches log2 N. Hand value: .79*log2(.79/.125) + 7*.03*log2(.03/.125)
    hand_cic = 0.79 * math.log2(0.79 / 0.125) + 7 * 0.03 * math.log2(0.03 / 0.125)
    assert g4["cic_bits"] == pytest.approx(hand_cic, abs=1e-6)


def test_collapsed_sender_fails_g3():
    bank = _bank(torch.zeros(E, dtype=torch.long), _faithful_receiver())
    g3 = gate_g3(bank, permutations=200)
    assert not g3["strong"] and not g3["weak"]
    assert g3["v_eff"] == pytest.approx(1.0, abs=1e-6)
    assert g3["mi_debiased"] == pytest.approx(0.0, abs=0.05)


def test_deaf_receiver_fails_g4_via_cic_and_interventions_do_nothing():
    """A receiver that ignores the symbol: same answer distribution under every
    counterfactual. Interventions cannot move accuracy, CIC must be ~0."""
    p = torch.full((V + 1, E, N), 0.03)
    p[:, :, 0] = 0.79  # always answers referent 0, whatever it hears
    bank = _bank(torch.arange(E) % N, p)
    g4 = gate_g4(bank, mi_mr_debiased=3.0, permutations=200)
    assert g4["cic_bits"] == pytest.approx(0.0, abs=1e-9)
    assert not g4["checks"]["cic"]
    # Accuracy under removal equals accuracy under the real symbol: 1/N.
    assert g4["acc_removed"] == pytest.approx(1 / N, abs=1e-6)
    assert g4["cse"] == 0.0


def test_g4_permutation_is_never_identity_and_purity_needs_all_symbols():
    bank = _bank(torch.arange(E) % N, _faithful_receiver())
    g4 = gate_g4(bank, mi_mr_debiased=3.0, permutations=200)
    # A faithful receiver under a non-identity permutation answers pi(m) != r
    # for every referent the permutation moves: accuracy far below 0.225.
    assert g4["acc_permuted"] <= 0.225

    seven = torch.arange(E) % (N - 1)  # symbol 7 never used
    g4 = gate_g4(_bank(seven, _faithful_receiver()), mi_mr_debiased=3.0,
                 permutations=200)
    assert not g4["checks"]["purity"]
