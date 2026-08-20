"""Tests for the discrete channel and the Lewis game.

`test_forward_pass_is_genuinely_one_hot` is the load-bearing one. If a soft vector
ever reaches the receiver, the sender can smuggle unbounded information through
the decimals and every information-theoretic claim -- I(M;R), channel capacity,
the emergence gate -- becomes meaningless.
"""

from __future__ import annotations

import pytest
import torch

from morphos.seeding import make_rng
from morphos.substrate.nca import RECEIVER_LAYOUT, SENDER_LAYOUT, NCAOrganism
from morphos.task.channel import (
    GumbelChannel,
    assert_discrete,
    straight_through,
    symbol_stats,
)
from morphos.task.lewis import accuracy, run_episode, task_loss
from morphos.task.readout import VotePool

V = 8
CH = GumbelChannel(vocab=V, tau_start=2.0, tau_end=0.5, anneal_steps=100)


# --- channel ------------------------------------------------------------------


def test_forward_pass_is_genuinely_one_hot():
    """The bottleneck must be discrete or every discrete metric is void."""
    logits = torch.randn(64, V)
    g = torch.Generator().manual_seed(0)
    sym = CH(logits, generator=g, step=0, hard=True)

    assert torch.allclose(sym.sum(-1), torch.ones(64))
    assert torch.all((sym == 0) | (sym == 1)), "channel emitted a non-binary value"
    assert_discrete(sym, V)


def test_assert_discrete_catches_a_soft_vector():
    soft = torch.softmax(torch.randn(16, V), dim=-1)
    with pytest.raises(ValueError, match="smuggle"):
        assert_discrete(soft, V)


def test_gradient_flows_through_the_discrete_bottleneck():
    """Hard forward, soft backward. A misplaced detach here silently turns the
    channel into a no-op and the sender never learns anything."""
    logits = torch.randn(8, V, requires_grad=True)
    sym = CH(logits, generator=torch.Generator().manual_seed(0), step=0, hard=True)
    sym.sum().backward()

    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0, "no gradient reached the sender"
    assert torch.isfinite(logits.grad).all()


def test_straight_through_is_numerically_hard():
    soft = torch.tensor([[0.1, 0.7, 0.2]])
    out = straight_through(soft)
    assert torch.equal(out, torch.tensor([[0.0, 1.0, 0.0]]))


def test_temperature_anneals_between_endpoints():
    assert CH.temperature(0) == pytest.approx(2.0)
    assert CH.temperature(100) == pytest.approx(0.5)
    assert CH.temperature(10_000) == pytest.approx(0.5)
    assert CH.temperature(50) == pytest.approx(1.25)


def test_eval_path_is_deterministic():
    """Evaluation must not sample: a drop in the sender's confidence would
    otherwise be indistinguishable from a change in meaning."""
    logits = torch.randn(32, V)
    a = CH(logits, generator=torch.Generator().manual_seed(1), noise=False)
    b = CH(logits, generator=torch.Generator().manual_seed(99), noise=False)
    assert torch.equal(a, b)
    assert torch.equal(a.argmax(-1), logits.argmax(-1))


def test_noise_actually_explores():
    """Without the Gumbel draw the sender would lock onto whatever it emitted
    first instead of trying alternatives."""
    logits = torch.tensor([[2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3]]).repeat(200, 1)
    g = torch.Generator().manual_seed(0)
    picks = CH(logits, generator=g, step=0, hard=True).argmax(-1)
    assert picks.unique().numel() > 3, "Gumbel noise is not exploring"


def test_symbol_stats_detect_collapse():
    """Collapse gives chance accuracy, which is indistinguishable from 'not
    learning yet' unless entropy is tracked."""
    collapsed = torch.zeros(256, dtype=torch.long)
    s = symbol_stats(collapsed, V)
    assert s["entropy"] == pytest.approx(0.0, abs=1e-9)
    assert s["v_used"] == 1

    uniform = torch.arange(256) % V
    s = symbol_stats(uniform, V)
    assert s["entropy"] == pytest.approx(3.0, abs=1e-6)  # log2(8)
    assert s["v_eff"] == pytest.approx(8.0, abs=1e-4)
    assert s["v_used"] == V


# --- the game -----------------------------------------------------------------


def build_pair(grid=16, hidden=32, device="cpu"):
    # Weights draw from the GLOBAL stream: without a seed the outcome depends on
    # whichever test ran before, and an unlucky init dies within 8 steps and
    # returns an all-zero vote. Rollout RNG is separately seeded via make_rng.
    torch.manual_seed(7)
    s = NCAOrganism(layout=SENDER_LAYOUT, hidden=hidden, grid=grid).to(device)
    r = NCAOrganism(layout=RECEIVER_LAYOUT, hidden=hidden, grid=grid).to(device)
    for m in (s, r):
        torch.nn.init.normal_(m.fc2.weight, std=0.2)
    return s, r


def run(grid=16, n_ref=8, **kw):
    dev = torch.device("cpu")
    s, r = build_pair(grid=grid)
    rng = make_rng(0, dev)
    refs = torch.arange(n_ref) % n_ref
    return run_episode(
        s, r,
        referents=refs,
        sender_state=s.seed_state(n_ref, device=dev),
        receiver_state=r.seed_state(n_ref, device=dev),
        sender_readout=VotePool(channels=s.layout.vote, n_out=V),
        receiver_readout=VotePool(channels=r.layout.vote, n_out=n_ref),
        channel=CH, rng=rng, n_referents=n_ref,
        t_sender=8, t_receiver=8, t_inject=4, grid=grid, **kw
    ), s, r


def test_episode_runs_and_shapes_are_right():
    ep, _, _ = run()
    assert ep.symbol.shape == (8, V)
    assert ep.symbol_idx.shape == (8,)
    assert ep.answer_probs.shape == (8, 8)
    assert torch.allclose(ep.answer_probs.sum(-1), torch.ones(8), atol=1e-5)
    assert_discrete(ep.symbol, V)


def test_ear_injection_is_bipolar_full_energy():
    """A raw one-hot is 7/8 zeros and a written zero is indistinguishable from no
    injection (the referent-code argument, broadcast.py). The ear must therefore
    write bipolar {-1,+1}: every symbol a full-energy pattern, still exactly V
    distinct ones."""
    from morphos.task.lewis import make_ear_inject_fn

    grid, patch = 16, 3
    x = torch.zeros(2, RECEIVER_LAYOUT.total, grid, grid)
    sym = torch.zeros(2, V)
    sym[0, 3] = 1.0
    sym[1, 5] = 1.0

    inj = make_ear_inject_fn(sym, RECEIVER_LAYOUT, grid, t_inject=4, patch=patch)
    out = inj(x, t=0)
    ear = out[:, RECEIVER_LAYOUT.sensor, grid // 2, grid // 2]

    assert torch.all((ear == 1.0) | (ear == -1.0)), "ear write must be bipolar"
    assert ear[0, 3] == 1.0 and ear[0].sum() == 2 - V  # one +1, seven -1
    assert not torch.equal(ear[0], ear[1]), "distinct symbols, distinct patterns"
    assert torch.equal(inj(x, t=4), x), "writing must stop at t_inject"


def test_receiver_sees_only_the_symbol():
    """The core anti-cheat: identical symbols must produce identical receiver
    behaviour regardless of what the sender was looking at. If the referent
    reached the receiver by any other path, this fails."""
    dev = torch.device("cpu")
    s, r = build_pair()
    ro = VotePool(channels=r.layout.vote, n_out=8)

    fixed = torch.zeros(4, V)
    fixed[:, 3] = 1.0
    outs = []
    for refs in (torch.tensor([0, 1, 2, 3]), torch.tensor([7, 6, 5, 4])):
        ep = run_episode(
            s, r, referents=refs,
            sender_state=s.seed_state(4, device=dev),
            receiver_state=r.seed_state(4, device=dev),
            sender_readout=VotePool(channels=s.layout.vote, n_out=V),
            receiver_readout=ro, channel=CH, rng=make_rng(0, dev),
            n_referents=8, t_sender=8, t_receiver=8, t_inject=4, grid=16,
            override_symbol=fixed,
        )
        outs.append(ep.answer_probs)
    assert torch.allclose(outs[0], outs[1], atol=1e-6), (
        "receiver output depends on the referent even with the symbol held fixed "
        "-- there is a leak"
    )


def test_override_symbol_changes_the_answer():
    """Sanity in the other direction: if overriding the symbol changed nothing,
    the receiver would be ignoring the channel and every intervention is vacuous."""
    dev = torch.device("cpu")
    s, r = build_pair()
    ro = VotePool(channels=r.layout.vote, n_out=8)
    kw = dict(
        sender_readout=VotePool(channels=s.layout.vote, n_out=V), receiver_readout=ro,
        channel=CH, n_referents=8, t_sender=8, t_receiver=12, t_inject=6, grid=16,
    )
    probs = []
    for k in (0, 5):
        sym = torch.zeros(4, V)
        sym[:, k] = 1.0
        ep = run_episode(
            s, r, referents=torch.zeros(4, dtype=torch.long),
            sender_state=s.seed_state(4, device=dev),
            receiver_state=r.seed_state(4, device=dev),
            rng=make_rng(0, dev), override_symbol=sym, **kw,
        )
        probs.append(ep.answer_probs)
    assert not torch.allclose(probs[0], probs[1], atol=1e-6), (
        "different symbols produced identical receiver output -- channel is inert"
    )


def test_task_loss_and_accuracy_are_consistent():
    ep, _, _ = run()
    loss = task_loss(ep, 8)
    assert loss.item() >= 0 and torch.isfinite(loss)
    acc = accuracy(ep)
    assert 0.0 <= acc <= 1.0


def test_gradient_reaches_both_organisms():
    """The most likely week-one bug: a silently detached sender that never learns.
    Both parameter sets must receive gradient from the shared loss."""
    ep, s, r = run()
    task_loss(ep, 8).backward()
    for name, m in (("sender", s), ("receiver", r)):
        g = m.fc1.weight.grad
        assert g is not None and g.abs().sum() > 0, f"no gradient reached the {name}"
