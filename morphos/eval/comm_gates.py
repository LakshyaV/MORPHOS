"""Gates G3 (a protocol emerged) and G4 (the protocol is causally real).

Definitions are pre-registered in docs/PROTOCOL.md §8; estimators in §2.3-2.5.
Everything here evaluates held-out episodes: fresh organisms grown with a salted
RNG, deterministic argmax channel (a drop in the sender's *confidence* must not
masquerade as a change in *meaning*).

The expensive object is `EpisodeBank.probs_by_v`: for every episode, the
receiver's answer distribution under ALL V counterfactual symbols plus silence,
with the receiver's initial state and RNG stream held fixed within the episode.
Because m is thereby randomized within episode, conditioning equals do(m), and
every G4 intervention (removal, marginal resampling, vocabulary permutation,
CIC) is pure indexing into this tensor -- one set of rollouts, no re-runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from morphos.seeding import make_rng
from morphos.task.broadcast import make_inject_fn, referent_codes, sample_referents
from morphos.task.channel import GumbelChannel, assert_discrete, symbol_stats
from morphos.task.lewis import make_ear_inject_fn
from morphos.task.readout import VotePool

GATE_SALT = 0xC0FF


def wilson_lower_bound(k: int, n: int, z: float = 1.96) -> float:
    """Lower bound of the Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / denom


def mi_bits(a: Tensor, b: Tensor, na: int, nb: int) -> float:
    """Plug-in mutual information in bits. CPU double: MPS has no float64."""
    a, b = a.cpu().long(), b.cpu().long()
    joint = torch.bincount(a * nb + b, minlength=na * nb).double().reshape(na, nb)
    pj = joint / joint.sum()
    outer = pj.sum(1, keepdim=True) @ pj.sum(0, keepdim=True)
    nz = pj > 0
    return float((pj[nz] * (pj[nz] / outer[nz]).log2()).sum())


def debiased_mi(
    a: Tensor, b: Tensor, na: int, nb: int, *, permutations: int = 1000, seed: int = 0
) -> dict[str, float]:
    """Permutation-debiased MI (PROTOCOL §2.4): plug-in MI is upward-biased by
    ~(m_XY - m_X - m_Y + 1)/(2n ln2) bits, ~0.069 bits at N=V=8, n=512 -- not
    negligible against the effects under study. Subtract the permutation-null
    mean; the exact p-value comes free."""
    raw = mi_bits(a, b, na, nb)
    g = torch.Generator().manual_seed(seed)
    bc = b.cpu()
    null = torch.tensor([
        mi_bits(a, bc[torch.randperm(len(bc), generator=g)], na, nb)
        for _ in range(permutations)
    ])
    return {
        "mi_raw": raw,
        "mi_debiased": raw - float(null.mean()),
        "p_value": float(((null >= raw).sum() + 1) / (permutations + 1)),
    }


@dataclass
class EpisodeBank:
    """Held-out episodes with the full counterfactual answer table."""

    referents: Tensor  # (E,) int
    symbol_idx: Tensor  # (E,) int, deterministic argmax channel
    symbol_idx_noisy: Tensor  # (E,) int, Gumbel draw at tau_end (reported, not gated)
    probs_by_v: Tensor  # (V+1, E, N) receiver answer distributions; slot V = silence
    n_referents: int
    vocab: int

    @property
    def answer_probs(self) -> Tensor:  # (E, N) under the actually-emitted symbol
        return self.probs_by_v[self.symbol_idx, torch.arange(len(self.symbol_idx))]

    @property
    def answer_idx(self) -> Tensor:  # (E,)
        return self.answer_probs.argmax(-1)


def _receiver_probs(
    receiver, readout: VotePool, r0: Tensor, symbol: Tensor, *,
    seed: int, t_episode: int, t_inject: int, grid: int, patch: int,
) -> Tensor:
    g = torch.Generator(device=r0.device).manual_seed(seed)
    inj = make_ear_inject_fn(symbol, receiver.layout, grid, t_inject, patch=patch)
    out = receiver.rollout(r0, t_episode, generator=g, inject_fn=inj)
    return readout(out, receiver.alive_mask(out))


@torch.no_grad()
def collect_episodes(
    sender, receiver, cfg: dict, *,
    episodes: int = 512, batch: int = 64, seed: int = 0,
    device: torch.device | str = "cpu",
) -> EpisodeBank:
    device = torch.device(device)
    grid = cfg["nca"]["grid"]
    V, N = cfg["task"]["vocab"], cfg["task"]["n_referents"]
    t_inject, patch = cfg["task"]["t_inject"], cfg["task"]["patch"]
    t_grow = cfg["eval"]["t_grow"]
    t_episode = cfg["train"]["rollout_min"]  # fixed, not U[min,max]: eval is deterministic

    s_read = VotePool(channels=sender.layout.vote, n_out=V)
    r_read = VotePool(channels=receiver.layout.vote, n_out=N)
    channel = GumbelChannel(
        vocab=V, tau_start=cfg["task"]["tau_start"], tau_end=cfg["task"]["tau_end"],
        anneal_steps=cfg["task"]["tau_anneal_steps"],
    )
    rng = make_rng(seed ^ GATE_SALT, device)
    cpu_g = torch.Generator().manual_seed(seed ^ GATE_SALT)
    codes = referent_codes(N, sender.layout.n_sensor, device=device)
    eye = torch.eye(V, device=device)

    refs, sids, sids_noisy, probs = [], [], [], []
    done = 0
    while done < episodes:
        B = min(batch, episodes - done)
        s0 = sender.rollout(sender.seed_state(B, device=device), t_grow, generator=rng.update)
        r0 = receiver.rollout(receiver.seed_state(B, device=device), t_grow, generator=rng.update)
        ref = sample_referents(B, N, generator=rng.task, device=device)

        s_inj = make_inject_fn(codes[ref], sender.layout, grid, t_inject, patch=patch)
        s_out = sender.rollout(s0, t_episode, generator=rng.update, inject_fn=s_inj)
        logits = s_read.logits(s_out, sender.alive_mask(s_out))
        sym = channel(logits, generator=rng.task, hard=True, noise=False)
        assert_discrete(sym, V)  # channel purity is asserted on every batch
        sym_noisy = channel(
            logits, generator=rng.task, step=cfg["task"]["tau_anneal_steps"],
            hard=True, noise=True,
        )

        # One RNG seed per batch, shared by all V+1 receiver rollouts: within an
        # episode only the symbol varies, which is what makes CIC causal.
        r_seed = int(torch.randint(2**31 - 1, (1,), generator=cpu_g).item())
        counterfactuals = list(eye.unsqueeze(1).expand(V, B, V)) + [torch.zeros(B, V, device=device)]
        pv = torch.stack([
            _receiver_probs(
                receiver, r_read, r0, s, seed=r_seed,
                t_episode=t_episode, t_inject=t_inject, grid=grid, patch=patch,
            )
            for s in counterfactuals
        ])

        refs.append(ref.cpu())
        sids.append(sym.argmax(-1).cpu())
        sids_noisy.append(sym_noisy.argmax(-1).cpu())
        probs.append(pv.cpu())
        done += B

    return EpisodeBank(
        referents=torch.cat(refs),
        symbol_idx=torch.cat(sids),
        symbol_idx_noisy=torch.cat(sids_noisy),
        probs_by_v=torch.cat(probs, dim=1),
        n_referents=N,
        vocab=V,
    )


def gate_g3(bank: EpisodeBank, *, permutations: int = 1000, seed: int = 0) -> dict:
    E, N, V = len(bank.referents), bank.n_referents, bank.vocab
    k = int((bank.answer_idx == bank.referents).sum())
    acc, lb = k / E, wilson_lower_bound(k, E)
    mi = debiased_mi(bank.symbol_idx, bank.referents, V, N,
                     permutations=permutations, seed=seed)
    stats = symbol_stats(bank.symbol_idx, V)
    h_r = float(symbol_stats(bank.referents, N)["entropy"])

    acc_noisy = float(
        (bank.probs_by_v[bank.symbol_idx_noisy, torch.arange(E)].argmax(-1)
         == bank.referents).double().mean()
    )
    # Attribute-wise MI (referents are factored bits): far more informative than
    # a degenerate topsim at L=1 (PROTOCOL §2.6).
    n_bits = int(math.log2(N))
    attr_mi = [
        debiased_mi(bank.symbol_idx, (bank.referents >> b) & 1, V, 2,
                    permutations=permutations, seed=seed + 1 + b)["mi_debiased"]
        for b in range(n_bits)
    ]

    need_mi = 0.8 * math.log2(N)
    strong = acc >= 0.90 and lb >= 0.85 and mi["mi_debiased"] >= need_mi and stats["v_eff"] >= 0.75 * V
    weak = acc >= 0.50 and mi["mi_debiased"] >= 1.0
    return {
        "gate": "G3", "episodes": E,
        "acc": acc, "acc_wilson_lb": lb, "acc_noisy_channel": acc_noisy,
        **mi, "nmi": mi["mi_debiased"] / h_r if h_r > 0 else 0.0,
        "attr_mi": attr_mi,
        "entropy": stats["entropy"], "entropy_mm": stats["entropy_mm"],
        "v_eff": stats["v_eff"], "v_used": stats["v_used"],
        "strong": bool(strong), "weak": bool(weak), "passed": bool(strong),
    }


def gate_g4(
    bank: EpisodeBank, *, mi_mr_debiased: float,
    permutations: int = 1000, seed: int = 0,
) -> dict:
    E, N, V = len(bank.referents), bank.n_referents, bank.vocab
    idx = torch.arange(E)
    acc_of = lambda ans: float((ans == bank.referents).double().mean())
    g = torch.Generator().manual_seed(seed ^ 0x5EED)

    acc_removed = acc_of(bank.probs_by_v[V, idx].argmax(-1))

    marginal = torch.bincount(bank.symbol_idx, minlength=V).double()
    marginal = marginal / marginal.sum()
    v_resampled = torch.multinomial(marginal, E, replacement=True, generator=g)
    acc_resampled = acc_of(bank.probs_by_v[v_resampled, idx].argmax(-1))

    while True:
        pi = torch.randperm(V, generator=g)
        if not torch.equal(pi, torch.arange(V)):
            break
    acc_permuted = acc_of(bank.probs_by_v[pi[bank.symbol_idx], idx].argmax(-1))

    # CIC = E_e KL[ p(a|m_e,e) || sum_v marginal(v) p(a|v,e) ], KL on the softmax
    # (here: pooled vote distribution), never the argmax (PROTOCOL §2.5).
    eps = 1e-12
    p_act = bank.answer_probs.double().clamp_min(eps)
    baseline = (marginal.view(V, 1, 1) * bank.probs_by_v[:V].double()).sum(0).clamp_min(eps)
    cic = float((p_act * (p_act / baseline).log2()).sum(-1).mean())
    cse = float((bank.probs_by_v[:V].argmax(-1) != bank.answer_idx.unsqueeze(0)).double().mean())

    mi_ar = debiased_mi(bank.answer_idx, bank.referents, N, N,
                        permutations=permutations, seed=seed + 7)
    dpi_ok = mi_ar["mi_debiased"] <= mi_mr_debiased + 0.1
    purity_ok = int(torch.unique(bank.symbol_idx).numel()) == V  # one-hot asserted at collect

    chance = 1.0 / N
    checks = {
        "removed": acc_removed <= chance + 0.05,
        "resampled": acc_resampled <= chance + 0.05,
        "permuted": acc_permuted <= chance + 0.10,
        "cic": cic >= 1.0,
        "dpi": dpi_ok,
        "purity": purity_ok,
    }
    return {
        "gate": "G4", "episodes": E,
        "acc_removed": acc_removed, "acc_resampled": acc_resampled,
        "acc_permuted": acc_permuted,
        "cic_bits": cic, "cse": cse,
        "mi_answer_referent": mi_ar["mi_debiased"],
        "checks": checks, "passed": bool(all(checks.values())),
        "crossplay": "pending -- needs >=2 seeds",
    }
