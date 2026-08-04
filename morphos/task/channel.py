"""The discrete communication channel.

A choice among V symbols has zero gradient everywhere -- nudge the sender's
preferences and the argmax either does not move or jumps. So the receiver's error
cannot reach the sender at all. Straight-through Gumbel-Softmax is the standard
fix: sample a near-one-hot relaxation, snap it to hard one-hot on the forward
pass, and let the *soft* vector carry the gradient backward.

The forward pass must be genuinely discrete. If a soft vector ever reaches the
receiver, the sender can smuggle unbounded information through the decimal places
and every information-theoretic claim in the paper is void -- I(M;R), channel
capacity, the whole lot. `assert_discrete` exists to make that a checked
invariant rather than an assumption, and gate G4 calls it.

Havrylov & Titov report straight-through Gumbel converging faster and to better
protocols than REINFORCE at this scale, which is why it is the default here;
REINFORCE remains the honest robustness check because it optimises the truly
discrete objective with no relaxation mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

EPS = 1e-10


def gumbel_noise(shape, *, generator: torch.Generator, device, dtype=torch.float32) -> Tensor:
    """-log(-log(U)), the standard Gumbel(0,1) sample."""
    u = torch.rand(shape, generator=generator, device=device, dtype=dtype)
    return -torch.log(-torch.log(u.clamp(min=EPS, max=1.0 - EPS)) + EPS)


def straight_through(soft: Tensor) -> Tensor:
    """Hard one-hot forward, soft gradient backward.

    `hard - soft.detach() + soft` is exactly `hard` numerically, but its gradient
    is the gradient of `soft`. That identity is the whole trick.
    """
    idx = soft.argmax(dim=-1, keepdim=True)
    hard = torch.zeros_like(soft).scatter_(-1, idx, 1.0)
    return hard - soft.detach() + soft


@dataclass(frozen=True)
class GumbelChannel:
    """A V-symbol bottleneck between two organisms."""

    vocab: int
    tau_start: float = 2.0
    tau_end: float = 0.5
    anneal_steps: int = 5000

    def temperature(self, step: int) -> float:
        """Anneal high -> low. High tau explores; low tau is near-discrete but
        higher-variance. Annealing gets exploration early and fidelity late."""
        if self.anneal_steps <= 0:
            return self.tau_end
        frac = min(1.0, max(0.0, step / self.anneal_steps))
        return self.tau_start + frac * (self.tau_end - self.tau_start)

    def __call__(
        self,
        logits: Tensor,
        *,
        generator: torch.Generator,
        step: int = 0,
        hard: bool = True,
        noise: bool = True,
    ) -> Tensor:
        """(B, V) logits -> (B, V) symbol vector.

        `noise=False, hard=True` is the deterministic eval path: pure argmax, no
        Gumbel draw. Evaluation must be deterministic so that a drop in the
        sender's *confidence* cannot masquerade as a change in *meaning*.
        """
        tau = self.temperature(step)
        z = logits
        if noise:
            z = z + gumbel_noise(
                logits.shape, generator=generator, device=logits.device, dtype=logits.dtype
            )
        soft = F.softmax(z / tau, dim=-1)
        return straight_through(soft) if hard else soft

    def sample_index(
        self, logits: Tensor, *, generator: torch.Generator, noise: bool = False
    ) -> Tensor:
        """(B,) integer symbol. Deterministic argmax unless `noise=True`."""
        z = logits
        if noise:
            z = z + gumbel_noise(
                logits.shape, generator=generator, device=logits.device, dtype=logits.dtype
            )
        return z.argmax(dim=-1)


def assert_discrete(symbols: Tensor, vocab: int) -> None:
    """Verify the channel really is a V-symbol bottleneck.

    Gate G4 calls this. A violation means an analog side channel exists and every
    discrete metric downstream is invalid, so this raises rather than warns.
    """
    if symbols.shape[-1] != vocab:
        raise ValueError(f"expected last dim {vocab}, got {symbols.shape[-1]}")

    is_onehot = torch.allclose(
        symbols.sum(-1), torch.ones_like(symbols.sum(-1)), atol=1e-5
    ) and torch.all((symbols < 1e-5) | (symbols > 1 - 1e-5))
    if not is_onehot:
        raise ValueError(
            "channel carried a non-one-hot vector: the sender can smuggle "
            "continuous information and all discrete metrics are invalid"
        )

    distinct = torch.unique(symbols.argmax(-1)).numel()
    if distinct > vocab:
        raise ValueError(f"{distinct} distinct symbols exceeds vocabulary {vocab}")


def symbol_stats(indices: Tensor, vocab: int) -> dict[str, float]:
    """Entropy and effective vocabulary size.

    Collapse -- the sender emitting one symbol for everything -- yields chance
    accuracy, which is indistinguishable from "not learning yet" unless entropy is
    logged. Miller-Madow corrects the plug-in estimator's downward bias.
    """
    # MPS has no float64: count on device, widen on CPU.
    counts = torch.bincount(indices.flatten(), minlength=vocab).cpu().double()
    n = counts.sum().clamp(min=1)
    p = counts / n
    nz = p[p > 0]
    h = float(-(nz * nz.log2()).sum())
    h_mm = h + (len(nz) - 1) / (2 * float(n) * torch.log(torch.tensor(2.0)).item())
    return {
        "entropy": h,
        "entropy_mm": h_mm,
        "v_eff": float(2**h),
        "v_used": int((counts > n / (10 * vocab)).sum()),
        "max_entropy": float(torch.log2(torch.tensor(float(vocab)))),
    }
