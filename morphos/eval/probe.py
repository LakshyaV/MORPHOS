"""Propagation probe: can this substrate carry information across this body?

This answers the single assumption the whole communication phase rests on --
that T_comm is long enough for a signal injected at the sensor patch to reach the
rest of the organism. It needs no Lewis game, no vote channels, no Gumbel and no
receiver, so it can be run the same day the body first trains.

The key design choice is measuring an effect size over (distance, time) rather
than accuracy-vs-T. Accuracy-vs-T conflates two very different failures:

    SLOW        -- the signal arrives, just later than T_comm.  Fix: raise T.
    ATTENUATING -- the signal never reaches distant cells at any time.
                   Fix: longer injection, larger amplitude, or a hidden-channel
                   carrier. Raising T does nothing, and you could spend a week
                   discovering that.

Codes are bipolar {-1,+1}, never {0,1}: with {0,1} the all-zero code injects
nothing and is indistinguishable from no injection at all, which corrupts the
variance decomposition. This choice propagates into the Lewis game.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from morphos.eval.metrics import body_mask, geodesic_distance
from morphos.seeding import RNG
from morphos.substrate.nca import ChannelLayout, NCAOrganism

ETA2_THRESHOLD = 0.05


@dataclass
class ProbeResult:
    eta2: Tensor  # (T, H, W) fraction of state variance explained by the code
    dist: Tensor  # (H, W) geodesic distance from the sensor patch, +inf off-body
    eta2_by_dist: Tensor  # (T, D)
    front: Tensor  # (T,) furthest distance ring above threshold
    speed: float  # cells/step
    t_cover: float  # first t with >=95% of body above threshold
    eta2_max_by_dist: Tensor  # (D,) attenuation curve
    max_dist: int

    def verdict(self, t_comm: int = 64) -> str:
        far = self.eta2_max_by_dist[min(6, self.max_dist) :]
        if far.numel() and float(far.max()) < ETA2_THRESHOLD:
            return "ATTENUATING"
        if self.t_cover <= t_comm / 2:
            return "OK"
        if self.t_cover <= t_comm:
            return "MARGINAL"
        return "TOO_SHORT"

    def summary(self, t_comm: int = 64) -> str:
        v = self.verdict(t_comm)
        note = {
            "OK": f"signal covers the body by t={self.t_cover:.0f}, leaving the rest "
                  f"of T_comm={t_comm} for consensus to form",
            "MARGINAL": f"coverage at t={self.t_cover:.0f} is late in T_comm={t_comm}; "
                        "raise T_comm to 96 (memory is flat in rollout length)",
            "TOO_SHORT": f"coverage at t={self.t_cover:.0f} exceeds T_comm={t_comm}; "
                         "ladder T_comm 64->96->128 before shrinking the body",
            "ATTENUATING": "the signal never reaches distant cells at ANY time; "
                           "more time cannot help -- raise injection duration or "
                           "amplitude, or add a hidden-channel carrier",
        }[v]
        return (
            f"verdict: {v}\n"
            f"  front speed      {self.speed:.2f} cells/step\n"
            f"  t_cover          {self.t_cover:.0f}\n"
            f"  max geodesic     {self.max_dist}\n"
            f"  {note}"
        )


def sensor_patch_mask(grid: int, size: int = 3, *, device="cpu") -> Tensor:
    """(1,1,H,W) bool over the central size x size patch."""
    m = torch.zeros(1, 1, grid, grid, dtype=torch.bool, device=device)
    c, r = grid // 2, size // 2
    m[..., c - r : c + r + 1, c - r : c + r + 1] = True
    return m


def bipolar_codes(n_bits: int = 3, *, device="cpu") -> Tensor:
    """All 2**n_bits codes in {-1,+1}. -> (R, n_bits)"""
    r = 2**n_bits
    bits = ((torch.arange(r, device=device).unsqueeze(1) >> torch.arange(n_bits, device=device)) & 1)
    return bits.float() * 2.0 - 1.0


def _eta_squared(states: Tensor, n_codes: int, n_noise: int) -> Tensor:
    """Per-cell effect size of the code on the state.

    states: (R*K, C, H, W) -> (H, W) in [0,1]. Between-group variance over the
    total, i.e. the fraction of each cell's variance explained by which code was
    injected.
    """
    # Computed in float32 on-device (MPS has no float64) and widened only at the
    # end. Exactness where it matters is preserved: if the code has not reached a
    # cell its states are bit-identical across codes, so the between-group term is
    # exactly 0 in float32 too -- which is what the light-cone test asserts.
    c, h, w = states.shape[1:]
    s = states.view(n_codes, n_noise, c, h, w)
    mu_r = s.mean(dim=1)  # (R,C,H,W)
    mu = mu_r.mean(dim=0, keepdim=True)  # (1,C,H,W)

    between = ((mu_r - mu) ** 2).mean(dim=0).sum(dim=0)  # (H,W)
    within = ((s - mu_r.unsqueeze(1)) ** 2).mean(dim=(0, 1)).sum(dim=0)  # (H,W)
    return (between / (between + within).clamp(min=1e-12)).cpu().double()


@torch.no_grad()
def run_probe(
    model: NCAOrganism,
    layout: ChannelLayout,
    *,
    rng: RNG,
    device: torch.device,
    t_grow: int = 64,
    t_probe: int = 128,
    t_inject: int = 16,
    n_noise: int = 8,
    patch: int = 3,
) -> ProbeResult:
    """Grow a body, inject each code at the sensor patch, watch it spread."""
    grid = model.grid
    codes = bipolar_codes(layout.n_sensor, device=device)  # (R, n_sensor)
    n_codes = codes.shape[0]
    batch = n_codes * n_noise

    # Grow only n_noise bodies, then tile each across all codes. Every code then
    # starts from a bit-identical body, so the only difference between groups is
    # the injected code itself. Growing all R*K independently would make the body
    # differ across codes and inflate the between-group variance.
    grown = model.rollout(
        model.seed_state(n_noise, device=device), t_grow, generator=rng.update
    )
    x = grown.repeat(n_codes, 1, 1, 1).contiguous()  # index r*K+k -> grown[k]

    alive = body_mask(x[:1])
    src = sensor_patch_mask(grid, patch, device=device) & alive
    dist = geodesic_distance(alive, src)[0, 0]
    finite = dist[torch.isfinite(dist)]
    max_dist = int(finite.max().item()) if finite.numel() else 0

    c, r = grid // 2, patch // 2
    sl = (slice(c - r, c + r + 1), slice(c - r, c + r + 1))
    code_batch = codes.repeat_interleave(n_noise, dim=0)  # (B, n_sensor)

    eta_frames = []
    for t in range(t_probe):
        if t < t_inject:
            # Hard write. At t == t_inject we simply STOP writing; zeroing instead
            # would be a second injection event.
            x[:, layout.sensor, sl[0], sl[1]] = code_batch.view(batch, -1, 1, 1)
        eta_frames.append(_eta_squared(x, n_codes, n_noise))

        # One mask per NOISE draw, shared across all codes. Without this, two codes
        # differ by their fire masks as well as by the code, and the between-group
        # variance no longer isolates the code's effect.
        base = model.sample_fire_mask(n_noise, rng.update, device)
        x = model.rollout(x, 1, generator=rng.update,
                          fire_masks=[base.repeat(n_codes, 1, 1, 1)])

    eta2 = torch.stack(eta_frames)  # (T,H,W)
    dist_cpu = dist.cpu()

    rings = torch.arange(max_dist + 1)
    by_dist = torch.zeros(eta2.shape[0], max_dist + 1, dtype=torch.float64)
    for d in rings:
        m = dist_cpu == d
        if m.any():
            by_dist[:, d] = eta2[:, m].mean(dim=1)

    above = by_dist >= ETA2_THRESHOLD
    front = torch.where(
        above.any(dim=1),
        (above * rings.unsqueeze(0)).max(dim=1).values.double(),
        torch.zeros(eta2.shape[0], dtype=torch.float64),
    )

    body = torch.isfinite(dist_cpu) & alive[0, 0].cpu()
    covered = (eta2[:, body] >= ETA2_THRESHOLD).double().mean(dim=1)
    hit = (covered >= 0.95).nonzero()
    t_cover = float(hit[0].item()) if hit.numel() else float("inf")

    window = min(t_inject + 32, eta2.shape[0])
    ts = torch.arange(window, dtype=torch.float64)
    f = front[:window]
    speed = float(
        ((ts - ts.mean()) * (f - f.mean())).sum() / ((ts - ts.mean()) ** 2).sum().clamp(min=1e-9)
    )

    return ProbeResult(
        eta2=eta2,
        dist=dist_cpu,
        eta2_by_dist=by_dist,
        front=front,
        speed=speed,
        t_cover=t_cover,
        eta2_max_by_dist=by_dist.max(dim=0).values,
        max_dist=max_dist,
    )
