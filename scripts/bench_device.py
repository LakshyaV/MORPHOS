"""CPU vs MPS decision table for the MORPHOS workload.

The NCA is many small ops on small tensors, so kernel-launch overhead is a real
concern and the answer is not obvious a priori -- measure it rather than assume.

Decision rule: use whichever device wins the 2-organism communication step.
If they are within 20%, prefer CPU, because same-device determinism is easier to
reason about and MPS lacks float64.

Re-run this after any torch upgrade and on new hardware.
"""

from __future__ import annotations

import time

import torch

from morphos.substrate.nca import RECEIVER_LAYOUT, SENDER_LAYOUT, NCAOrganism

BATCH = 32
GRID = 32
T_GROW = 64
T_COMM = 24
REPEATS = 3


def sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def timed(fn, device: str, warmup: int = 1, repeats: int = REPEATS) -> float:
    """Median wall-clock of `fn` in milliseconds."""
    for _ in range(warmup):
        fn()
    sync(device)
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        sync(device)
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return samples[len(samples) // 2]


def bench_growth_forward(device: str) -> float:
    org = NCAOrganism(layout=SENDER_LAYOUT, grid=GRID).to(device)
    torch.nn.init.normal_(org.fc2.weight, std=0.05)
    g = torch.Generator(device=device).manual_seed(0)
    x0 = org.seed_state(BATCH, device=device)

    def run():
        with torch.no_grad():
            org.rollout(x0.clone(), T_GROW, generator=g)

    return timed(run, device)


def bench_bptt(device: str, checkpoint_every: int | None) -> float:
    org = NCAOrganism(layout=SENDER_LAYOUT, grid=GRID).to(device)
    torch.nn.init.normal_(org.fc2.weight, std=0.05)
    g = torch.Generator(device=device).manual_seed(0)
    x0 = org.seed_state(BATCH, device=device)

    def run():
        org.zero_grad(set_to_none=True)
        out = org.rollout(
            x0.clone(), T_COMM, generator=g, checkpoint_every=checkpoint_every
        )
        out.square().mean().backward()

    return timed(run, device)


def bench_two_organism_comm(device: str, checkpoint_every: int | None) -> float:
    """The step that actually dominates training: sender + receiver, both with BPTT."""
    sender = NCAOrganism(layout=SENDER_LAYOUT, grid=GRID).to(device)
    receiver = NCAOrganism(layout=RECEIVER_LAYOUT, grid=GRID).to(device)
    for org in (sender, receiver):
        torch.nn.init.normal_(org.fc2.weight, std=0.05)
    gs = torch.Generator(device=device).manual_seed(0)
    gr = torch.Generator(device=device).manual_seed(1)
    xs = sender.seed_state(BATCH, device=device)
    xr = receiver.seed_state(BATCH, device=device)

    def run():
        sender.zero_grad(set_to_none=True)
        receiver.zero_grad(set_to_none=True)
        s = sender.rollout(xs.clone(), T_COMM, generator=gs, checkpoint_every=checkpoint_every)
        # Parameter-free pooled vote -> symbol logits -> straight-through-ish handoff.
        alive = sender.alive_mask(s)
        votes = (s[:, sender.layout.vote] * alive).sum((2, 3)) / alive.sum((2, 3)).clamp(min=1)
        sym = torch.softmax(votes, dim=-1)
        r = xr.clone()
        r[:, receiver.layout.sensor, GRID // 2, GRID // 2] = sym
        out = receiver.rollout(r, T_COMM, generator=gr, checkpoint_every=checkpoint_every)
        out.square().mean().backward()

    return timed(run, device)


def peak_memory_mb(device: str) -> float:
    if device == "mps":
        return torch.mps.current_allocated_memory() / 1e6
    return float("nan")


def main() -> None:
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")

    print(f"torch {torch.__version__} | batch={BATCH} grid={GRID}x{GRID} "
          f"channels={SENDER_LAYOUT.total} T_grow={T_GROW} T_comm={T_COMM}")
    print()

    rows = [
        ("growth fwd-only (64 steps)", lambda d: bench_growth_forward(d)),
        ("BPTT 24 steps, no ckpt", lambda d: bench_bptt(d, None)),
        ("BPTT 24 steps, ckpt=8", lambda d: bench_bptt(d, 8)),
        ("2-organism comm, no ckpt", lambda d: bench_two_organism_comm(d, None)),
        ("2-organism comm, ckpt=8", lambda d: bench_two_organism_comm(d, 8)),
    ]

    results: dict[str, dict[str, float]] = {}
    header = f"{'workload':<30}" + "".join(f"{d:>12}" for d in devices) + f"{'winner':>12}"
    print(header)
    print("-" * len(header))
    for name, fn in rows:
        results[name] = {}
        for d in devices:
            results[name][d] = fn(d)
        best = min(devices, key=lambda d: results[name][d])
        cells = "".join(f"{results[name][d]:>11.0f}m" for d in devices)
        print(f"{name:<30}{cells}{best:>12}")

    print()
    key = "2-organism comm, ckpt=8"
    if "mps" in devices:
        cpu_t, mps_t = results[key]["cpu"], results[key]["mps"]
        ratio = cpu_t / mps_t
        if ratio > 1.2:
            choice, why = "mps", f"MPS is {ratio:.2f}x faster on the decisive workload"
        elif ratio < 1 / 1.2:
            choice, why = "cpu", f"CPU is {1/ratio:.2f}x faster on the decisive workload"
        else:
            choice, why = "cpu", f"within 20% ({ratio:.2f}x) -- prefer CPU for determinism"
    else:
        choice, why = "cpu", "MPS unavailable"

    print(f"DECISION: device = {choice}  ({why})")
    print(f"Set `device: {choice}` in configs/base.yaml")


if __name__ == "__main__":
    main()
