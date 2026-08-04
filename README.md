# MORPHOS

**Can an emergent communication protocol remain meaningful when the distributed substrate producing and interpreting it is damaged and regenerated?**

Two Neural Cellular Automata grow from a single seed cell, invent a discrete
symbol protocol to solve a cooperative referential game, then get cut open. We
measure whether the body comes back, whether the protocol comes back, and
whether those are the same event.

Thesis under test: *regeneration restores the body and the channel capacity, but
not the convention.*

## Status

Day 1–2 complete: substrate implemented and verified, environment benchmarked.

- [x] NCA substrate (`morphos/substrate/nca.py`) — 10 unit tests, deterministic
- [x] Device benchmark and memory profile
- [ ] Morphology training + growth/persistence gate G1
- [ ] Damage/regeneration + gate G2
- [ ] Lewis referential game + gate G3

## Setup

```bash
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
make fast      # unit suite, <1s on CPU
make bench     # CPU vs MPS decision table
```

Requires a **native arm64** Python. The conda environments on this machine are
x86_64 under Rosetta and should not be used.

## Measured facts

Benchmarked on Apple M2 (8 GB), torch 2.9.0, batch 32, 32×32 grid, 28 channels.
Both columns are native arm64.

| Workload | CPU | MPS | Winner |
|---|---:|---:|---|
| growth fwd-only (64 steps) | 1696 ms | **187 ms** | mps |
| BPTT 24 steps, no ckpt | 1827 ms | **180 ms** | mps |
| BPTT 24 steps, ckpt=8 | 2678 ms | **246 ms** | mps |
| 2-organism comm, no ckpt | 3727 ms | **358 ms** | mps |
| 2-organism comm, ckpt=8 | 5487 ms | **489 ms** | mps |

**Decision: `device: mps`** — 11.2× faster on the decisive workload. The
kernel-launch-overhead concern was real but does not flip the decision.

### Memory (the actually binding constraint — 8 GB machine)

| Config | Peak MB |
|---|---:|
| T=64, batch 32, no checkpointing | 3300 |
| T=64, batch 32, `checkpoint_every=8` | **1110** |
| T=128, batch 32, `checkpoint_every=8` | **1119** |

Gradient checkpointing is **mandatory**, not an optimisation. Note the second
finding: with checkpointing, memory is nearly independent of rollout length.
Long rollouts are therefore affordable, which matters because information
propagates only ~0.5 cells/step under stochastic updates — a 24-step rollout
would reach ~12 cells and leave most of the body blind to the referent.

## Design notes

Two things in `nca.py` are easy to get wrong and fail *silently*:

1. **Perception kernel grouping.** `F.conv2d(..., groups=C)` assigns output
   channels to groups in contiguous blocks of 3, so the layout is
   `[id_0, sx_0, sy_0, id_1, sx_1, sy_1, ...]`, not
   `[id_0..id_{C-1}, sx_0..., sy_0...]`. Wrong ordering still trains, just worse.
   `test_sobel_on_ramp` catches it.
2. **Fire masks are pre-generated for the whole rollout**, not drawn inside
   `step`. Gradient checkpointing recomputes the forward pass, which would
   otherwise advance the generator and desynchronise the recomputed masks.
   `test_checkpointed_rollout_matches_plain_rollout` catches it.

Determinism is asserted **same-device only**. CPU and MPS produce different
streams from the same seed, so device is part of run identity. MPS also has no
float64, so metric accumulation must go through `.cpu().double()`.

## Layout

```
morphos/substrate/nca.py   the substrate: perception, alive masking, rollout
morphos/seeding.py         four separate RNG streams (update/task/damage/pool)
configs/base.yaml          defaults every experiment inherits
scripts/bench_device.py    reproduces the table above
tests/                     fast suite, must stay under 30s on CPU
```

The full scientific design — hypotheses, damage taxonomy, metric definitions,
confound controls, and pass/fail gates — lives in the plan document.
