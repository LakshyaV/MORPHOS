# MORPHOS

**Can an emergent communication protocol remain meaningful when the distributed substrate producing and interpreting it is damaged and regenerated?**

Two Neural Cellular Automata grow from a single seed cell, invent a discrete
symbol protocol to solve a cooperative referential game, then get cut open. We
measure whether the body comes back, whether the protocol comes back, and
whether those are the same event.

Thesis under test: *regeneration restores the body and the channel capacity, but
not the convention.*

## Status

**Milestone 1** (substrate) complete. **Milestone 2** (grow → damage → regrow) code
complete, final training run in progress.

- [x] NCA substrate — deterministic, light-cone verified
- [x] Device benchmark and memory profile
- [x] Target morphologies + morphology metrics
- [x] Config system, training loop, JSONL logging, checkpointing
- [x] Sample pool, damage operators, gates G1/G2
- [x] Frame rendering and mp4 output
- [x] Propagation probe
- [x] Full 6000-step run — **G1 and G2 both PASS** + `media/regen.mp4`
- [ ] Lewis referential game (Milestone 3)

88 tests, `make fast` under 30 s on CPU.

**Milestone 2 result** (full numbers in [`docs/RESULTS_M2.md`](docs/RESULTS_M2.md)):

| | | |
|---|---:|---|
| G1 growth | IoU **0.9898** at t=64 | need ≥ 0.90 |
| G1 persistence | IoU **0.9993** at t=256 | need ≥ 0.85 |
| G2 regeneration | **100%** of 256 trials recovered | need ≥ 90% |
| G2 recovery time | median **13 steps** after losing 30% of cells | — |

The organism scores *higher* at four times the training horizon than at the
horizon itself, so the target shape is a genuine attractor rather than a
memorised trajectory.

The propagation probe returned the milestone's most consequential finding: a
signal injected at the sensor patch is linearly decodable out to only ~5 cells,
against a body radius of 9. That is a lower bound — this model was never asked to
relay anything — but it means Milestone 3 must actively build the pathway rather
than assume it. See `docs/RESULTS_M2.md` for what changes as a result.

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

## The science

The full design is in `docs/`, and it is the substance of this project — the
code is downstream of it.

- **[`docs/DESIGN.md`](docs/DESIGN.md)** — the research contribution and why the
  gap is real; substrate choice argued against alternatives; the four
  operational definitions of recovery; hypotheses H1–H4; how a centralized
  module is prevented from secretly solving the task; baselines and ablations;
  failure modes with a diagnostic for each; pass/fail gates; compute budget;
  staged roadmap.
- **[`docs/PROTOCOL.md`](docs/PROTOCOL.md)** — the experimental protocol:
  exact formulas for every metric and estimator, the damage factorial and which
  cells are actually run, confound analysis with a control for each alternative
  explanation, the intervention suite ranked by necessity, the statistical plan,
  and the numeric pass/fail gates.

### The question

Damage the sender, keep the receiver frozen and healthy, and the receiver
becomes a fixed semantic reference frame: any drop in task accuracy is the
sender's protocol drifting away from the meaning its partner was trained to
read. Then plot morphological recovery and semantic recovery on the same axis.

The claim under test is that those two curves come apart — formally, that the
event `MR ≥ 0.95 ∧ struct ≥ 0.90 ∧ align < 0.60` occurs: body perfect, still
speaking a language, but no longer mutually intelligible with its unchanged
partner.

The naive version of this claim ("RGBA recovered but the hidden channels
didn't") is a linear-algebra triviality, since RGBA is 4 of 28 channels and the
rest were never constrained to recover. Requiring `struct ≥ 0.90` — that the
protocol still carries its information — is what rules out "damage just made it
stupid" and makes the result substantive.

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
