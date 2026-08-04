# Milestone 2 results — grow, damage, regrow

Run `runs/milestone2`, 6000 steps, ~94 min on an M2, seed 0, no death resets.
Regimes: Growing (0–1000) → Persistent + Regenerating (1000–6000).

## Gates: both PASS

**G1 — growth and persistence** (128 rollouts)

| metric | value | threshold |
|---|---:|---|
| `iou_64_mean` | **0.9898** | ≥ 0.90 |
| `iou_64_p5` | 0.9826 | ≥ 0.85 |
| `rmse_64_mean` | **0.0109** | ≤ 0.05 |
| `iou_256_mean` | **0.9993** | ≥ 0.85 |
| `area_ratio_mean` | 0.9931 | ∈ [0.9, 1.1] |
| `area_ratio_in_band_frac` | 1.0000 | ≥ 0.90 |

Note `iou_256 > iou_64`: the organism is *better* at four times the training
horizon than at the horizon itself. The target is a genuine attractor, not a
memorised trajectory that decays once the training window ends. This is the
clause that a 30-step model failed with `area_ratio = 4.31`.

**G2 — regeneration** (256 trials, disk containing 30% of alive cells)

| metric | value | threshold |
|---|---:|---|
| `recovered_frac` | **1.0000** | ≥ 0.90 |
| `sustained20_frac` | 1.0000 | reported |
| `best_iou_mean` | 0.9974 | — |
| `tau_recover_median` | **13 steps** | — |
| `severity_mean` | 0.2982 | 0.30 ± 0.02 |
| `recovered_frac_centre_covered` | 1.0000 | — |
| `recovered_frac_centre_spared` | 1.0000 | — |

Every trial recovered, including under the stricter sustained-for-20-steps
criterion, with a median recovery time of 13 steps. Destroying the grid centre
made no difference, which is worth noting ahead of the sensor-targeted damage arm.

Artefacts: `media/growth.mp4`, `media/regen.mp4`.

## Propagation probe: the Week-3 finding

The probe asks whether a signal injected at the centre sensor patch can cross the
body — the assumption the whole communication phase rests on. Run on this
morphology-trained model, max geodesic radius 9.

**η² (fraction of per-cell state variance explained by the injected code):**

```
d=0  0.951   d=2  0.174   d=4  0.011   d=6  0.001   d=8  0.000
d=1  0.494   d=3  0.039   d=5  0.003   d=7  0.000   d=9  0.000
```

Verdict `ATTENUATING`, front stalled at d=2 from t≈30 onward.

But η² is a coarse effect size, and "explains little variance" is not the same as
"carries no information". A ridge-regression decoder over each distance ring,
trained on half the noise draws and tested on the other half (8 codes, chance
0.125), tells a different story:

| ring d | cells | t=24 | t=32 | t=48 |
|---:|---:|---:|---:|---:|
| 0 | 9 | 1.000 | 1.000 | 1.000 |
| 1 | 16 | 0.990 | 1.000 | 1.000 |
| 2 | 24 | 0.646 | 0.719 | 0.854 |
| 3 | 32 | 0.375 | 0.448 | 0.542 |
| 4 | 40 | 0.312 | 0.344 | 0.396 |
| 5 | 48 | 0.167 | 0.188 | 0.260 |
| 6 | 56 | 0.135 | 0.167 | 0.177 |
| 7 | 52 | 0.125 | 0.156 | 0.135 |
| 8 | 43 | 0.125 | 0.146 | 0.125 |
| 9 | 22 | 0.125 | 0.125 | 0.125 |

**Two things follow.** Decodable information reaches roughly **d ≈ 5**, not d ≈ 3
as η² implied — so the wall is soft, not hard. And accuracy *rises with time at
every ring* (t=24 → t=48), so the signal is still spreading slowly rather than
having reached a fixed point. Longer rollouts help; they are simply not enough on
their own to cover a radius-9 body.

**The caveat that makes this a lower bound.** This model was trained only to hold
a shape. Nothing in its objective ever asked it to relay the sensor channels, and
its loss never referenced them. That ~5-cell range is what the substrate does
*incidentally*. A communication-trained model is optimising for exactly this, so
the honest reading is: passive dynamics carry information about halfway across the
body, and the task must supply the rest.

### Consequences for Milestone 3

1. **Per-cell voting loss is now load-bearing, not just an anti-centralisation
   device.** Supervising every alive cell's vote forces the outer cells to be
   correct, which they can only manage by receiving the referent. That is the
   propagation pressure this model never had. It also protects the pooled readout
   from being diluted by uninformed outer cells.
2. **Size the body to what the substrate can carry.** Radius 10 (max geodesic 9)
   sits well outside the demonstrated range. Radius 6–7 gives max geodesic 6–7 and
   still leaves ~130–160 cells — ample for the distributedness and redundancy
   analyses, and closer to the measured envelope.
3. **Raise `T_comm` and `T_inject`.** Both help, cheaply: memory is nearly flat in
   rollout length (1110 MB at T=64 vs 1119 MB at T=128).
4. **Keep a hidden-channel carrier in reserve.** If the task alone does not open a
   pathway, injecting into hidden channels rather than dedicated sensor channels
   is the next lever.

Building the probe before the Lewis game is what surfaced this. Measured as
accuracy-versus-`T_comm` instead, it would have looked like "needs more steps",
and the fix would have been to raise `T_comm` — against a problem that time alone
does not solve.
