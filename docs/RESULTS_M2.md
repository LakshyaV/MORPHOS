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

---

# Milestone 3a — the broadcast task

Run `runs/broadcast`, 4000 steps, warm-started from the M2 morphology checkpoint,
body radius 7. Held-out evaluation, 128 episodes, salted RNG.

**The M2 probe's pessimism is overturned.** Per-cell supervision closes the gap it
identified.

| metric | value | threshold |
|---|---:|---|
| pooled vote accuracy | **0.9766** | ≥ 0.90 (chance 0.125) |
| cell agreement | **0.9495** | ≥ 0.80 |
| morphology IoU | **0.9179** | ≥ 0.85 |
| quorum fraction | **0.9189** | — |

**Per-cell accuracy by geodesic distance from the sensor:**

```
d=0  0.872    d=3  0.969    d=6  0.957
d=1  0.957    d=4  0.969    d=7  0.938
d=2  0.971    d=5  0.965    d=8  0.731
```

Against a chance floor of 0.125, cells eight rings out from the sensor vote
correctly 73% of the time; everything from d=1 to d=7 sits above 0.93. Compare the
morphology-only model, where linear decodability fell to chance by d≈6 and η² was
below threshold past d=3. The information now reaches the periphery.

**Quorum fraction 0.92 is the anti-centralisation result.** It is the smallest
fraction of cells whose removal flips the pooled decision: you would have to
delete 92% of the body to change its mind. A megaphone solution — the nine sensor
cells shouting while the rest of the body is decorative — would score ≈0.02. This
is the number that says the computation is genuinely distributed, and it is the
one to put in an abstract.

One oddity worth recording: d=0 (the sensor patch itself, 9 cells) scores 0.872,
*lower* than every ring from d=1 to d=7. The cells holding the raw injected code
are slightly worse at voting than their neighbours. Not yet explained; a small
sample, but consistent across episodes.

## The loss-balance finding

The first attempt reached vote accuracy 0.97 while morphology IoU collapsed from
0.52 to 0.25 — the organism dissolved its own body to solve the task. Measuring
per-term gradient norms on a warm model showed why: the vote-loss gradient was
**20× the morphology gradient**, so at `lambda_vote=1.0` communication was
outvoting morphology 20:1.

The loss *values* badly understate this — 0.036 versus 0.226 looks like a 6× gap.
Only the gradient norms reveal the true 20×. Setting `lambda_vote=0.03` (parity is
≈0.05) produced accuracy 0.98 *and* IoU 0.92 together.

The ratio is also not static: on a freshly warm-started model it starts near 0.5,
because the model has never seen an injected referent and its vote channels barely
move. A single constant is therefore a compromise across a moving target, and a
ramp may do better if the comm phase needs one.

---

# Milestone 3b — the Lewis game

## A failed first attempt, and what it measured

The first comm run sat at chance (accuracy 0.062–0.188 against a 0.125 floor) for
1050 steps. Direct gradient measurement found the cause:

```
task  gradient -> sender          1.02e-04
morph gradient x lambda_morph=30  2.59e+01
ratio                             255,000 : 1
```

The task gradient reaching the sender is tiny — it crosses a discrete channel, a
second organism, and ~64 steps of BPTT — and `lambda_morph=30` buried it entirely.
That value came from over-generalising the M3a lesson: there the *vote* gradient
dominated morphology 20×, so morphology was up-weighted. In the comm phase the
relationship inverts. Corrected to `3.0e-5` (parity ≈1.2e-4).

A second symptom pointed at a deeper problem: the sender's symbol logits varied by
only **0.012** across referents. The vote channels had never learned the referent
was present at all, because warm-starting from the *morphology* model gave a body
with no propagation — exactly the deficit the M2 probe measured and M3a fixed.

## The fix, and an honesty caveat that must reach the paper

Warm-starting from the **broadcast** checkpoint instead gives a sender whose cells
already carry the referent. Logit spread rises 0.012 → **2.02**, within-referent
consistency is **96.9%**, and all 8 symbols are used.

**But the resulting map is referent 0→symbol 0, 1→1, 2→2, …** The broadcast task
trained the vote channels to encode referent *identity*, so pretraining installed
an encoder rather than the sender inventing one. This must not be described as a
protocol that emerged from scratch.

What *does* still emerge is the **convention**. The receiver has never seen these
symbols and holds no prior about them; "symbol 3 means referent 3" is meaningless
until the receiver learns it, and the two must co-adapt for the pair to score above
chance. So the shared meaning emerges even though the encoder was pretrained.

The precise claim is therefore: *the convention is emergent, the encoder is
pretrained.* Anything stronger is unsupported.

To recover the stronger claim, a from-scratch arm is needed — warm-start the body
but not the vote channels, with `lambda_morph` correctly set. That is the
comparison that separates "can they invent a code" from "can they agree on one",
and it is worth running once the damage experiments are underway, since the
research question is about survival of a protocol rather than its invention.
