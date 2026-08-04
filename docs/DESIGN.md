# MORPHOS — Emergent Communication in Regenerating Substrates

## Context

`/Users/laky.vasu/Desktop/ai_env/MORPHOS` is an empty git repo (no commits). This is a greenfield research project, so the plan is a scientific design first and a code plan second.

**The gap is real.** A literature sweep (NCA + emergent communication, ~15 targeted novelty queries across arXiv, ALIFE/MIT Press, ICLR/NeurIPS workshops) found no work combining (i) NCA-grown, damage-regenerating agents with (ii) a *learned, discrete, emergent* protocol, tested for (iii) semantic survival under substrate damage. Nearest misses, and precisely what each leaves open:

| Work | What it does | What it leaves open |
|---|---|---|
| Singh 2026, arXiv:2606.21202 | NCA sub-populations with different "languages" | Language is an **imposed permutation matrix**, not learned; no damage |
| Stovold 2025, arXiv:2508.06389 | Identity channels so NCA organisms coexist | No signaling, no task, no protocol |
| Sakana PD-NCA (arXiv:2604.11248) | Many NCA organisms on a shared substrate | Interaction via attack/defense channels, no discrete vocabulary |
| Variengien 2021, arXiv:2106.15240 | NCA cart-pole controller survives damage | **Feasibility precedent** — but single-agent, no communication |
| ESCELL, arXiv:2007.09469 | Real Lewis game, "cellular" in the title | "Cellular" = microscopy images. Standard CNN agents. False positive |

Both literatures are mature and well-instrumented separately. Their intersection is empty.

**Decisions taken:** target the Aug 29 NeurIPS-workshop shape as a forcing function, decide at the Week-3 gate whether to submit there or fall back to *Artificial Life* Letters (rolling, best community fit). Both agents are NCAs. Disk cleanup = caches only.

---

## Verdict on the vision: what changes and why

Preserving the objective — emergent communication inside developmental systems that can be damaged and regenerated — requires cutting several proposed mechanisms. The vision as written is a 6–12 month program with internal contradictions at 4 weeks.

**Cut, with reasons:**

1. **All reinforcement learning, movement, energy, hazards, and the gridworld.** Replaced by a fully differentiable Lewis referential game. This is the single biggest de-risking decision. RL + NCA + emergent communication from scratch, solo, on a laptop, in 4 weeks is a near-certain failure. The referential game preserves the research question completely (distributed substrate producing and interpreting invented symbols, damaged mid-operation) while removing the dominant source of training variance. Movement returns as a Stage-3 extension, not a Stage-1 requirement.
2. **Energy / age / death channels.** Non-differentiable, destabilizing, no payoff for the core question.
3. **The web frontend.** Week-4 matplotlib multi-panel animation via the `ffmpeg` already at `/usr/local/bin/ffmpeg` delivers the same scientific content as the "language microscope" for ~60 lines instead of a React app.
4. **The 4×4 compositional extension (16 referents, message length 2).** No hypothesis attached in v1. Deferred.
5. **D3 global channel ablation and the CNN baseline.** The former hits every cell identically so there is no spatial redundancy to exploit; the latter cannot be damaged comparably.

**Changed:**

6. **Never run N > V^L.** Canonical: N=8 factored referents (`{0,1}³`), V=8, L=1. A paper reporting 50% accuracy because 16 referents were forced through an 8-symbol channel is a self-inflicted wound.
7. **28 channels, not 16.** 4 RGBA + 3 sensor + 8 vote + 13 hidden. At 16 channels, taking 8 for votes leaves 4 for morphogenesis; Mordvintsev needed all 12 for morphology alone. Capacity starvation would fail silently and cost a week to diagnose.
8. **T_S = 64, not 24.** With stochastic update p=0.5, information advances ~0.5 cells/step, so T_c=24 reaches ~12 cells — a large fraction of the body would never see the referent. This is the most likely cause of a Week-1 failure. Sensor sits at the body centroid to halve max geodesic distance.
9. **Loss is L2-to-one-hot, not cross-entropy.** Self-classifying MNIST found CE on per-cell logit channels causes unbounded logit growth, so the residual update never decays and cells never quiesce.

**The first-milestone question, answered directly.** Your proposed first milestone (organism develops, stays stable, regenerates, with eval + visualization) is correct as *engineering* sequencing but wrong as *risk* sequencing. Growth-and-regeneration is a replication of a known 2020 result — low risk. The novel, unproven part is whether an NCA can emit a discrete symbol at all through a parameter-free consensus readout. So: build growth/regeneration in days 2–4 (known recipe), then immediately attack the riskiest unknown on days 5–7. Critically, **prototype the mechanical-vs-dynamical decomposition in Week 1** — if mechanical pooling shift explains the whole effect, there is no paper, and that must surface on day 7, not day 25.

---

## 1. The contribution and the smallest meaningful experiment

**Contribution.** The first study of emergent communication where the communicating agents are self-organizing, damage-regenerating substrates; and a measurement protocol for *semantic* recovery, which the NCA literature has never defined (it measures pixel distance-to-target or downstream task score) and the emergent-communication literature has never needed (its agents cannot be structurally damaged).

**Thesis sentence.** *Regeneration restores the body and the channel capacity, but not the convention.*

**Smallest meaningful experiment.** One NCA sender, one NCA receiver, N=V=8, damage the sender only with the receiver frozen and healthy, one damage condition (disk, f=0.30, at maturity) plus sham, full recovery time course to Δ=256, with the causal-reality battery. 4–6 seeds. That is a workshop paper if the controls are airtight — the controls are what make it a result rather than an anecdote.

---

## 2. Substrate and why NCA wins

**Representation: 2D grid NCA**, 32×32, 28 channels, fixed Sobel-x/Sobel-y/identity perception (84-dim), update MLP 84→128(ReLU)→28 with zero-initialized final layer, residual update, stochastic update p=0.5, alive-masking via 3×3 maxpool on α>0.1. Both agents; separate parameters, identical architecture, no shared state.

**Why, against the alternatives:**

| Alternative | Why not, for *this* question |
|---|---|
| **Graph / Mesh NCA** (arXiv:2311.02820) | Strictly more general, but damage geometry becomes topology-dependent and "kill a disk" loses its clean severity parameterization. No gain for the question. |
| **Developmental / HyperNCA** (arXiv:2204.11674) | Grows *weights*, not a body. Damage would hit a weight tensor, which is exactly the non-distributed thing we're contrasting against. |
| **Modular / Deep Sets** | Distributed and weight-shared but *no locality*. Better as a **baseline (B4a)** to isolate whether locality does the work — not as the substrate. |
| **Conv-GRU recurrent CNN** | Iterative and local but not cell-autonomous. Also a **baseline (B4c)**, not the substrate. |

NCA is chosen because it uniquely supplies all four requirements at once — shared local update rule, developmental growth, self-maintenance, and damage recovery — with a published, reproducible recipe and known hyperparameters. The alternatives each drop one.

---

## 3. Precise definitions

Notation: `t=0` is damage. `0⁻`/`0⁺` = immediately before/after. Every damaged episode has a **matched healthy twin** (identical seed, referent, RNG stream, no damage). All recovery is measured against the twin *and* the pre-damage state.

**Development.** Growth from a single seeded cell over T_grow=64 steps to a fixed target morphology, identical for all referents. *The target body must be referent-independent* — otherwise morphological and semantic recovery are the same variable and the hypotheses are vacuous.

**Stability.** `IoU(T_grow) ≥ 0.90` and at `t=4·T_grow=256`: `IoU ≥ 0.85`, `0.9 ≤ |A₂₅₆|/|A₆₄| ≤ 1.1`. The second clause catches the classic "explodes past the training horizon" failure.

**Damage.** Severity `f` is always defined as **fraction of currently-alive cells**, solved by per-episode binary search on the disk radius to hit target f within ±2%. Defining severity by radius confounds it with body size and regeneration state.

**Four recoveries, deliberately distinct:**

```
Morphological   MR_t = (IoU_t − IoU_0⁺)/(IoU_0⁻ − IoU_0⁺)     recovered iff ≥0.95·IoU_0⁻, sustained 20 steps
Behavioural     BR_t = (Acc_t − 1/N)/(Acc_0⁻ − 1/N)            chance-corrected; non-inferiority test at δ=0.05
Computational   CR_t: centered-logit recovery ≥0.9 AND RSA ≥0.90 AND FieldCorr ≥0.8
Semantic        SIA_t = (1/N)Σ_r 1[σ̂_t(r) = σ̂_0⁻(r)]         macro-averaged over referents
```

Semantic recovery splits in two, and the split is the paper:

```
struct_t  = I_t(M;R)/I_0⁻(M;R)      "does it still speak a language?"    permutation-INVARIANT
align_t   = SIA_t                    "does it speak the SAME language?"   permutation-SENSITIVE
relabel_t = SIA^perm_t − SIA_t       how much loss is pure relabeling
```

`SIA^perm` uses the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) on the V×V pre/post confusion matrix — an *exact* maximum over all 8!=40,320 permutations, not a greedy match. Say so in the paper.

**Hypotheses (H4 is secondary; do not let it hold up the narrative):**

- **H1** — semantic recovery lags morphological recovery. Pre-registered minimum effect: `median(τ_sem − τ_morph) ≥ 16 steps`.
- **H2** — damage-trained organisms drift less *at matched morphology*.
- **H3** — the dissociation, defined as the event `MR_t ≥ 0.95 ∧ struct_t ≥ 0.90 ∧ align_t < 0.60`. Body perfect, still linguistic, no longer mutually intelligible with its unchanged partner.
- **H4** *(secondary)* — communication-relevant computation is more spatially redundant.

**H3's formulation is load-bearing.** The naive version ("RGBA recovered but hidden channels didn't") is a linear-algebra triviality — RGBA is 4 of 28 channels and the rest were never constrained to recover; a reviewer will write exactly that sentence. Requiring `struct ≥ 0.90` rules out "damage just made it stupid" and makes the claim substantive. Also report `M_hidden(t)` alongside `M_RGBA(t)`; without it the result is indefensible.

---

## 4. Environment, and why communication is necessary

A Lewis referential game. Sender observes referent `r ∈ {0,1}³` (N=8, factored so attribute-wise MI is well-defined); receiver observes nothing but the symbol and must identify `r`. Communication is necessary **by construction**: the receiver has no other input path, verified by gate G4 (message removed ⇒ accuracy ≤ 0.175 against chance 0.125).

**Referent injection window T_inj=16, then removed.** This forces the sender to *store* the referent in cell state, so damage can destroy memory. Continuous injection means the sender simply re-reads the referent after damage — a much weaker phenomenon. Run continuous-injection as a control and report both; reviewers will ask which you did.

---

## 5. Distributed sensing, computation, action, communication

1. Referent injected at a 3×3 sensor patch at the **body centroid** for `t ∈ [0,16)`.
2. Information propagates only through local 3×3 updates — provably ≤1 cell/step, ~0.5 with stochastic updates.
3. Sender symbol logits = **hard-mask uniform mean** over alive cells of the 8 vote channels:
   ```
   ℓ_v = (1/|A_t|) Σ_{x∈A_t} z_v(x)        A_t(x) = 1[max_{3×3} α(x) > 0.1]
   ```
4. Straight-through Gumbel-Softmax → discrete symbol → injected as hard one-hot at the receiver's ear patch.
5. Receiver answer logits, same parameter-free pooling over its answer channels.

**Do not use alpha-weighted pooling** (`Σα·z/Σα`). It makes morphological state a multiplicative modulator of the message, so the message changes during regeneration even when every cell's computation is identical. That one implementation detail would confound the entire paper.

---

## 6. Preventing a centralized module from solving the task

This is where the red-team analysis changed the design. `softmax(mean over cells)` is *parameter-free* but not *decentralized* — a global sum has a global receptive field and instantaneous propagation. Worse, gradient descent actively prefers the centralized solution: getting information to distant cells costs 64 steps of attenuating BPTT, while having the 9 sensor cells shout costs 2. **The shortest gradient path is the megaphone.**

Four mechanisms, in order of force:

1. **Per-cell loss (true Randazzo).** Penalize *every* alive cell's vote against the one-hot target, not just the pooled result. Consensus becomes an emergent *measured* property rather than an imposed aggregation. This is the correct reading of Self-classifying MNIST, which our design was previously misciting.
2. **Proposition 1 (deletion-invariance under consensus).** *If `argmax_v z_v(x) = v*` for every `x ∈ A`, then for every nonempty `S ⊆ A`, `argmax_v (1/|S|)Σ_{x∈S} z_v(x) = v*`.* Proof: full consensus means `z_{v*}(x) > z_u(x)` pointwise; averaging preserves strict inequality termwise. ∎
   Combined with (1), this is the key result: **training for per-cell consensus makes deletion provably unable to flip the symbol**, so any observed symbol change must reflect genuine changed computation. This converts the central confound from an argument into a theorem.
3. **Readout cell-dropout during training** — pool over a random 40–70% subset of alive cells each forward pass. The only mechanism that supplies gradient pressure toward electorate-invariance.
4. **Probes that must pass:** zero-step probe (0 NCA steps ⇒ chance); propagation-horizon probe (accuracy vs. steps must ramp consistently with the geodesic distance transform on the alive mask); sensor-exclusion probe (exclude sensor cells from the pool ⇒ accuracy survives).

**Distributedness diagnostics** (all closed-form, zero extra forward passes — leave-one-out is exact for mean pooling: `φ(x) = (z_{v*}(x) − ℓ_{v*})/(|A|−1)`): participation ratio, Gini of `|φ|`, adversarial deletion tolerance `k*`, and the **random-subset readout curve** `SP(f)` — flat down to small f means distributed, a cliff means megaphone.

---

## 7. Bottleneck and leakage

- Hard one-hot forward at eval; soft warmup with temperature annealing early in training, then hard straight-through. Report the soft-vs-hard gap. Purity at initialization buys nothing and destroys the initial SNR.
- **Gate G4 channel purity:** assert the set of distinct ear-injection tensors has cardinality exactly V. Violation ⇒ analog side channel ⇒ every discrete metric in the paper is invalid.
- **Data-processing sanity check:** the pipeline is a Markov chain R→M→Â, so `I(Â;R) ≤ I(M;R) + 0.1 bit` must hold. Violation ⇒ referent reaching the receiver by some other path ⇒ stop and debug. Automated.
- Capacity is matched by construction: N=8 needs 3 bits, V=8 supplies exactly 3.

---

## 8. Training

**Fully differentiable end-to-end. No RL.** Adam, per-parameter-tensor L2 gradient normalization (Mordvintsev), gradient checkpointing every 8 steps (mandatory — see compute).

Sample-pool training: pool of 1024 post-growth states amortizes T_grow; batch 32; reseed the worst-loss sample to a single seed cell; damage the lowest-loss samples.

**Five training details that are load-bearing:**

1. **Train for message persistence.** Sample readout time `T_S ~ U[64,128]` and apply the symbol loss at *multiple* readout times. Otherwise the referent-dependent hidden state has no attractor pull, decays, and you measure message half-life while calling it damage. **Report the healthy organism's message half-life as a headline number** — it is the denominator for everything else.
2. **Damage *during* communication, not just before.** If damage is only applied at pool-sample time, the model only ever gets gradient for morphological regeneration and never for "recover the message after being damaged." H2 would then test something never trained for.
3. **Clear the vote channels at the start of each communication phase.** Pooled states carry hidden residue from the previous referent, making the map `(referent, history) → symbol` rather than `referent → symbol`. Verify round-to-round fidelity of an undamaged organism is ≈1.0 before trusting any measurement.
4. **Mask the task loss when `alive_count < 60%` of target.** Otherwise freshly reseeded 5-cell blobs receive the communication loss — unlearnable, pure gradient noise.
5. **Continuous loss ramps, not hard curriculum stages.** Stage boundaries poison the pool with states generated under old dynamics.

**Watch gradient norms per loss term, not loss magnitudes.** Morphology L2 over 1024 cells × 4 channels versus an 8-dim answer L2 differ by ~3 orders of magnitude; the default outcome is beautiful morphology stuck at 12.5% accuracy for a week.

---

## 9. Baselines, ablations, interventions

**Mandatory interventions** (all are "change one line in the eval loop"; omitting 1–3 makes the paper unpublishable under the Lowe et al. standard):

| # | Intervention | Falsifies |
|---|---|---|
| 1 | Remove message | Acc unchanged ⇒ no communication or a leak |
| 2 | Random symbol from marginal σ̄ | Acc unchanged ⇒ protocol not causally real |
| 3 | Permute vocabulary | Acc holds ⇒ receiver reads presence, not identity |
| 4 | Cross-play across seeds | High cross-play ⇒ no convention emerged. Supplies the "totally incompatible" reference scale that makes post-damage drift interpretable |
| 5 | Region ablation at readout | Confound control C1 |

Deferred: activation patching (the workshop→strong-paper upgrade), channel ablation (basis-dependent; needs a random-rotation control), symbol noise, symbol removal, message delay. "Replay message from another episode" is **vacuous here** — with hard one-hot symbols, replaying the same index is literally the identical tensor; replaced by the G4 purity assertion.

**Baselines:** B0 sham (f=0) — mandatory, without it every drift number is uninterpretable; B1 untrained NCA (floor, free); **B2 the T1/T2/T3 training conditions**; B3 receiver re-adaptation (`k_relearn` = gradient steps for the frozen receiver to regain 0.95·Acc — directly measures the cost of not being able to renegotiate); B4d learned-readout NCA (defuses "your whole result is a property of mean pooling").

**T3 is the crucial control:** damage in the training pool but with the task loss *detached* during damaged steps. It learns to regrow but never learns to communicate while damaged. T2 vs T3 is what makes H2 a mechanistic claim rather than a correlation.

**Do not promise "NCA vs MLP."** An MLP with zeroed units stays zeroed forever — comparing a system with a recovery mechanism against one with none is not an interesting result and a good reviewer will say so. The defensible comparisons are all *within*-NCA and isolate what makes an NCA an NCA: **B4a non-local** (adds a global pooled context vector — isolates locality), **B4c Conv-GRU** (isolates recurrence vs. cell-autonomy), **B4d learned readout**. If a cross-substrate comparison is reported at all, match on **immediate post-damage accuracy** (titrate each substrate until both drop equally), then compare only recovery trajectories.

---

## 10. Metrics

Enumerate rather than sample: all N referents, all V counterfactual symbols in CIC (exact, an improvement on Lowe's sampled estimator). Damage placement is the **bootstrap unit** — SIA per placement is a binomial with n=8, so bootstrap over the J=32 placements, never over referents.

- **Morphology:** IoU, RMSE, `|A_t|`, MR
- **Behaviour:** Acc (Wilson CI, never Wald — you will be near p=1), BR, per-referent accuracy vector
- **Semantics:** SIA, SoftSIA, TV-drift, Cohen-κ (a sender collapsing to one symbol gets free credit otherwise), `SIA^perm`, `struct`, `Acc^perm`, `RenegCost`
- **Information:** H(M) with Miller–Madow correction, V_eff = 2^H(M), **permutation-debiased I(M;R)** (plug-in MI is upward-biased by ~0.069 bits at N=V=8, n=512 — not negligible), attribute-wise MI
- **Causal:** CIC by exact enumeration over V, using the receiver's softmax not argmax
- **Consensus:** plurality fraction, agreement with emitted symbol, full-consensus rate (Randazzo's comparison point: 88.1%)
- **Distributedness:** PR, Gini, `k*`, `SP(f)`, `f_50`

**Topsim is nearly meaningless at L=1** — edit distance is binary and it degenerates to a point-biserial correlation. Report attribute-wise MI instead, and say why in the paper rather than printing a number that looks like Chaabouni's but isn't.

---

## 11. Confounds and their controls

The central claim — *damage the sender, freeze the receiver, so accuracy drop = semantic drift* — rules out receiver drift and nothing else. Each of these needs its own control:

| Confound | Control |
|---|---|
| **Mechanical pooling shift** — deleting 30% of cells changes the mean even if every survivor is unchanged | **Frozen-state deletion decomposition** (exact, zero forward passes): apply the same mask to the healthy pre-damage state and recompute the pooled readout with *no dynamics*. Splits total drift into mechanical vs. dynamical components. Becomes Fig 6. Plus Proposition 1 where consensus is high. **This is the single most important missing control — prototype it in Week 1.** |
| **Message half-life** — the sender was trained to be right at T_S, not T_S+40 | Sham (f=0) and the matched healthy twin. Report `S_damaged(t) − S_healthy(t)`, never `S_damaged(t)` alone |
| **Confidence, not identity** — flatter softmax rather than a changed argmax | Report `Acc^argmax` and `Acc^sampled` at every Δ; track logit margin and H(M\|R); temperature sweep |
| **Sensor destroyed** — the referent literally cannot enter | Stratify by sensor overlap; headline result on the non-overlapping stratum; run sensor-targeted damage (D5) as the explicit contrast |
| **Mid-regeneration** ("not done computing") | Run to T_max=256 ≫ T_grow=64; report **semantics binned by IoU**, not by wall-clock, which definitionally excludes "not done yet" |
| **Referent heterogeneity** | Never report only the mean; referent×Δ heatmap always in supplement; pre-damage difficulty as covariate |
| **T1/T2/T3 are just different-quality models** | ANCOVA on pre-damage accuracy; checkpoint matching to ±2 points; plus T3 |
| **Receiver not actually frozen** | Hash receiver parameters before/after; assert identical initial state and RNG stream |

**H2 must match on the treatment, not the outcome.** "At equal morphological recovery" conditions on a post-treatment variable — if naive organisms only reach M=0.9 under mild damage while damage-trained reach it under severe damage, "at equal M" silently compares mild vs. severe. That is a collider. Report S and M as parallel functions of severity, or use a pre-specified matched-pairs design with published severity distributions.

---

## 12. Failure modes and diagnostics

| Failure | Diagnostic | Fix |
|---|---|---|
| NCA doesn't grow / explodes | alive-count trace vs. light-cone bound `(2k+1)²` | Per-tensor grad norm; pool training; check kernel grouping |
| Symbol collapse | `H(M)`, `V_eff` — logged every N steps. Collapse gives 12.5%, **indistinguishable from "not learning yet"** without this | Entropy bonus (0.01–0.1, annealed off); auto-restart if `H(M) < log 2` after k steps |
| Megaphone readout | `SP(f)` cliff; `k*/|A|` ≈ 0.02; sensor-exclusion probe fails | Per-cell loss; per-cell softmax before pooling; readout dropout |
| Vanishing sender gradient | pooled-logit norm growing; saturated softmax | Normalize pooled scale (LayerNorm or L2) — also fixes Gumbel gradient death |
| Task loss invisible | per-term **gradient norms**, not losses | Normalize both to per-element means; tune λ |
| Signal never crosses the body | accuracy flat vs. T_S | T_S=64+, sensor at centroid, or shrink grid to 24×24 |
| Protocol is state-dependent | round-to-round fidelity of an undamaged organism < 1.0 | Clear vote channels between phases; evaluate from a canonical state |
| OOM at hour 3 | activation memory ≈1.2 GB at T=24, 2.4 GB at T=48, vs ~2.7 GB free | Gradient checkpointing from day 3, not after the first OOM |

---

## 13. Gates (every one is a number a script checks)

- **G1 growth/persistence** — `IoU(64) ≥ 0.90`, `RMSE ≤ 0.05`; at t=256 `IoU ≥ 0.85` and `0.9 ≤ |A₂₅₆|/|A₆₄| ≤ 1.1`
- **G2 regeneration** — after f=0.30 at maturity, `IoU ≥ 0.90` within 128 steps in ≥90% of 256 trials
- **G3 protocol emerged** — `Acc ≥ 0.90` (Wilson LB ≥ 0.85, n=512), `I(M;R) ≥ 2.4 bits` debiased, `V_eff ≥ 6`. Weak tier: `Acc ≥ 0.50`, `I(M;R) ≥ 1.0 bit`. **Report emergence rate across seeds — this is itself a result that emergent-communication papers routinely hide**
- **G4 causally real** — message-removed ≤0.175; random-symbol ≤0.175; permuted-vocab ≤0.225; `CIC ≥ 1.0 bit`; cross-play ≤0.225; DPI holds; channel purity = exactly V
- **G5 damage produced signal** — immediate drop ≥0.20; sham `|ΔAcc| ≤ 0.03` and sham `SIA ≥ 0.97`; asymptotic drift > sham (paired Wilcoxon, p<0.05, `ΔSIA ≥ 0.10`); `MR(128) ≥ 0.95` in ≥80% of trials

**Fallback ladder if G0 (compute) fails, pre-registered in this order:** N=8→4 (V=4, chance 0.25); grid 32→24; C=28→20; T_S=64→48.

---

## 14. Compute — measured, not estimated

Benchmarked on this machine (torch 2.2.2, batch 32, 32×32 NCA). **MPS wins; the kernel-launch worry was wrong:**

| Workload | CPU (Rosetta) | MPS | Speedup |
|---|---|---|---|
| 64-step growth, forward | 1096 ms | **104 ms** | 10.5× |
| 24-step BPTT fwd+bwd | 1338 ms | **277 ms** | 4.8× |
| 2-organism comm step | 2463 ms | **684 ms** | 3.6× |

MPS hits only 3.3% of the M2's ~3.6 TFLOP/s peak, so the workload *is* launch-bound — it's just still faster on GPU. Consequence: **batch size is nearly free up to ~128** (2.83 ms/sample at B=32 vs 2.11 at B=512). Stay at B=32 on 8 GB; go to B=128 on the M5 Pro.

**Memory is the binding constraint, not disk.** Activations ≈860 KB/sample/step → 1.76 GB for 64-step growth BPTT at B=32, against ~2.7 GB free. Gradient checkpointing every 8 steps: ~440 MB for ~1.33× compute. Non-negotiable.

**Budget:** morphology pretrain ~1.9 h + comm training ~3.8 h = **~5.7 h per run**. Full matrix at 5 seeds × 3 conditions ≈ 85 h ≈ 7 calendar days at a realistic 12 h/day unattended. That fits the 26 days but leaves only one full re-run of slack, and the probability that the first full matrix has a bug is well above 50%.

**So: cut to ~45–50 h.** In order: (1) 5 seeds for the headline condition, 3 for ablation arms; (2) **calibrate comm training length on day 6–7 rather than assuming 20k steps** — if it plateaus at 12k that is a 40% cut for free, and it is the single highest-leverage measurement in the project; (3) 24×24 grid for non-headline conditions. The damage sweep is eval-only and costs ~15 minutes total.

---

## 15. Roadmap

**Stage 1 — proof of concept (Week 1).** Days 1: environment. Day 2: NCA substrate + tests. Day 3: morphology training (gate: L2 < 0.02 by 8k steps — if this fails the NCA is wrong, stop and fix, nothing downstream works). Day 4: damage/regeneration. Day 5: the Lewis game wired end-to-end. Days 6–7: buffer plus the calibration run.
*Week-1 gate: accuracy > 0.5 on 8 referents (chance 0.125), and a working mechanical-vs-dynamical decomposition.*
If the gate fails, diagnose in this order: τ too low too early; ear injection too weak; **T_S too short** (most likely).

**Stage 2 — research-grade (Weeks 2–3).** Week 2: full metric suite with hand-verified tests, tiered episode recorder, mandatory interventions, J-pilot, launch training. Week 3: core block (18 conditions: f ∈ {0,.1,.2,.3,.45,.6} × {T1,T2,T3}, fully crossed because f×training-condition *is* H2) plus sham; recovery time courses at `Δ ∈ {0,1,2,4,8,...,256}`; Figures 2–6.
*Week-3 gate (the submit/defer decision): G3, G4, G5 all pass on ≥3 seeds.*
If H1's CI straddles zero, **pivot the framing by Wed Aug 19, not Aug 28**, to the `SIA^perm` result — "morphology and protocol recover at indistinguishable rates, but the recovered protocol is a *permutation* of the original," which is arguably the more interesting finding.

**Stage 3 — advanced (deferred).** Activation patching; sweeps over damage type/agent/timing; receiver-damage arm; the compositional 4×4 extension with posdis/bosdis; the interactive language microscope; movement and RL.

**Week 4.** Final figures; **code freeze Wed Aug 26**; write. Reserve the last 4 days entirely for writing.

**The four-figure minimum paper**, if everything else fails: (A) setup schematic + regeneration filmstrip with the emitted symbol printed under each frame — *the body comes back, the word changes*; (B) recovery time course, morphology vs. semantics, sham overlaid; (C) intervention bar chart with CIC; (D) mechanical-vs-genuine decomposition. If no protocol emerges at all, the paper becomes a characterized negative result plus the measurement protocol itself — a defined floor, which a 4-week project needs.

---

## 16. Code architecture

~28 files. Plain YAML + frozen dataclasses (**not Hydra** — it changes the working directory, breaks relative paths, and its value is composition across many groups; there are four axes here). JSONL logging (**not TensorBoard/wandb** — ~60 KB/run, greppable, crash-safe, no server on an 8 GB machine).

```
MORPHOS/
├── configs/            base, morph_only, comm_nca, comm_learned_readout, sweep_damage
├── morphos/
│   ├── config.py       dataclasses + _base_ inheritance + --set overrides
│   ├── registry.py     the 4 swappable axes only
│   ├── seeding.py      4 separate RNG streams (update/task/damage/pool)
│   ├── substrate/      organism.py (Protocol), nca.py, mlp.py, pool.py
│   ├── task/           lewis.py, channel.py, readout.py, targets.py
│   ├── damage/ops.py   DamageOp Protocol + D1/D2/D4/D5, binary-search radius
│   ├── train/          loop.py (ONE phase-parameterized loop), losses.py, checkpoint.py
│   ├── eval/           metrics.py (@metric registry), recovery.py, controls.py
│   ├── io/             runlog.py, episode.py (tiered recorder)
│   └── viz/            frames.py, video.py (ffmpeg rawvideo pipe), figures.py
├── scripts/            bench_device, train, evaluate, sweep_damage, make_video
└── tests/
```

Four axes get registries because the ablation table depends on them: **organism type, readout type, damage type, baseline agent type**. Nothing else does.

**Explicitly do not build:** web frontend, trainer class hierarchy, Lightning/Hydra/wandb/Docker/CI, plugin discovery, dataset/dataloader abstraction (referents are `torch.randint`), abstract Environment interface, results database, `torch.compile` (it will spend longer compiling a 64-step Python loop than it ever saves).

**Two interface details that matter:**
- `MLPOrganism` uses `state_shape=(64,1,1)` so every downstream tensor op keeps the same rank — that single choice removes essentially all special-casing the baseline would otherwise force.
- `DamageOp.expected_severity(alive) -> Tensor` returns fraction-of-alive-cells-killed, **comparable across ops**. Without a shared severity currency, "8 damage levels" means different things per op and the damage-type ablation has no shared x-axis.

**Record now or re-run later:** the tiered episode recorder must save `symbol_vote_field: (N,T,V,32,32) f16` — the per-cell, *pre-pooling* vote. This is the spatial map of where in the body the symbol is computed and where it relocates after that region is cut out. It costs ~393 KB/episode (~75 MB total for 24 episodes). If only the pooled 8-vector is saved, the entire spatial narrative is unrecoverable and requires re-running ~80 hours of experiments.

**Testing.** Fast suite <30 s on CPU: zero-init identity (one step must be bit-identical — catches init bugs, residual sign errors, stray bias); Sobel correctness on a hand-built ramp (catches the x/y transpose, otherwise invisible); alive-mask light-cone bound; dead cells stay dead; fire-rate statistics and reproducibility; damage non-mutation and severity accuracy; straight-through Gumbel (one-hot forward, nonzero backward — catches a misplaced `.detach()` that silently turns the channel into a no-op); `assert readout.n_params == 0`; metrics against hand-computed values (uniform 8-symbol entropy = 3.0 exactly, to 1e-6). One slow integration test: 8×8 grid, 300 steps, **must reach loss < 0.05** — this is what catches "the loop runs but learns nothing."

Note: CPU and MPS produce different RNG streams from the same seed (verified). Device is part of run identity; assert same-device reproducibility only, and don't chase cross-device parity. MPS has no float64 — force `.cpu().double()` inside the `@metric` decorator so it cannot be forgotten.

---

## 17. Step 0 — disk, verified safe

Free space is **3.2 GB** (99% full). Per your instruction, `~/.git` is untouched.

I verified the Origin question directly rather than assuming: the three "origin" hits in the pip cache are `origin.json` — pip's own 330-byte wheel-cache metadata recording each cached wheel's source URL. Unrelated to your Origin project. `Origin/.venv` has **link count 1** on its binaries, so it is fully self-contained and cannot be affected by clearing any cache. It was built with uv against Homebrew Python 3.13; `~/.cache/uv` (135 MB) is left alone.

```bash
rm -rf ~/Library/Caches/pip                 # 7.4 GB — download cache only; installed packages unaffected
~/anaconda3/bin/conda clean --all -y        # ~9 GB net — 158k unshared files freed, 296k hardlinked files kept
```
No environment is deleted. Result: 3.2 GB → ~19 GB.

Then: venv at `MORPHOS/.venv` from `/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11` (python.org universal2, arm64, not brew-managed so `brew upgrade` can't break it mid-project). Install `torch==2.9.0, numpy, matplotlib, pyyaml, pytest` ≈ **750 MB**. Defer scipy to Week 2 (needed only for `linear_sum_assignment` and bootstrap CIs).

---

## 18. The exact first implementation task

**File:** `MORPHOS/morphos/substrate/nca.py` — `NCAOrganism`: fixed Sobel/identity perception buffer, update MLP with zero-initialized final layer, alive masking, `seed_state`/`step`/`rollout`/`rgba`, gradient checkpointing hook.

The one genuinely error-prone detail: with `F.conv2d(..., groups=C)`, a weight of shape `(3C,1,3,3)` assigns output channels to groups in **contiguous blocks of 3 per input channel**, so the layout is `[id_0, sx_0, sy_0, id_1, sx_1, sy_1, ...]` — *not* `[id_0..id_{C-1}, sx_0..., sy_0...]`. Get it wrong and the model still trains, just worse, and you won't notice for a week.

**Proof it works:** `tests/test_nca.py`, and specifically —

```python
def test_zero_init_is_identity():
    """fc2 zero-init => dx == 0 => first step is a no-op on every alive cell."""
    org = NCAOrganism(channels=6, hidden=16, grid=8)
    x = torch.rand(2, 6, 8, 8, generator=torch.Generator().manual_seed(0))
    x[:, 3] = 1.0                       # force all alive
    y = org.step(x.clone(), generator=torch.Generator().manual_seed(1))
    assert torch.equal(x, y), (x - y).abs().max()
```

Expected: 6 tests pass in <2 s on CPU. Plus a one-time smoke check (randomize `fc2` so growth happens, seed one cell, print alive count every 8 steps) showing monotone growth strictly under the light-cone bound `(2k+1)²` — jumping to 1024 immediately means alive-masking/padding is wrong; staying at 1 means the kernel grouping is wrong.

---

## Verification

Each stage has an automated gate; nothing advances on a subjective read.

1. `make fast` — unit suite <30 s on CPU, including the readout-is-parameter-free assertion and metrics checked against hand-computed values.
2. `make test` — tiny end-to-end training run must reach loss <0.05.
3. `scripts/bench_device.py` — reproduces the CPU/MPS table; picks the device (rule: use whichever wins the 2-organism comm step; if within 20%, use CPU for easier determinism).
4. `media/regen.mp4` — visual confirmation of grow → damage → regrow.
5. `scripts/evaluate.py --gates` — checks G1–G5 numerically and prints pass/fail per gate.
6. `make figures` — regenerates every paper figure from committed `runs/` data, proving the analysis is reproducible from logs rather than from a notebook's memory.

**Statistical plan:** seed is the unit of analysis for every cross-condition claim (episodes/placements/referents are nested within seeds — treating episodes as independent is pseudoreplication and is endemic in this literature). Four pre-registered primary tests, one per hypothesis, Holm–Bonferroni at α=0.05; everything else labeled exploratory. Kaplan–Meier with censoring for recovery times, since some episodes never recover and imputing T_max biases every mean. State honestly that n=8 powers only d ≥ 1.2.
