<!-- Experimental protocol for MORPHOS. Companion to docs/DESIGN.md. -->

# MORPHOS — Experimental Protocol, Metric Definitions, and Statistical Plan

## 0. Design decisions that must be locked before any code is written

These five choices determine whether the headline claim is even measurable. Each is a reviewer attack surface.

**0.1 The target body must be referent-independent.** One fixed target morphology per agent, identical for all referents. The referent enters *only* through the sensor channels. If the target body varied per referent, morphological recovery and semantic recovery would be the same variable and H1/H3 would be vacuous. State this in the paper in one sentence; it is the reason the dissociation is well-posed.

**0.2 Pooling must be a hard-mask uniform mean, not alpha-weighted.**

```
ℓ_v = (1/|A_t|) · Σ_{x ∈ A_t} z_v(x)          A_t(x) = 1[ max_{3x3 nbhd} α_t(x) > 0.1 ]
```

The tempting choice is `ℓ_v = Σ α(x)z_v(x) / Σ α(x)`. Do not use it. Alpha-weighting makes the morphological state a direct multiplicative modulator of the message, so during regeneration the message changes even if every cell's computation is identical. That single implementation detail would confound the entire paper. Hard mask + uniform mean decouples them, and it buys you Proposition 1 (§2.8) which is your strongest defence against the pooling confound.

**0.3 Referents must be factored, and N ≤ V^L.**
Canonical: `R = {0,1}^3`, N = 8, V = 8, L = 1 (one symbol per episode). Ceiling: 100% accuracy, I(M;R) = 3 bits. Factored referents cost nothing and make Chaabouni's topsim (Hamming meaning distance) well-defined, plus attribute-wise MI.
Extension if time permits: `R = {0,1}^4`, N = 16, L = 2, V = 8 → V^L = 64 ≥ 16, unlocks posdis/bosdis.
**Never run N > V^L.** A paper reporting 50% accuracy because 16 referents were pushed through a single 8-symbol channel is a self-inflicted wound.

**0.4 Referent injection window.** Inject the referent one-hot/binary code at the sensor patch for `t ∈ [0, T_inj)`, T_inj = 16, then remove it. This forces the sender to *store* the referent in cell state, so damage can destroy memory. If you inject continuously, the sender re-reads the referent after damage and you are studying a much weaker phenomenon. Run continuous-injection as a control condition and report both — reviewers will ask which one you did.

**0.5 The receiver receives exactly V distinct inputs.** With straight-through Gumbel the forward pass is hard one-hot. Assert at eval time that the set of distinct ear-injection tensors has cardinality exactly V. If the soft vector ever leaks through, there is an analog side channel and every discrete information measure in the paper is wrong. Make this an automated gate (§8, G4).

**Canonical hyperparameters** (adjust only via the fallback ladder in §8):
grid 32×32; C = 4 RGBA + 3 sensor + 8 vote + 13 hidden = 28 channels; T_grow = 64; T_S = 64 (sender readout); T_R = 48 (receiver rollout); Gumbel τ = 1.0 train, argmax at eval (both reported); loss `L = CE(â, r) + λ·MSE(RGBA, target)` for both agents, λ tuned so neither term dominates (start λ = 1.0, check gradient norms).

---

## 1. Four operational definitions of recovery

Notation. `t = 0` is the damage step. `0⁻` = state immediately before damage, `0⁺` = immediately after. `Δ` = steps since damage. Every damaged episode has a **matched healthy twin**: identical seed, referent, RNG stream, no damage. All "recovery" is measured against the twin *and* against the pre-damage state; the twin controls for the fact that a healthy NCA also drifts over 256 steps.

### 1.1 Morphological recovery

```
IoU_t   = Σ_x A_t(x)·A*(x)  /  Σ_x max(A_t(x), A*(x))
RMSE_t  = sqrt( Σ_{x ∈ A_t ∪ A*} ||RGBA_t(x) − RGBA*(x)||² / |A_t ∪ A*| )

MR_t = (IoU_t − IoU_{0⁺}) / (IoU_{0⁻} − IoU_{0⁺})           ∈ (−∞, ~1.05]
```

`A*` is the *target* mask (primary); also report against `A_{0⁻}` (secondary), since a trained NCA rarely hits target exactly.

**Recovered** iff `IoU_t ≥ 0.95·IoU_{0⁻}` AND `RMSE_t ≤ 1.1·RMSE_{0⁻}` AND `0.9 ≤ |A_t|/|A_{0⁻}| ≤ 1.1`, **sustained for 20 consecutive steps**. The sustain requirement kills transient touch-and-go crossings, which otherwise wreck the recovery-time statistic.

```
τ_morph = min{ t : condition holds for all of [t, t+20] },  else ∞ (right-censored at T_max = 256)
```

### 1.2 Behavioural recovery

System-level: does the receiver still get the right answer?

```
Acc_t = (1/|D|) Σ_{r ∈ D} 1[ â(r; s_t) = r ]
BR_t  = (Acc_t − 1/N) / (Acc_{0⁻} − 1/N)                    chance-corrected
```

Chance correction is mandatory. Reporting "accuracy recovered to 50%" when chance is 12.5% is meaningless without it.

**Recovered** iff a one-sided **non-inferiority test** at margin δ = 0.05 rejects `Acc_{0⁻} − Acc_t ≥ δ` at α = 0.05, sustained 20 steps. Use non-inferiority, not "no significant difference" — failing to reject a null is not evidence of recovery, and a reviewer who knows this will say so.

### 1.3 Computational recovery

Behaviour can be right for the wrong reasons. Three nested probes:

**(a) Centered-logit recovery.** Softmax is shift-invariant, so center first: `ℓ̃ = ℓ − mean_v(ℓ)`.

```
CR^logit_t = 1 − E_r|| ℓ̃_t(r) − ℓ̃_{0⁻}(r) ||₂ / E_r|| ℓ̃_{0⁺}(r) − ℓ̃_{0⁻}(r) ||₂
```

**(b) Field recovery** — did the same cells go back to the same job? Over cells alive in both states:

```
FieldCorr_t = corr_{x ∈ A_t ∩ A_{0⁻}, v} ( z_{t,v}(x), z_{0⁻,v}(x) )
```

**(c) Representational recovery (RSA).** Build the N×N dissimilarity matrix from mean-pooled **hidden** channels only (exclude RGBA and vote channels, else this is circular with (a)):

```
RDM_t[i,j] = 1 − cos( h_t(r_i), h_t(r_j) )
RSA_t      = ρ_Spearman( vec(RDM_t), vec(RDM_{0⁻}) )    over the N(N−1)/2 upper-triangular entries
```

**Recovered** iff `CR^logit_t ≥ 0.9` AND `RSA_t ≥ 0.90` AND `FieldCorr_t ≥ 0.8`.

Note the relation to the others: computational recovery implies semantic implies behavioural, but not conversely. Present this as a **lattice, not a total order** — a permuted code is structurally intact, behaviourally dead.

### 1.4 Semantic / communication recovery

Define the sender's **code** at time t as the conditional

```
σ_t(v | r) = Pr[ m = v | r, s_t ]        estimated over K Gumbel draws
σ̂_t(r)     = argmax_v ℓ_t(v | r)         the deterministic code
```

#### (A) Identity-preserving recovery — same symbol for the same referent

```
SIA_t     = (1/N) Σ_r 1[ σ̂_t(r) = σ̂_{0⁻}(r) ]                        hard
SoftSIA_t = (1/N) Σ_r Σ_v σ_t(v|r)·σ_{0⁻}(v|r)                        expected agreement of two draws
Drift_t   = (1/N) Σ_r ½·|| σ_t(·|r) − σ_{0⁻}(·|r) ||₁    ∈ [0,1]      total-variation drift  ← headline H2 quantity
```

Always **macro-average over referents** (uniform 1/N), not micro-average over episodes, so referent frequency cannot distort it. Report both if they differ.

Chance-correct with a Cohen-κ style adjustment, because a sender that collapses to always emitting symbol 3 gets free credit on any referent that used symbol 3 pre-damage:

```
κ_t = (SIA_t − p_e) / (1 − p_e),      p_e = Σ_v σ̄_t(v)·σ̄_{0⁻}(v)
```

**Recovered (identity-preserving)** iff the lower bound of the 95% bootstrap CI on SIA_t (bootstrapping **over damage placements**, §2.2) is ≥ 0.90, sustained 20 steps.

#### (B) Recovery up to relabeling — functioning but permuted code

Build the V×V confusion matrix `C[u,v] = #{ r : σ̂_{0⁻}(r) = u ∧ σ̂_t(r) = v }` and solve the linear assignment problem:

```
SIA^perm_t = (1/N) · max_{π ∈ S_V} Σ_u C[u, π(u)]
```

**Use the Hungarian algorithm** (`scipy.optimize.linear_sum_assignment(-C)`), O(V³) = 512 ops. This is an *exact* maximum over all V! = 40,320 permutations, not an approximation — it is a linear assignment problem. Say that explicitly in the paper; a reviewer will wonder if you greedily matched.

Soft version: same LAP with cost `C[u,v] = Σ_r σ_{0⁻}(u|r)·σ_t(v|r)`.

Permutation-*invariant* structure needs no matching at all:

```
struct_t = I_t(M;R) / I_{0⁻}(M;R),        I_t(M;R) = Σ_{r,v} p(r)σ_t(v|r) log₂[ σ_t(v|r) / σ̄_t(v) ]
```

#### The decomposition that carries the paper

```
struct_t   = I_t(M;R)/I_{0⁻}(M;R)          "does it still speak a language?"   (permutation-invariant)
align_t    = SIA_t                          "does it speak the SAME language?"  (permutation-sensitive)
relabel_t  = SIA^perm_t − SIA_t ≥ 0         "how much of the loss is pure relabeling?"
```

**H3 (dissociation) is exactly the event:** `MR_t ≥ 0.95 ∧ struct_t ≥ 0.90 ∧ align_t < 0.60`.
Body perfect, still linguistic, but no longer mutually intelligible with its unchanged partner. This is strictly stronger than "accuracy dropped," because it rules out "damage just made it stupid." Make this the paper's thesis sentence.

Two more semantic quantities worth reporting:

```
Acc^perm_t = max_π Pr[ â(π(m)) = r ]        accuracy under an oracle relabeler
RenegCost_t = Acc^perm_t − Acc_t            the cost of being unable to renegotiate
k_relearn   = # receiver gradient steps to return to 0.95·Acc_{0⁻}   (see baseline B3)
```

**Censoring.** Any of τ_morph, τ_behav, τ_sem may be ∞ within the horizon. Do **not** drop those episodes or impute T_max — that biases every mean. Use survival analysis (§7).

---

## 2. The metric suite

**Sampling design, stated once.** The spaces are tiny, so *enumerate* rather than sample wherever possible.

- Referents: enumerate all N. (Removes an entire variance source.)
- Damage placements: J random positions per condition. **This is the dominant variance source and the bootstrap unit.**
- Gumbel draws: K per (referent, placement).
- Counterfactual symbols in CIC: enumerate all V. (Exact, not Monte-Carlo — a genuine improvement over Lowe et al.'s sampled estimator; worth one sentence in the paper.)

Episodes per condition per seed = N·J·K = 8·32·4 = **1024**.

**Episodes control within-seed standard error; seeds control between-condition inference.** Keep those two roles separate throughout — conflating them is the most common statistical error in this subfield.

### 2.1 Task accuracy
Formula: `Acc = Pr[â = r]`. Chance = 1/N = **0.125** for N = 8.
Estimator: plug-in with **Wilson score 95% CI** (never Wald — Wald is broken near p = 1 and you will be near 1).
Sample size: to hit within-seed SE ≤ 0.015 near p = 0.9 you need n ≈ p(1−p)/SE² ≈ 400. **512 episodes per condition per seed** suffices; use 1024 for anything feeding an MI estimate.
Cost: one batched rollout. 512 × 32 × 32 × 28 floats ≈ 58 MB per state buffer — comfortable in 8 GB. Seconds.
**Also always report the per-referent accuracy vector** (needed for confound C3).

### 2.2 Protocol fidelity / SIA
Defined in §1.4. The statistical trap: SIA per damage placement is a binomial with n = N = 8 — tiny. Fix: treat the **damage placement as the sampling unit**. With J = 32 placements you get 32 quasi-independent SIA values; referents within a placement are correlated, so **bootstrap over placements, never over referents**, or use a cluster-robust CI. With between-placement sd ≈ 0.2, SE ≈ 0.035 — enough to resolve 0.1 differences.
**Set J from a pilot:** run J = 128 on one seed at the canonical cell, measure the between-placement sd, and choose J so SE ≤ 0.03. Sample-size justification from a pilot is exactly what a methods-attentive reviewer wants.
Cost: free given the rollouts.

### 2.3 Message entropy, symbol usage, effective vocabulary

```
H(M) = −Σ_v σ̄(v) log₂ σ̄(v),        σ̄(v) = Σ_r p(r) σ(v|r)
H(M|R) = H(M) − I(M;R)                                    sender stochasticity
V_eff  = 2^{H(M)}                                          perplexity  (primary — no arbitrary threshold)
V_used = #{ v : σ̄(v) > 1/(10V) }                          secondary
```

Estimator: plug-in with **Miller–Madow** correction, `Ĥ_MM = Ĥ_MLE + (m̂ − 1)/(2n)` nats, m̂ = number of observed symbols. With V = 8, n = 512 the bias is ≈ 0.01 bit — negligible, but apply and report the correction anyway; it costs one line and removes an objection. NSB is overkill here and you should say why: n/V = 64 ≫ 1.
Cost: free.

### 2.4 I(M;R) and I(M;Â)

```
I(X;Y) = Σ_{x,y} p̂(x,y) log₂[ p̂(x,y) / (p̂(x)p̂(y)) ]
```

**Plug-in MI is upward-biased.** For an N×V table, bias ≈ (m_XY − m_X − m_Y + 1)/(2n ln2) bits. With N = V = 8, n = 512: ≈ **0.069 bits** — not negligible relative to the effects you care about.
**Mandatory estimator: permutation-debiased.** Shuffle referent labels B = 1000 times, compute the null distribution of Î, report `Î_corr = Î − E[Î_null]` plus an exact permutation p-value. Cost: milliseconds on a 512-row table.
Ceiling: `I(M;R) ≤ min(H(R), H(M)) ≤ log₂ 8 = 3 bits`. Report `NMI = I(M;R)/H(R)`.
Sample size: n ≥ 16·N·V = 1024 for a comfortable MI estimate. Use n = 1024–2048 for MI-specific evaluations.
**Free correctness check:** the pipeline is a Markov chain R → M → Â, so the data-processing inequality demands `I(Â;R) ≤ I(M;R)`. If you ever measure the reverse, there is a leak (referent reaching the receiver by some path other than the symbol). Make this an automatic gate (§8, G4).
Also compute **attribute-wise MI** `I(M; a_1), I(M; a_2), I(M; a_3)` given factored referents — far more informative than a degenerate topsim.

### 2.5 Positive signalling, positive listening, CIC — adapted to L = 1

Lowe et al. define these for multi-step RL trajectories. The single-symbol setting makes them *cleaner*, not harder, and you should say so.

**Positive signalling** (message correlates with sender observation). Here the sender's observation is exactly the referent:

```
PS  = I(M;R) / H(R)              ∈ [0,1]
SC  = (1/N) Σ_r max_v σ(v|r)     "speaker consistency" — determinism of the code
```

**Positive listening / CIC** (message causally changes receiver behaviour). Intervene on m with the receiver's state and seed held fixed, and **enumerate all V counterfactual symbols**:

```
CIC = (1/|E|) Σ_{e ∈ E}  D_KL[  p(â | m_e, e)  ||  Σ_v σ̄(v)·p(â | v, e)  ]
```

Two important points. (i) Because m is *randomized within episode*, conditioning equals `do(m)`, so this is a genuine causal effect, not a correlation — that is the whole content of Lowe's critique. Formally `CIC = I(M; Â | E)`. (ii) Use the receiver's **softmax over answer logits** as `p(â|·)`, not the argmax — a smooth low-variance estimate, and KL of a degenerate distribution is ill-behaved.

A blunter companion with an intuitive scale:

```
CSE = Pr_{v ~ Unif(V)} [ â(v) ≠ â(m_e) ]        fraction of counterfactual symbols that flip the answer
```

Cost: V receiver rollouts per episode = 8× base. 256 episodes × 8 = 2048 batched receiver rollouts — trivial. Sample size: 256 episodes (variance is across episodes only, since v is enumerated).

### 2.6 Topographic similarity — and when *not* to use it

```
ρ_topo = ρ_Spearman( { d_R(r_i, r_j) }_{i<j} , { d_M(m_i, m_j) }_{i<j} )
d_R = Hamming on the attribute vector;  d_M = edit distance on the message string
```

**At L = 1, edit distance is binary {0,1} and topsim degenerates into a point-biserial correlation. It is nearly meaningless.** Say this in the paper rather than reporting a number that looks like Chaabouni's but isn't.

Use topsim only when (a) referents have attribute structure (guaranteed by §0.3) **and** (b) L ≥ 2. At L = 1, report attribute-wise MI instead.
Estimator: exact enumeration over C(N,2) = 28 (N=8) or 120 (N=16) pairs; significance by referent-label permutation test, B = 10,000. Cost: trivial.
**Chaabouni caveat, always printed next to the number:** across 141 settings they found no correlation between compositionality metrics and generalization. Topsim is reported *alongside* accuracy, never as a proxy for it.

### 2.7 Cell-level consensus (adapted from Self-classifying MNIST)

Each alive cell has a private opinion `v̂(x) = argmax_v z_v(x)`.

```
Consensus_t   = max_v (1/|A_t|) Σ_{x∈A_t} 1[ v̂_t(x) = v ]          plurality fraction ∈ [1/V, 1]
Agree_t       = (1/|A_t|) Σ_{x∈A_t} 1[ v̂_t(x) = σ̂_t(r) ]           agreement with the emitted symbol
FullConsensus = fraction of episodes with Agree_t = 1.0
```

Randazzo et al. report 95.3% accuracy at step 200 with **88.1% of samples reaching full per-cell agreement** — a direct numerical comparison point for your paper. Use it.
Cost: one argmax over the grid. Free.

### 2.8 Distributedness of symbol computation (H4)

**Proposition 1 (deletion-invariance under consensus).** If `argmax_v z_v(x) = v*` for every `x ∈ A`, then for every nonempty `S ⊆ A`, `argmax_v (1/|S|) Σ_{x∈S} z_v(x) = v*`.
*Proof.* Full consensus means `z_{v*}(x) > z_u(x)` for all x and all u ≠ v*. Averaging preserves strict inequality termwise. ∎

This is the single most useful lemma in the project: **under full per-cell consensus, cell deletion cannot mechanically flip the symbol**, so any symbol change must reflect a genuine change in cell computation. It is the backbone of confound control C1.

Quantitative version. Per-cell margin `μ_u(x) = z_{v*}(x) − z_u(x)`. The symbol survives deletion of S iff

```
Margin(S) = min_{u ≠ v*}  (1/|A\S|) Σ_{x ∈ A\S} μ_u(x)  > 0
```

**Worst-case adversarial deletion tolerance.** For fixed u, removing the k largest values of μ_u minimizes the mean of the remainder (swapping a removed element for a larger kept one strictly decreases it). So

```
k* = min{ k : ∃u, mean of the |A|−k smallest μ_u values ≤ 0 }
```

is computable *exactly* in `O(V·|A|·log|A|)` per episode with zero extra forward passes. `k*/|A|` is a crisp per-episode robustness number.

**Per-cell causal contribution — closed form, zero cost.** For mean pooling, leave-one-out is exact:

```
φ(x) = ℓ_{v*}(A) − ℓ_{v*}(A\{x}) = ( z_{v*}(x) − ℓ_{v*} ) / (|A| − 1)
```

**Distributedness statistics:**

```
PR   = ( Σ_x |φ(x)| )² / ( |A| · Σ_x φ(x)² )        ∈ [1/|A|, 1]      1 = perfectly uniform
IPR  = ( Σ_x |φ(x)| )² / ( Σ_x φ(x)² )               ∈ [1, |A|]        "effective number of participating cells"
Top10 = fraction of Σ|φ| carried by the top 10% of cells
Gini(|φ|)
```

**Ablation redundancy curve** (dynamics-free probe at readout, so it measures computational redundancy rather than regeneration):

```
SP(f) = E_{|S| = f|A|} [ 1[ argmax ℓ(A\S) = argmax ℓ(A) ] ]
f_50  = min{ f : SP(f) < 0.5 };      AURC = ∫₀¹ SP(f) df
```

**Dynamics-aware version** (stronger): ablate an 8×8 tile at t₀, roll to T_S, measure symbol-flip probability → causal importance map ψ(x). 16 tiles per episode is cheap; single-cell (≈400 ablations/episode, batched over cells) only if time permits.

**⚠ The H4 confound you must confront head-on.** Mean pooling *architecturally imposes* uniform participation on the vote channels. "Communication is distributed" could be a tautology of the readout. Three within-architecture controls:
1. **Channel-matched:** compare `f_50^vote` vs `f_50^RGBA` — both are mean-pooled parameter-free readouts, so this is apples-to-apples.
2. **Untrained-NCA normalization:** report ΔPR relative to a random-weight NCA, i.e. the deviation from the architectural ceiling.
3. **Probe-matched:** fit post-hoc linear probes on frozen per-cell hidden states for (a) the referent and (b) a referent-irrelevant quantity (cell position, local body-part identity, step count); compare their ablation-redundancy curves.

Even so, H4 is the weakest of the four hypotheses and I would **demote it to a secondary result** in the paper. Do not let it hold up the main narrative.

### 2.9 Morphological IoU / RGBA distance
Defined in §1.1. Two additions: (i) track `|A_t|` explicitly — overgrowth/explosion is the classic NCA failure mode; (ii) impose a stability gate `0.7 ≤ |A_t|/|A*| ≤ 1.3` for all `t ∈ [T_grow, 4·T_grow]`.

### 2.10 Readout algebra — a remark worth including in the paper

Under full consensus, **mean-pool** argmax is deletion-invariant (Prop. 1) but the logit *magnitude* changes little. Under **sum-pool with dead cells contributing zero**, deleting cells shrinks all logits, which flattens the softmax and raises `H(M|R)` even when the argmax is stable. So the pooling choice determines whether damage manifests as *argmax flip* (semantic change) or as *entropy increase* (confidence change). That is precisely the C1/C2 distinction. Train with mean-pool, and report the sum-pool readout evaluated on the same states as a diagnostic that separates the two channels.

---

## 3. The intervention suite

| # | Intervention | Tests | A null result falsifies | Status |
|---|---|---|---|---|
| 1 | **Remove message** (zero ear injection) | Is the message used at all | Acc unchanged ⇒ no communication; receiver is using priors or there is a leak | **MANDATORY** |
| 2 | **Random symbol**, resampled from the marginal σ̄ | Positive listening / CIC on-distribution | Acc unchanged ⇒ protocol not causally real | **MANDATORY** |
| 3 | **Permute vocabulary** (fixed bijection π before the ear) | Does the receiver read symbol *identity*, or just "message present"? | Acc holds under π ⇒ receiver reads energy/presence, not identity | **MANDATORY** |
| 4 | **Cross-play across seeds** (sender_i × receiver_j) | Is the protocol a convention or a universal | Cross-play accuracy high ⇒ no real convention emerged | **MANDATORY** — it supplies the reference scale ("totally incompatible") that makes post-damage drift interpretable |
| 5 | **Ablate cell regions** at readout | H4 + confound C1 | SP(f) flat ⇒ symbol is carried by few cells | **MANDATORY** for the confound use, optional for H4 |
| 6 | **Activation patching healthy→damaged** (copy the twin's state, restricted by channel subset × region) | *Where* the semantic information was lost: regional vs readout-level | Patching hidden channels in the regrown region restores the symbol ⇒ deficit is local | High value, optional — this is the workshop-paper → strong-paper upgrade |
| 7 | **Ablate channels** (zero hidden channel c) | Functional specialization; "the protocol lives in 3 of 13 channels" | — | Optional, cheap. **Caveat to pre-empt:** channels are not a privileged basis; add a random-rotation or PCA-direction control or a reviewer will note that channel ablations are basis-dependent |
| 8 | **Symbol noise** (flip w.p. ε) | Graceful degradation / channel capacity | — | Optional; also useful as a *training* condition (noise-trained baseline) |
| 9 | **Remove one symbol** (forbid v; fall back to argmax of the rest) | Per-symbol necessity, code redundancy | Removing v costs nothing ⇒ v is unused or duplicated | Optional, ~8 extra eval runs |
| 10 | **Delay / pulse messages** (inject at step d; hold for k steps) | Receiver temporal integration and memory | Acc invariant to d ⇒ receiver has no temporal structure | Optional, cheap: sweep d ∈ {0,5,10,20}, k ∈ {1,5,∞} |
| 11 | **Replay message from another episode** | Episode-specific analog side channel | — | **Largely vacuous here.** With hard one-hot symbols, replaying a same-index symbol from another episode is *literally the identical tensor*. Only meaningful if a soft/continuous vector ever reaches the ear. Replace with the assertion in §0.5 |

**Ranking by value per unit cost:** 1 > 2 > 3 > 4 > 5 > 6 > 7 > 8 > 9 > 10 > 11.

**Mandatory minimum set for a 4-week workshop paper: {1, 2, 3, 4} plus 5 restricted to the C1 confound control.** All five are "change one line in the eval loop" once the harness exists; there is no excuse for omitting any of them, and omitting 1–3 makes the paper unpublishable under the Lowe et al. standard.

---

## 4. The damage experiment protocol

### 4.1 Factors and levels

**F1 — Damage type (5 levels)**
- **D1 Disk kill** (canonical): zero all channels in a disk at a random on-body position.
- **D2 Half-cut**: zero one half-plane — the planarian-style bisection, deterministic, ~50% of cells.
- **D3 Hidden-channel-only damage**: zero hidden channels in a region, leave RGBA intact. **This is the key manipulation for H3** — it induces computational damage with minimal morphological damage.
- **D4 Noise damage**: add `N(0, σ²)` in a region (perturbation, not deletion).
- **D5 Sensor-targeted damage**: kill the sensor patch specifically. Distinguishes destroying the *input pathway* from destroying the *computation*. Reviewers will ask; run it.

**F2 — Damage magnitude (6 levels), defined on ALIVE cells**
`f ∈ {0 (sham), 0.10, 0.20, 0.30, 0.45, 0.60}`
Solve the disk radius ρ **per episode by binary search on the current alive mask** to hit target f within ±2%. Defining magnitude by radius instead of alive-fraction confounds severity with body size and regeneration state — a subtle but real error.

**F3 — Damage timing (3 levels)**
`t_dmg ∈ {T_grow/2 (mid-growth), T_grow (maturity — canonical), T_grow + 50 (late/persistent)}`

**F4 — Which agent (4 levels)**
`{none (sham), sender-only (canonical), receiver-only, both}`
Receiver-only is a mandatory secondary: it is the mirror image (intact speaker, damaged listener) and tests whether the asymmetry is real.

**F5 — Training condition (4 levels)**
- **T1 damage-naive**: grow + persist only.
- **T2 damage-trained**: damage sampled into the training pool (Mordvintsev-style), applied to the sender, with the task loss active during damaged steps.
- **T3 damage-trained, morphology-only**: damage in the pool, but the task loss is detached/masked during the damaged steps. It learns to regrow but never learns to *communicate while damaged*. **This is the crucial control separating regeneration ability from semantic robustness — it is what makes H2 a mechanistic claim rather than a correlation.**
- **T4 both agents damage-trained** (optional).

Full factorial = 5 × 6 × 3 × 4 × 4 = 1440 cells. Do not run it.

### 4.2 What is actually run — a star design around a centre point

Centre point: **D1, f = 0.30, timing = maturity, sender-only, T1**.

| Block | Design | Conditions |
|---|---|---|
| **Core (paper's spine)** | D1 × f ∈ {0,.1,.2,.3,.45,.6} × maturity × sender-only × {T1,T2,T3} — **fully crossed** | 18 |
| Sweep A: damage type | type ∈ {D1,D2,D3,D5} at f=.3, sender-only, maturity, {T1,T2} | 8 |
| Sweep B: which agent | agent ∈ {sender, receiver, both} at D1, f=.3, {T1,T2} | 6 |
| Sweep C: timing | timing ∈ {mid, maturity, late} at D1, f=.3, {T1,T2} | 6 |
| Sham | f = 0 in every training condition | 3 |

≈ 40 conditions × 8 seeds. Everything except the core block is one-factor-at-a-time.

**Justify the design in the paper:** you care about main effects on the recovery curve, and about exactly **one interaction — f × training-condition — which is the entire content of H2 and is therefore fully crossed (6 × 3).** High-order interactions are out of budget and out of scope. Saying this pre-empts "why not a full factorial."

### 4.3 Recovery time-course protocol

Damage at Δ = 0. Evaluate at

```
Δ ∈ {0, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256}
```

(dense early, log-ish later; T_max = 256 ≫ T_grow = 64).

At each Δ record: `IoU, RMSE, |A|, MR, Acc, BR, SIA, SoftSIA, Drift, κ, SIA^perm, I(M;R), H(M), H(M|R), V_eff, Consensus, Agree, logit vector ℓ, logit margin γ, PR, k*, CR^logit, RSA, FieldCorr`.

**Receiver protocol:** for each Δ, run the receiver **fresh from its own seed for T_R steps** using the symbol emitted at that Δ. This decouples receiver dynamics from sender time. (The alternative — one continuous receiver run with a time-varying symbol — entangles them and is much harder to interpret. State which you did.)

Beyond wall-clock Δ, also evaluate at **matched morphology**: bin episodes by IoU into 20 bins and report semantic metrics per bin. This is what H2 and H3 actually require, and it eliminates "the sender just wasn't done yet" as an explanation (see C4).

### 4.4 Episodes per condition

```
N referents (enumerated) × J = 32 damage placements × K = 4 noise draws = 1024 episodes / condition / seed
```

K = 1 with deterministic argmax for pure code metrics; K = 16 for stochastic-behaviour metrics; J = 64 for MI-heavy evaluations. **Set J from a pilot** (§2.2).

**Pair the design:** use the *same 32 placement seeds* across T1/T2/T3 and across damage magnitudes. Paired comparisons across training conditions dramatically increase power at n = 8 seeds. This is free and it is the difference between a significant and a non-significant H2.

Memory: batch 256 episodes at a time (~30 MB per state buffer at C = 28). Fits easily.

---

## 5. Confound analysis for the central control

The claim under attack: *"damage the sender, freeze the receiver, therefore the accuracy drop measures semantic drift."* Here is every alternative explanation I can construct, with a control for each.

### C1 — Mechanical pooling shift
*The readout is a mean over alive cells. Deleting 30% of cells changes the mean even if every surviving cell computes exactly what it did before.* This is the most dangerous objection and it deserves three independent answers.

**(a) The consensus argument (Proposition 1, §2.8).** If per-cell argmax consensus ≈ 1.0 and the per-cell margin is positive everywhere, deletion *provably cannot* flip the symbol. Report `Consensus`, `Agree`, and `min_x μ(x)` pre- and post-damage at every Δ. Where consensus is high, C1 is formally excluded, not merely argued against.

**(b) The frozen-state deletion control — exact, zero forward passes, and the single best answer.** Take the healthy pre-damage state `s_{0⁻}`, apply the *same* deletion mask S, and recompute the pooled readout **without running any dynamics**:

```
ℓ^mech = pool( z_{0⁻}, A_{0⁻} \ S )

Mechanical component = SIA( ℓ^mech , ℓ_{0⁻} )      pure pooling artifact
Dynamical  component = SIA( ℓ_t     , ℓ^mech  )     genuine change in cell computation
```

This decomposes total drift into (mechanical | dynamical) exactly, costs nothing, and becomes Figure 6. Any reviewer raising C1 is answered by pointing at that panel.

**(c) Readout-family robustness.** Evaluate the same states under sum-pool, median-pool, and per-cell majority vote (see the readout algebra remark, §2.10). If the phenomenon exists only under mean-pool, it is an artifact. If time permits, train **one extra seed set with a deletion-robust readout (per-cell majority vote)** and show the dissociation survives — this converts a robustness check into a positive result.

Also report **centered** logits throughout, so a uniform offset is not miscounted as semantic change.

### C2 — Entropy increase (uncertainty), not argmax flip
*The sender may become less confident without changing its preferred symbol; sampled accuracy drops from a flat softmax, not from a changed code.*

Controls:
- Report **both** `Acc^argmax` (τ→0) and `Acc^sampled` (train τ) at every Δ. The gap *is* the uncertainty component:
  `ΔAcc = [drift component: argmax flips] + [uncertainty component: Acc^argmax − Acc^sampled]`.
- Track the **logit margin** `γ(r) = ℓ_{v*} − max_{u≠v*} ℓ_u` and `H(M|R)` over Δ.
- **Temperature sweep**: if the drop vanishes as τ → 0, it is confidence, not code.
This decomposition should be a supplementary panel; it is cheap and it is exactly what a careful reviewer will ask for.

### C3 — Referent heterogeneity
*Some referents are intrinsically harder; damage may only kill the marginal ones, and the mean hides it.*

Controls:
- **Never report only the mean.** The referent × Δ accuracy/SIA heatmap goes in the supplement always.
- **Mixed-effects model:** `outcome ~ condition + (1|referent) + (1|placement) + (1|seed)`.
- Use **pre-damage per-referent accuracy as a covariate** and test whether Δacc correlates with pre-damage difficulty. If it does, that is a *finding* ("damage preferentially destroys marginal referents"), not a bug — report it as such.
- Enumerate referents; never sample them.

### C4 — Mid-regeneration ("not done computing")
*The sender may simply need more steps.*

Controls:
- The **time course itself** is the control: run to T_max = 256 ≫ T_grow = 64 and report the asymptote (mean over the last 32 steps).
- **Matched-morphology analysis** (§4.3): report semantic metrics binned by IoU rather than by wall-clock Δ. At matched IoU, "not done yet" is definitionally excluded.
- **Convergence criterion:** require `||ℓ_t − ℓ_{t−8}||_∞ < ε` before calling the symbol final; report the fraction of episodes meeting it.
- **Sham (f = 0) control is non-negotiable.** Healthy NCAs drift over long horizons. Any "drift" measured at f = 0 is metric noise and must be shown to be ≈ 0 (gate G5).

### C5 — Damage destroyed the *sensor*, not the computation
*If the disk overlaps the sensor patch, the referent information is literally deleted. Trivial, not interesting.*

Controls:
- **Stratify by sensor overlap** (disk ∩ sensor = ∅ vs ≠ ∅). Report the headline result on the **non-overlapping stratum**.
- Run **D5 (sensor-targeted)** explicitly as the contrast case.
- Related and important: the referent-injection window (§0.4) determines whether you are studying *memory* regeneration or *computation* regeneration. Report both the T_inj = 16 (canonical) and continuous-injection variants.

### C6 — The receiver's input amplitude changed
Impossible by construction if the ear receives a hard one-hot. **Assert** that the ear input takes exactly V distinct values across all episodes (gate G4).

### C7 — The receiver is not actually frozen
Hash receiver parameters before and after; assert identical receiver initial state and RNG stream across all conditions. Trivial, but state it — "frozen receiver" is the paper's core control and it should be verified, not assumed.

### C8 — Damage-trained models are just better (or worse) models
T1/T2/T3 differ in training, so pre-damage accuracy may differ, contaminating every H2 comparison. Three layers of control, all of them:
- **ANCOVA** with pre-damage accuracy as a covariate.
- **Checkpoint matching**: select checkpoints with pre-damage accuracy matched to ±2 points.
- **Normalized recovery** BR (chance-corrected *and* pre-damage-normalized) as the reported outcome.
- Plus **T3**, which shares regeneration training with T2 but not semantic-under-damage training — the cleanest available control.

### C9 — Selection effects
Pre-register the seed list and the learning-success gate. Report **all** seeds including failures, and report the emergence rate separately (§7).

### C10 — Phase mismatch
If damaged dynamics are slowed, a fixed readout time compares different dynamical phases. Report readouts at both matched wall-clock Δ and matched dynamical phase (the C4 convergence criterion).

---

## 6. Baselines

### 6.1 Within-experiment baselines (all mandatory or near-mandatory)

- **B0 — Sham damage (f = 0).** Controls for time drift of the metrics and of the healthy NCA. **Mandatory.** Without it, every drift number is uninterpretable.
- **B1 — Untrained / random-weight NCA sender.** Floor for every metric. Mandatory, free.
- **B2 — T1 / T2 / T3.** The H2 comparison plus its mechanistic control. Mandatory.
- **B3 — Receiver re-adaptation.** After damage, allow the frozen receiver to fine-tune for k gradient steps; report `k_relearn` = steps to regain 0.95·Acc_{0⁻}. **High value, cheap.** If a handful of steps restores accuracy, the code is intact-but-permuted and you have directly measured "the cost of not being able to renegotiate." This is the empirical counterpart to `SIA^perm`.
- **B4d — Learned-readout NCA.** Replace the parameter-free mean pool with a learned linear readout on the flattened grid. Prediction: position-dependent readout ⇒ far less damage-robust. **This directly defuses the attack "your whole result is a property of mean pooling."** Cheap, one extra training run. Recommended.

### 6.2 Substrate baselines — and how to make the damage comparison fair

This is where a hostile reviewer will concentrate fire, so be explicit about the matching problem.

**Structural matching does not work across substrates.** "30% of NCA cells" versus "damage to an MLP" has no structurally commensurable answer:
- *Match parameters*: an NCA shares one small MLP across 1024 cells; a parameter-matched MLP has wildly different per-unit capacity. Weak.
- *Match units/activations*: match hidden width so total activation count equals H·W·C. Reasonable for "kill 30% of units," but still not equal *function* destroyed.
- *Match FLOPs*: matches compute, not vulnerability.

**The defensible matched quantity is destroyed task-relevant information, measured empirically at the instant of damage.** Structural quantities are incommensurable across substrates; the only common currency is *how much of the referent-relevant representation the damage removed*. Define, with a linear probe R̂ fit on the pre-readout representation:

```
Sev = 1 − I( R̂_{0⁺} ; R ) / I( R̂_{0⁻} ; R )
```

Then titrate the ablation fraction independently in each substrate to hit matched Sev levels {0.25, 0.5, 0.75, 1.0}, and ask: **given equal immediate functional damage, who recovers?**

The intuitive headline version of the same idea, and the one to put in the paper: **titrate damage in each substrate so that the immediate post-damage accuracy is equal** (e.g. both drop to chance), then compare recovery trajectories. Very hard to argue with.

Also mandate: **all substrates trained to within ±2 points of the same pre-damage accuracy**, or reported with pre-damage accuracy as a covariate. A baseline that was worse before damage cannot be compared for robustness after it.

Report the alternative matchings (matched activations zeroed, matched parameters zeroed) as robustness rows in a table, and say plainly that they disagree — that honesty is worth more than a false claim of a canonical match.

**A structural point that reframes the comparison:** most non-iterative baselines *cannot regenerate at all* — an MLP with zeroed units stays zeroed forever, since there is no state-update dynamics. Comparing NCA-vs-MLP recovery is therefore close to comparing a system that has a recovery mechanism with one that has none. That is not an interesting result, and a good reviewer will say so.

**My recommendation: do not promise "NCA vs MLP" in the paper.** Instead run within-NCA ablations that isolate what actually makes an NCA an NCA — locality, weight sharing, and readout — which are far better controlled:

- **B4a — Non-local NCA**: same per-cell MLP plus a global average-pooled context vector appended to every cell's input. Isolates **locality**.
- **B4b — Non-shared NCA**: per-cell parameters (cheap variant: per-cell learned gain/bias only, to stay inside 8 GB). Isolates **weight sharing** — is regeneration a consequence of translation-equivariant shared weights?
- **B4c — Conv-GRU recurrent CNN**: iterative and local but not cell-autonomous (no alive masking, global normalization). Isolates the **NCA framing vs. plain recurrence**.
- **B4d — Learned readout** (above). Isolates the **parameter-free pooling**.

**Necessity ranking:** B0 > B2 > B1 > B4d > B3 > B4a > B4c > B4b > cross-substrate.
**4-week set: B0, B1, B2, B3, B4d.** Everything else is deferrable.

---

## 7. Statistical plan

### 7.1 Seeds and units
**S = 8 seeds per training condition** (field norm is 4–6: Chaabouni 4, Rita 6 — exceeding it is cheap here and is a stated differentiator).

*Varies across seeds:* NCA parameter init, training pool sampling / data order, Gumbel noise stream, training-time damage placements.
*Held fixed across seeds:* target body, referent set, sensor/ear locations, architecture, hyperparameters, and — critically — the **evaluation damage placements** (paired design).

Sensor/ear geometry is fixed in the main experiment; run a 3-position robustness check on 3 seeds so you can answer "does this depend on where you put the sensor?"

**The seed is the unit of analysis for every cross-condition claim.** Compute a per-seed summary statistic, then infer across seeds (n = 8). Episodes, placements, and referents are *nested within* seeds. Treating episodes as independent samples for cross-condition tests is pseudoreplication and it is endemic in this literature — say explicitly that you avoided it.

### 7.2 Reported statistics
Mean ± std across seeds (the near-universal field convention) **plus every individual seed plotted as an overlaid point**, plus a BCa bootstrap 95% CI over seeds (B = 10,000). With n = 8 there is no excuse for hiding the seed-level scatter.

### 7.3 Tests, one per hypothesis

| Hypothesis | Outcome | Test | Effect size |
|---|---|---|---|
| **H1** τ_sem > τ_morph | per-seed median paired difference | **Wilcoxon signed-rank** (n=8, paired, non-normal) + **stratified log-rank** on Kaplan–Meier curves for time-to-recovery, stratified by seed, censored at T_max | median paired Δ (steps); rank-biserial r |
| **H2** damage-trained drifts less at matched morphology | per-seed `Drift` at the first Δ with IoU ≥ 0.95·IoU₀ | paired **Wilcoxon signed-rank** if paired by placement; else Mann–Whitney U (8 vs 8). Confirmatory: mixed model `drift ~ train_cond * IoU_bin + (1|seed) + (1|placement) + (1|referent)` | Cliff's δ |
| **H3** dissociation exists | rate of `{MR ≥ .95 ∧ struct ≥ .90 ∧ align < .60}` | it is an **existence claim**: report the rate with a **cluster bootstrap CI over seeds**; one-sided test that rate > 0. Companion: paired Wilcoxon on `f_50^morph` vs `f_50^sem` from the dose-response curves | dissociation rate; f_50 gap |
| **H4** communication more distributed | per-seed `PR^comm` vs `PR^morph`, `f_50^comm` vs `f_50^morph` | paired **Wilcoxon signed-rank** | matched-pairs rank-biserial |

**Censoring is not optional.** Some episodes never recover. Kaplan–Meier + log-rank is the correct treatment; means over recovery times with ∞ dropped or imputed as T_max are biased and a reviewer who knows survival analysis will catch it.

**Power, stated honestly.** At n = 8 vs 8, Mann–Whitney has minimum attainable p ≈ 2×10⁻⁴, and power ≈ 0.8 only for large effects (d ≳ 1.2–1.5). **Pre-register that the study is powered for d ≥ 1.2 and say so in the paper.** Formal significance testing is not standard in this subfield; doing it *and* being candid about power is a genuine methodological contribution rather than statistical theatre.

### 7.4 Multiple comparisons
Pre-register exactly **four primary tests** (one per hypothesis) and control FWER at α = 0.05 with **Holm–Bonferroni** across those four. Everything else is explicitly labelled **exploratory**, reported with uncorrected p-values *and* effect sizes, clearly marked as such (or BH-FDR at q = 0.10 within the exploratory family). Say this in the methods section.

### 7.5 Seeds that fail to learn a protocol
- Pre-register the emergence gate (§8, G3).
- Report **protocol emergence rate** = fraction of seeds passing the gate, per training condition, with a Wilson CI. **This is itself a result** — emergent-communication papers routinely hide it.
- Run primary analyses on **gate-passing seeds only**, reporting the exclusion count; add an intent-to-treat analysis over all seeds as a robustness check.
- **Do not replace failed seeds with fresh ones.** Pre-commit to S = 8 and report all 8. If fewer than 5 pass, the paper reframes (§9).
- **Collider warning to pre-empt:** if damage-training changes the emergence rate, conditioning on gate-passage is post-treatment selection. Report emergence rate per condition *first*; if the rates differ, state the caveat explicitly.

### 7.6 Figures

**Fig 1 — Setup schematic** (no data). Sender grid with sensor patch → vote channels → mean pool → Gumbel symbol → receiver ear patch → answer; inset timeline showing growth, damage, regeneration, and the readout times.
*Claim:* this is a well-posed referential game on a regenerating substrate.

**Fig 2 — The dissociation time course (money figure).** x = steps since damage (log-ish); y = normalized recovery ∈ [0,1]; four curves: MR (morphology), BR (behaviour), SIA (semantic identity), struct (permutation-invariant structure). Mean ± 95% CI across seeds; sham dashed.
*Claim (H1):* morphology recovers first and fully; semantics lags, and in some regimes never returns.

**Fig 3 — Dose-response and dissociation map.** Panel A: x = damage fraction f, y = asymptotic recovery, one curve each for morphology / accuracy / SIA / MI. Panel B: scatter, x = morphological recovery, y = SIA, one point per (seed, placement), coloured by f, with the dissociation quadrant (MR > 0.95, SIA < 0.6) shaded and its occupancy percentage annotated.
*Claim (H3):* a damage regime exists where morphology fully recovers and semantics does not.

**Fig 4 — Damage-training reduces drift at matched morphology.** x = IoU bin (morphology explicitly controlled), y = semantic drift; three lines (damage-naive T1, damage-trained T2, morphology-only T3), per-seed points overlaid.
*Claim (H2):* the effect is semantic robustness, not merely better regeneration.

**Fig 5 — The protocol is causally real.** Bar chart: intact / message removed / marginal-resampled symbol / vocabulary permuted / cross-play, with a chance line; CIC value with CI annotated.
*Claim:* Lowe et al. compliance — positive listening is demonstrated causally, not correlationally.

**Fig 6 — Mechanical vs genuine drift.** Over Δ: total drift decomposed into the frozen-state deletion component and the dynamical component; second axis shows per-cell consensus and min margin.
*Claim:* the drop is not an artifact of mean pooling.

**Fig 7 (H4, demotable to supplement) — Spatial redundancy.** Panel A: ablation-redundancy curves SP(f) for symbol vs morphology vs matched control probe. Panel B: per-cell causal contribution heatmaps and PR distributions.
*Claim:* communication-relevant computation is more spatially redundant.

**Fig 8 (supplementary) — Recovery up to relabeling.** Bars: Acc, Acc under optimal permutation, Acc after k receiver fine-tuning steps; plus pre/post confusion matrices with the Hungarian matching drawn.
*Claim:* the regenerated organism speaks a coherent but different language.

Plus a **table**: all metrics × conditions, mean ± std across seeds, with the pre-registered gates marked pass/fail.

---

## 8. Pass/fail gates

Every gate is a number a script can check.

**G0 — Compute budget.** Training one seed ≤ 90 min on M2; full eval suite for one seed ≤ 30 min. If exceeded, descend the pre-registered **fallback ladder**: N = 8 → N = 4 (V = 4, chance 0.25); grid 32 → 24; C = 28 → 20; T_S = 64 → 48. Budget check: 8 seeds × 3 training conditions × 45–90 min ≈ 18–36 h wall clock — feasible over ~1 week of overnight runs. **Run a 2-seed pilot first.**

**G1 — Growth and persistence.** Over 128 rollouts: `IoU(T_grow) ≥ 0.90` AND `RMSE(T_grow) ≤ 0.05` AND at `t = 4·T_grow = 256`: `IoU ≥ 0.85` and `0.9 ≤ |A_256|/|A_64| ≤ 1.1`. (The persistence clause catches the classic "explodes past the training horizon" failure.)

**G2 — Regeneration.** After D1, f = 0.30 at maturity: `IoU ≥ 0.90 within 128 steps in ≥ 90% of 256 (placement × seed) trials`.

**G3 — A protocol emerged.**
*Strong tier:* `Acc_{0⁻} ≥ 0.90` (chance 0.125) with Wilson lower bound ≥ 0.85 over 512 episodes, AND `I(M;R) ≥ 0.8·log₂N = 2.4 bits` (permutation-debiased), AND `V_eff ≥ 0.75·N = 6`.
*Weak tier ("some protocol"):* `Acc ≥ 0.50` AND `I(M;R) ≥ 1.0 bit`.
Report emergence rate at both tiers.

**G4 — The protocol is causally real.** All must hold:
- Message removed: `Acc ≤ 1/N + 0.05 = 0.175`
- Marginal-resampled symbol: `Acc ≤ 0.175`
- Vocabulary permuted (random π ≠ id): `Acc ≤ 1/N + 0.10 = 0.225`
- `CIC ≥ 1.0 bit` (ceiling 3 bits)
- Cross-play across seeds: `Acc ≤ 1/N + 0.10` (confirms a convention, not a universal)
- **Data-processing sanity: `I(Â;R) ≤ I(M;R) + 0.1 bit`. Violation ⇒ pipeline leak ⇒ stop and debug.**
- **Channel purity: the set of distinct ear-injection tensors has cardinality exactly V.** Violation ⇒ analog side channel ⇒ every discrete metric is invalid.

**G5 — The damage experiment produced a signal.** At the canonical cell (D1, f = .3, sender-only, maturity, T1):
- Immediate drop: `Acc_{0⁻} − Acc_{0⁺} ≥ 0.20` (else damage is too weak to study)
- Sham validity: `|Acc(Δ=128) − Acc_{0⁻}| ≤ 0.03` AND `sham SIA ≥ 0.97` (else your metrics are noise)
- Real effect: asymptotic (Δ = 256) drift under damage exceeds sham drift, paired Wilcoxon over 8 seeds, p < 0.05, with `ΔSIA ≥ 0.10`
- Dissociation is possible: `MR(Δ=128) ≥ 0.95 in ≥ 80% of trials` (else you can only claim "damage breaks things," not a dissociation)
- **Minimum interesting effect for H1, pre-registered:** `median(τ_sem − τ_morph) ≥ 16 steps`

---

## 9. The minimum publishable result

If everything else fails, the contribution that survives is **the dissociation itself, demonstrated once, cleanly, with the protocol proven causally real.** Nobody has measured semantic recovery in a regenerating substrate, so even a single-condition, damage-naive result is a workshop contribution — *provided* the causal-reality controls are airtight. The controls are what make it a result rather than an anecdote.

**Minimum experiments (fits in ~2 weeks):**
1. Train damage-naive sender + receiver, 4–6 seeds; pass G1, G3, G4.
2. One damage condition (D1, f = 0.30, sender-only, maturity) + sham, full time course to Δ = 256.
3. Lowe intervention battery: remove / marginal-resample / permute, plus cross-play.
4. The mechanical-vs-genuine decomposition (free — no extra forward passes).

**The four-figure paper:**
- **Fig A — Hook.** Setup schematic plus a regeneration filmstrip with the emitted symbol printed under each frame. The body comes back; the word changes. This single image is the paper.
- **Fig B — H1.** Recovery time course, morphology vs semantics, mean ± CI across seeds, sham overlaid.
- **Fig C — Causal reality.** Intervention bar chart with CIC. Proves the protocol is not a correlational artifact.
- **Fig D — Not an artifact.** Mechanical-vs-genuine drift decomposition plus consensus/margin traces.

*Claim:* "Morphological regeneration does not restore communicative meaning. We exhibit and quantify a morphology/semantics dissociation in a regenerating neural cellular automaton, with the emergent protocol verified causally real by intervention."

**Fallback below the minimum.** If no protocol emerges at all, the paper becomes a negative result on the trainability of discrete emergent communication through local-update substrates: emergence rate across seeds and hyperparameters, characterized failure modes (symbol collapse, consensus failure, morphology/task loss interference), and the diagnostic suite itself. The metric definitions and gates in this document are a contribution in their own right — "a measurement protocol for semantic recovery in regenerating substrates" — and that framing is publishable at a workshop. A 4-week solo project needs a defined floor; this is it.

---

## Appendix — Suggested 4-week schedule

- **Week 1** — Implement NCA + growth/persistence; pass G1, G2. Implement the referential game and train to G3 on 2 pilot seeds. Fix the fallback ladder if G0 fails.
- **Week 2** — Eval harness: all metrics (§2), all mandatory interventions (§3), G4. Run the J = 128 pilot to fix J. Launch the 8-seed × 3-condition training overnight.
- **Week 3** — Core block (18 conditions) + sham; recovery time courses; Figures 2–6. Check G5. Start sweeps A/B/C only if the core block passes.
- **Week 4** — H4 / patching if time remains; statistics; writing. Reserve the last 4 days entirely for writing — a workshop paper is won or lost there, and the analysis is already scripted by then.

