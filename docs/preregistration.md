# Pre-Registration — Core / Loss Recovered Reproduction

**Project:** SAEBench reproducibility study (Karvonen et al., 2025, arXiv:2503.09532)
**Component:** Core metrics — Loss Recovered (the sparsity–fidelity "proxy" anchor)
**Owner:** Ari
**Date written:** 2026-06-22
**Status:** Locked before running any evaluation (Second Look Principles I & III — pre-register the bar; no ghost-hunting).

This document fixes, *before any numbers are produced*, (a) exactly what we are computing, (b) the
numerical tolerance that counts as "reproduced," (c) the ranking-agreement criterion, and (d) what we
will report regardless of outcome. It is intentionally written so that a later reader cannot accuse us of
moving the goalposts post-hoc.

---

## 1. What "Loss Recovered" is (paper definition)

From SAEBench §3.2.1, Eq. (4):

> Loss Recovered = (H\* − H₀) / (H_orig − H₀)

where, for next-token prediction with cross-entropy loss H:

- **H_orig** = CE loss of the unmodified model.
- **H\*** = CE loss when the target activation `x` is replaced, during the forward pass, by its SAE
  reconstruction `x̂`.
- **H₀** = CE loss when the target activation `x` is replaced by **zeros** (zero-ablation).

Loss Recovered = 1.0 means the SAE reconstruction preserves the model's loss perfectly; 0.0 means it is
no better than deleting the activation entirely. This is the quantity the released SAEBench code calls
`frac_recovered`, computed identically as `(loss_reconstructed − loss_zero) / (loss_original − loss_zero)`
(see `dictionary_learning/evaluation.py::loss_recovered`, which is algebraically Eq. 4).

We also report **L0** (mean number of non-zero latents per token) as the sparsity axis, since the
load-bearing SAEBench claim is about the *sparsity–fidelity frontier*, not Loss Recovered alone.

## 2. The exact evaluation configuration (paper, Table 4 + released `CoreEvalConfig`)

Authoritative source = SAEBench Table 4 (Core metrics hyperparameters) cross-checked against the released
`sae_bench/evals/core/eval_config.py`.

| Parameter | Value | Source |
|---|---|---|
| Dataset | `Skylion007/openwebtext` | `CoreEvalConfig.dataset` |
| Context length | 128 tokens | Table 4 / `context_size` |
| Batch size (prompts) | 16 | `batch_size_prompts` |
| Loss Recovered samples | **3,200 sequences** (= 200 batches × 16) | Table 4 |
| Sparsity / variance samples | **32,000 sequences** (= 2,000 batches × 16) | Table 4 |
| Model dtype | float32 | `CoreEvalConfig.llm_dtype` |
| BOS handling | prepend BOS (transformer_lens `ActivationsStore` default) | released harness |
| CE loss reduction | mean over all predicted tokens (flattened), `ignore_index = pad_token_id` | `loss_recovered()` |

**Intervention site.** The activation `x` is the **residual-stream output of a specific layer**
(`resid_post_layer_L`), i.e. the output hidden state of transformer block `L`. Anchor models:
- Gemma-2-2B, layer 12 (`resid_post_layer_12`).
- Pythia-160M-deduped, layer 8 (`resid_post_layer_8`).

## 3. Compute reality and the two-config plan

The reproduction environment for this milestone has **no GPU** (4 CPUs, ~4 GB RAM). Gemma-2-2B with
65k-width SAEs is therefore out of scope for *execution* here (the project plan budgets ~400–600
A100-hours for the full suite). We do two things:

1. **Build the harness as an independent reimplementation** from the paper + Eq. 4, not by importing the
   authors' `evaluate()`. (Second Look Principle IV — "implement exactly as written, then check we match.")
2. **Validate it end-to-end on a CPU-feasible released SAE**: Pythia-160M-deduped, layer 8, 4k-width
   Standard (ReLU) SAE — a real released SAEBench SAE. The full Gemma-2-2B run is deferred to GPU via the
   same code path.

### Validation references (ground truth we must match)

Each released SAE ships a bundled `eval_results.json` containing SAEBench's *own* measured values for that
exact SAE. For `Standard / trainer_0` (4k, Pythia-160M, layer 8) the bundle reports
(`n_inputs=1000, context_length=1024`, training-distribution data = the Pile):

```
loss_original (H_orig)      = 2.5913
loss_reconstructed (H*)     = 2.7094
loss_zero (H0)              = 12.9791
frac_recovered              = 0.9872
l0                          = 465.59
```

Two evaluation **configs** are pre-registered, each with its own reference and tolerance:

- **Config A — "bundle-match"** (mechanical validation): ctx 1024, ~1,000 sequences, data = the Pile
  (`monology/pile-uncopyrighted`, the dictionary_learning default eval distribution). Target = the bundled
  `eval_results.json` values above.
- **Config B — "paper Table 4"** (the actual paper procedure): `Skylion007/openwebtext`, ctx 128, batch
  16. On CPU we run a reduced number of batches (documented per run); the *config* is identical to the
  paper and scales to 3,200 sequences unchanged on GPU. Reference = SAEBench's published Core results for
  this SAE where retrievable (Neuronpedia / results repo); otherwise reported as a faithful-procedure
  measurement with its sampling-noise band.

## 4. Pre-registered tolerances ("reproduced" means…)

A measured value **reproduces** the reference if it falls inside the band below. Bands account for (i)
data-sampling differences (we cannot recover the authors' exact shuffled sequences), (ii) float32
nondeterminism, and (iii) our reduced sample counts on CPU.

| Quantity | Tolerance for "reproduced" | Rationale |
|---|---|---|
| **Loss Recovered / frac_recovered** | absolute Δ ≤ **0.01** (i.e. within 1 percentage point) under the *same config* as the reference | The released SAEBench architecture gaps on this metric are large (often 5–30+ pts); 1 pt is well inside-noise and below any claim-relevant gap. |
| **H_orig (model CE)** | absolute Δ ≤ **0.05** nats under same data/ctx | Model+data property; should be highly reproducible. |
| **H₀ (zero-ablation CE)** | relative Δ ≤ **5%** | Larger because zero-ablation CE has higher variance and is data-dependent. |
| **L0** | relative Δ ≤ **5%** under same data/ctx | Sparsity is data-dependent but stable. |

If a value lands outside its band, the **first hypothesis is our own bug** (Principle V). We will isolate
the cause (data distribution, BOS handling, layer indexing, dtype) and document the investigation before
drawing any conclusion about the paper.

## 5. Ranking-agreement criterion (for the eventual 7-architecture reproduction)

The load-bearing Loss-Recovered claim is a *ranking / frontier* claim, not a single number. When we
extend to all 7 architectures × 6 sparsities (Stage-1 reproduction on GPU), we pre-commit to:

- **Frontier reproduction:** for each architecture, our (L0, Loss Recovered) points must lie on the same
  sparsity–fidelity frontier as the paper's Figure 2 within the per-point tolerance in §4.
- **Ranking agreement:** at matched L0 bins, the *ordering* of architectures by Loss Recovered must match
  the paper's, measured by **Spearman rank correlation ρ ≥ 0.9** against the published values, AND no
  pair whose published gap exceeds 0.01 may invert.
- We explicitly pre-register the paper's qualitative Loss-Recovered findings we are testing:
  (i) the sparsity–fidelity frontier does **not** reliably predict downstream/disentanglement performance
  (Claim 1); (ii) Loss Recovered increases with dictionary width (Claim re: scaling); (iii) higher L0
  yields better Loss Recovered.

## 6. Commitments (anti-confirmation-bias)

- We report **confirming and null results with equal prominence**. "Everything reproduced within
  tolerance" is the expected and fully acceptable outcome (Principle I/II — no ghost-hunting).
- We do **not** tune our harness to hit the reference number. The tolerance bands above are fixed now.
- Loss Recovered is, per the project plan, **"reproduce only" (deterministic by design, large clean
  gaps)** — there is nothing to "audit" here beyond verifying the PCA / residual-stream baselines are not
  artificially weak (deferred to the baseline-audit task).
- All code, configs, seeds, raw per-run JSON, and these tolerances are released for independent re-running.

---

*Locked 2026-06-22. Any later change to §1–§5 must be recorded as a dated amendment below, with reason.*

## Amendments
*(none yet)*

---

# Pre-Registration — SCR / TPP / Sparse Probing Reliability Audit

**Project:** SAEBench reproducibility study (Karvonen et al., 2025; Chanin et al., 2026)
**Component:** SCR, TPP, and Sparse Probing
**Owner:** Mary
**Date written:** 2026-06-23
**Status:** Pre-reproduction infrastructure validated; audit protocol to be finalized before seed-sweep experiments.

## Scope

This audit targets the three benchmark metrics identified as the highest-priority reliability concerns:

- TPP
- SCR
- Sparse Probing

These metrics were selected because they contain stochastic components including probe training, dataset sampling, feature selection, and randomized optimization procedures.

## Current Status

Completed before preregistration:

- SCR acceptance test reproduced on CUDA.
- TPP acceptance test reproduced on CUDA.
- Sparse Probing acceptance test reproduced on CUDA.
- Pythia-160M loading-path investigation completed.
- Dictionary-learning SAE loading path validated.
- Initial Pythia-160M TPP benchmark execution completed.

## Audit Questions

1. How sensitive are metric values to random seed variation?
2. Do architecture rankings remain stable across repeated runs?
3. Are some sparsity regimes more sensitive to randomness than others?
4. Do SCR and TPP capture distinct information or largely the same signal?
5. Are reported architecture gaps substantially larger than run-to-run variance?

## Planned Experiments

### Stage 1: Faithful Reproduction

- Reproduce released benchmark results on Pythia-160M.
- Reproduce released benchmark results on Gemma-2-2B.
- Validate outputs against published benchmark artifacts where available.

### Stage 2: Reliability Audit

- Single-SAE seed sweeps.
- Multi-SAE seed sweeps.
- Architecture-ranking stability analysis.
- SCR/TPP correlation analysis.
- Variance decomposition where feasible.

## Reporting Commitments

- Report both confirming and non-confirming results.
- Preserve all raw outputs.
- Record all deviations from the released benchmark configuration.
- Distinguish reproduction findings from reliability-audit findings.

---

# Pre-Registration — Feature Absorption Reproduction

**Project:** SAEBench reproducibility study (Karvonen et al., 2025, arXiv:2503.09532)
**Component:** Feature Absorption (first-letter) — a concept-detection / disentanglement metric,
load-bearing for the Matryoshka claim.
**Owner:** Alor
**Date written:** 2026-07-01
**Status:** Locked before producing any Absorption numbers (Principles I & III — pre-register the bar).

Absorption is on the project's **"reproduce only"** list: the paper shows 30–40% architecture gaps that
dwarf run-to-run noise, so the metric is robustly discriminative and its low audit yield is deferred to
Stage 2. This section fixes what we compute, how we obtain it, and what "reproduced" means.

## 1. What Absorption is (paper definition, Appendix D / Table 8)

For each first letter, a ground-truth logistic-regression probe is trained on the model's residual
stream over an ICL spelling task ("{word} has the first letter:"). k-sparse probing identifies the SAE's
"main"/split latents for that letter. **Absorption** occurs on probe true-positive tokens where the main
latents do **not** fire yet a probe-aligned latent carries the concept. Two headline scores (both
reported): **`mean_absorption_fraction_score`** (fraction of the probe projection carried by absorbing
latents, averaged over letters) and **`mean_full_absorption_score`** (rate of single-latent full
absorption). We also record `mean_num_split_features` and all three `std_dev_*` fields.

## 2. Implementation & exact configuration

**Approach: wrap upstream.** Stage-1 faithful numbers come from running the authors' code
(`sae_bench.evals.absorption`, sae-bench 0.6.0) end-to-end, packaged in this repo's resumable
`run_absorption.py → aggregate_results.py --metric absorption` flow
(`src/saebench_audit/metrics/absorption.py`). This matches the Probe-Rig "wrap, don't reimplement"
design spec for probe-family metrics; independent reimplementation is deferred to Stage 2.

**Shipped constants (used verbatim for Stage 1).** The code hardcodes four thresholds as module
constants (`feature_absorption.py:34-44`) that **differ from Table 8**. To reproduce the paper's numbers
we use the **shipped** values, exposing them as config for the audit toggle:

| Threshold | Shipped (used) | Table 8 (audit) | Consumed at |
|---|---|---|---|
| absorption-fraction cosine gate (τ_ps) | **0.1** | −1 | `feature_absorption_calculator.py:181` |
| full-absorption cosine gate | **0.025** | — | `:108-109` |
| projection-proportion gate (τ_pa) | **0.4** | 0 | `:119`, `:206` |
| max-absorbing-latents (A_max) | **3** | dict size | `:190` |

Other faithful defaults: `f1_jump_threshold=0.03`, `max_k_value=10`, GT-probe filter `min_GT_probe_f1=0.6`,
`min_feats_for_eval=20`, 80/20 split, GT probe = torch-BCE multi-probe (Adam, 50 epochs). Anchor:
Pythia-160M-deduped L8, 4k Standard trainer_0 on CPU (green pipeline) → all 7 architectures × 6
sparsities. `llm_batch_size` affects only speed/memory, not the metric.

**Seed is inert (a genuine reliability gap).** `random_seed=42` is declared but never applied upstream
(no `random.seed`/`manual_seed`; `probing.py:276` uses `random.sample`, shuffled DataLoaders/ICL), so
results **drift run-to-run**. We characterize this drift explicitly (§4) rather than pretend determinism.

## 3. Validation reference

The released per-SAE `eval_results.json` bundle does **not** carry absorption; the published Pythia-160M
absorption values live in the results repo `adamkarvonen/sae_bench_results_0125` (the same source the Core
full-suite comparison used). Comparison is drop-in via a `published_ref.json` placed in the run workdir.

## 4. Pre-registered, drift-aware tolerances ("reproduced" means…)

Because no seed is applied, **exact matching is impossible**. We first measure our own run-to-run drift by
re-running one SAE (Standard 4k trainer_0) N≥3 times; call its per-score standard deviation σ_drift.

| Quantity | Tolerance for "reproduced" | Rationale |
|---|---|---|
| `mean_absorption_fraction_score` / `mean_full_absorption_score` (per SAE) | \|mine − published\| ≤ **max(0.05, 2·σ_drift)** | absolute band that absorbs the un-seeded drift; well below the 30–40% architecture gaps |
| **Architecture ranking** (the real bar) | Spearman **ρ ≥ 0.9** vs published ordering on each score, no inversion of a pair whose published gap > 0.05 | the load-bearing claim is a ranking, and the gaps are large |

Qualitative findings we pre-commit to testing: (i) Matryoshka has low absorption (strong); (ii) plain
ReLU/Standard is comparatively high; (iii) non-hierarchical architectures worsen (inverse-scale) with
width. If a value lands outside its band the first hypothesis is our own bug (Principle V).

## 5. Commitments (anti-confirmation-bias)

- Report **both** headline scores and the observed run-to-run drift with equal prominence.
- Do **not** tune the harness or lower the `min_GT_probe_f1` / `min_feats_for_eval` guards to force a
  number. If a real released SAE trips the min-features guard, that is itself a documented finding.
- We use the shipped (not Table 8) thresholds for reproduction; the Table-8 vs shipped discrepancy is a
  **Stage-2 audit** question, not a reproduction change.
- All code, configs, per-SAE raw JSON, and these tolerances are released.

*Locked 2026-07-01. Any later change to §1–§4 must be recorded as a dated amendment.*