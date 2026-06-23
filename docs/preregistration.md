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