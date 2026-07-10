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

### Amendment 2026-07-02 — reproduction baseline for `absorption_fraction`

Empirically, the published `absorption_fraction` (results repo `sae_bench_results_0125`) is reproduced by
the release that generated it — **sae-bench 0.3.2** (`@141aff72`): 0.164 vs published 0.155 on Standard 4k
trainer_0, within the drift band. Current code (**0.6.0**) *redefined* the fraction metric (PR #62:
cos-gate + top-3 cap + proportion floor) and yields ~10× lower values (0.016). Therefore:
- The **faithful reproduction baseline for the fraction score is the 0.3.2-era code**, not 0.6.0.
- `mean_full_absorption_score` is version-stable (reproduces under both) and needs no version pin.
- Under §4, the fraction "reproduced" check is evaluated against the 0.3.2 run.
- The shipped-vs-Table-8 note in §2 is superseded for the fraction by this shipped-vs-published-results
  finding: the 0.6.0 shipped constants do **not** reproduce the published fraction.
Evidence + mechanism: `docs/findings/absorption_version_drift.md`; regression test:
`tests/test_absorption_version_drift.py`.

---

# Pre-Registration — Unlearning (WMDP-bio) Reproduction

**Project:** SAEBench reproducibility study (Karvonen et al., 2025, arXiv:2503.09532)
**Component:** Unlearning (WMDP-bio) — a disentanglement metric measuring whether clamping SAE latents
removes harmful (biology) knowledge while preserving general capability.
**Owner:** Alor
**Date written:** 2026-07-05
**Status:** Locked before producing any Unlearning numbers (Principles I & III — pre-register the bar).

Unlearning is on the project's **"reproduce only"** list: the authors already concede it is **degenerate
on Gemma-2-2B** (near-zero, base-task-limited), so the finding is pre-conceded and you cannot meaningfully
audit a metric the base model can't support. This section fixes what we compute, how, and what
"reproduced" means.

## 1. What Unlearning is (paper definition, Appendix D / Table 9)

The eval clamps the top-N SAE latents (selected on a WMDP-bio forget set, excluding latents that fire on a
retain set) to a negative multiplier, then measures multiple-choice accuracy on **WMDP-bio** (should drop)
and on side-effect **MMLU** subsets (should stay high). It sweeps a 16-config grid
(`retain_thresholds [0.001,0.01] × n_features [10,20] × multipliers [25,50,100,200]`) and reduces to a
single headline **`unlearning_score = 1 − min(WMDP-bio accuracy)`** over the configs whose pooled
side-effect MMLU stays **≥ 0.99**. **Gemma-only:** upstream hard-requires an instruct model; there is no
Pythia unlearning (verified — the published `unlearning/` results exist only for `gemma-2-2b`).

## 2. Implementation & exact configuration

**Approach: wrap upstream.** Stage-1 faithful numbers come from running the authors' code
(`sae_bench.evals.unlearning`) end-to-end, packaged in this repo's resumable `run_unlearning.py →
aggregate_results.py --metric unlearning` flow (`src/saebench_audit/metrics/unlearning.py`), reusing the
shared wrap-upstream scaffolding. Independent reimplementation is deferred.

- **Eval model:** `gemma-2-2b-it` (instruct), layer 12. The released **base-model** SAEs are applied to the
  instruct model exactly as the published eval does. Anchor width for the green milestone: 4k
  (`adamkarvonen/…width-2pow12_date-0108`, whose naming matches the published results); then 65k headline + 16k.
- **Shipped sweep (used verbatim):** the 16-config grid above; `random_seed=42` **is applied** upstream
  (unlike absorption). `llm_batch_size`/`llm_dtype` affect only speed/memory (bfloat16, batch 4).
- **Hardcoded score reduction (the audit knob).** The **0.99** side-effect-MMLU gate and the **min**-WMDP
  selection are hardcoded in `main.py:72-96`, NOT config fields. Stage-1 uses these shipped values. They
  are re-computable from the saved per-config `.pkl` sweep with **zero inference** via
  `src/saebench_audit/metrics/unlearning_score.py` (`unlearning_score_from_sweep`, `mmlu_gate`/`reducer`
  overridable) — the Stage-2 sensitivity check for the "best-of-16" optimistic selection the project plan
  flags as the single biggest hidden choice.

| Reduction knob | Shipped (used) | Audit toggle | Consumed at |
|---|---|---|---|
| side-effect MMLU gate | **0.99** | sweep (e.g. 0.95–1.0) | `main.py:93` |
| WMDP reducer over admissible configs | **min** (best-of-16) | mean / median | `main.py:96` |
| side-effect exclusion set | **{college_biology, wmdp-bio}** | — | `main.py:72` |

**External requirements:** GPU; `google/gemma-2-2b-it` license (`huggingface-cli login`); the **gated
`bio-forget-corpus.jsonl`** (Google-form request to `cais/wmdp-corpora`, placed at the CWD-relative
`./sae_bench/evals/unlearning/data/`); a one-time ~20-min question-id generation. ~10 min/SAE. See
`docs/aws_unlearning_runbook.md`.

## 3. Validation reference

Published per-SAE `unlearning_score` values live in `adamkarvonen/sae_bench_results_0125` under the
`unlearning/` prefix (only `gemma-2-2b`, widths 4k/16k/65k, 42 SAEs/width). Fetched drop-in via
`fetch_published_unlearning.py` → `published_ref.json`. (16k/65k results use `date-0108` naming while the
registry SAE-weight repos are `canrager/…date-0107`; pass `--results_prefix` for those widths.)

## 4. Pre-registered tolerances ("reproduced" means…)

Unlearning has **no known sae-bench version drift** (the eval module is logically identical across 0.3.2 /
0.6.0), so a single current-version run is the reproduction. But the scores are **near-zero/degenerate** on
Gemma-2-2B, so absolute bands and rankings must be read with that caveat.

| Quantity | Tolerance for "reproduced" | Rationale |
|---|---|---|
| `unlearning_score` (per SAE) | \|mine − published\| ≤ **0.05** | absolute band; scores are small and near-zero |
| **Architecture ranking** | Spearman **ρ ≥ 0.9** vs published ordering | secondary — near-zero scores make the ranking noise-sensitive; report ρ but do not over-claim |

## 5. Commitments (anti-confirmation-bias)

- Report the near-zero/degenerate nature of the metric on Gemma-2-2B plainly; this is a **pre-conceded**
  property of the base model, not a finding against the SAEs.
- Use the shipped **best-of-16 min-WMDP** selection + **0.99** gate for reproduction. The optimistic-
  selection concern is a **Stage-2 audit** question (mean/median recompute from the `.pkl` sweep), not a
  reproduction change.
- Keep the per-config `.pkl` artifacts (`clean_up_artifacts=False`) so the audit recompute needs no inference.
- All code, configs, per-SAE raw JSON, and these tolerances are released.

*Locked 2026-07-05. Any later change to §1–§4 must be recorded as a dated amendment.*

---

# Pre-Registration — RAVEL (disentanglement) Reproduction

**Project:** SAEBench reproducibility study (Karvonen et al., 2025, arXiv:2503.09532)
**Component:** RAVEL — a disentanglement metric measuring whether an SAE can isolate one entity attribute
(the "cause") while leaving the others (the "iso") intact, via a trained mask over SAE latents.
**Owner:** Alor
**Date written:** 2026-07-07
**Status:** Locked before producing any RAVEL numbers (Principles I & III — pre-register the bar).

RAVEL is on the project's **"reproduce only — subset"** list: it is the single most expensive metric
(~45 min/SAE, ~41% of the per-SAE all-8 cost), so it is run once for the headline reproduction and excluded
from audit reruns (Compute-risk guardrail). This section fixes what we compute, how, and what "reproduced"
means.

## 1. What RAVEL is (paper definition, Appendix D / Table 7)

For each entity type (`city`, `nobel_prize_winner`) and each of its attributes-as-cause, a **Multi-task
Distributed Binary Mask (MDBM)** — a per-latent binary mask, SGD-trained via Adam, explicitly **not** the
linear probe of the original Huang method — is trained to make an intervention that swaps the SAE latents
change the *cause* attribute of a generated completion while preserving the *iso* attributes. The per-cause
score is **disentanglement = mean(cause_score, isolation_score)**; the headline
**`disentanglement_score`** is that averaged over attributes and entity types (`cause_score`,
`isolation_score` reported alongside). Layer 12, base `gemma-2-2b`, greedy 6-token generation.

## 2. Implementation & exact configuration

**Approach: wrap upstream.** Stage-1 faithful numbers come from running the authors' code
(`sae_bench.evals.ravel`, sae-bench 0.6.0) end-to-end, packaged in this repo's resumable
`run_ravel.py → aggregate_results.py --metric ravel` flow (`src/saebench_audit/metrics/ravel.py`). This
matches the "wrap, don't reimplement" spec; independent reimplementation is deferred to Stage 2.

**Model = base `gemma-2-2b`** (NOT the instruct model — RAVEL is a base-LM completion task, unlike
unlearning). The intervention layer is read from each SAE's own `cfg.hook_layer`. RAVEL's prompt data
(`adamkarvonen/ravel_prompts`) is **public** and auto-downloads (no gated corpus); only the model is gated.

**Shipped defaults (used verbatim for Stage 1):** `top_n_entities=500`, `num_pairs_per_attribute=5000`,
`train_test_split=0.7`, MDBM `learning_rate=1e-3`, `num_epochs=2`, temperature annealed 1.0→1e-4,
`n_generated_tokens=6` (the paper text says 8; the code ships 6 — we use the shipped 6),
`random_seed=42` (applied upstream, numpy/torch), `llm_batch_size=8` on a 24 GB card (speed/memory only,
no metric effect; upstream's un-reduced 2048 default is not used).

**Undocumented choices we adopt to match their numbers (each a Stage-2 audit item, not a reproduction
knob — reverting any of them needs a source edit + rerun, so there is no compute-free recompute):**

| Choice | Shipped behavior | Consumed at |
|---|---|---|
| Reconstruction error term | **dropped** (`add_error=False`, hardcoded, un-toggleable) | `mdbm.py:44,90-92` |
| `top_n_templates=90` filter | **dead** — defined but never applied; only entities are filtered | `eval_config.py:25`; `instance.py:425-439` |
| Entity ranking | by raw **correct count**, not accuracy rate | `instance.py:430-434` |
| Generated tokens | **6** (paper says 8) | `eval_config.py:84-88` |
| Completion matching | hardcoded special cases (UTC/coords/country aliases) | `validation.py:70-125` |
| MDAS skyline (disentangle 0.87) | not jointly trained; **not reproducible** as written | `mdas.py`; `main.py:573-575` |

## 3. Validation reference

Published per-SAE RAVEL values are in `adamkarvonen/sae_bench_results_0125` under the `ravel/` prefix
(**confirmed 2026-07-07**: base `gemma-2-2b`, widths 4k/16k/65k, 42 files each; scores at
`eval_result_metrics.ravel.{disentanglement,cause,isolation}_score` — e.g. BatchTopK 4k t0 disentanglement
= 0.4795), fetched drop-in via `fetch_published_ravel.py` → `published_ref.json`. The **4k** results dir
`ravel/saebench_gemma-2-2b_width-2pow12_date-0108/` matches the SAE-repo-derived prefix (no
`--results_prefix` needed). **16k/65k** published results use `date-0108` naming while the registry SAE
weights are `canrager/…date-0107`, so those widths need `--results_prefix ravel/saebench_gemma-2-2b_width-
2pow{14,16}_date-0108` — exactly as for unlearning.

## 4. Pre-registered tolerances ("reproduced" means…)

RAVEL has **no known sae-bench version drift**, so a single current-version run is the reproduction. The
MDBM masks are SGD-trained, so scores carry intrinsic run-to-run variance even with the seed applied.

| Quantity | Tolerance for "reproduced" | Rationale |
|---|---|---|
| `disentanglement_score` (per SAE) | \|mine − published\| ≤ **0.05** | absolute band absorbing SGD-mask variance |
| **Architecture ranking** (the real bar) | Spearman **ρ ≥ 0.9** vs published ordering, no inversion of a pair whose published gap > 0.05 | the load-bearing claim is a ranking (Matryoshka wins disentanglement) |

Qualitative claims we pre-commit to testing: (i) MatryoshkaBatchTopK leads disentanglement in the L0 40–200
range; (ii) plain ReLU/Standard is comparatively outperformed. If a value lands outside its band the first
hypothesis is our own bug (Principle V).

## 5. Commitments (anti-confirmation-bias)

- Report `disentanglement_score` **and** its cause/isolation components with equal prominence.
- Use the shipped defaults (dropped error term, dead template filter, 6 tokens) for reproduction; each is a
  **Stage-2 audit** question (re-include the error term; enforce the 90-template filter; rank entities by
  accuracy), **not** a reproduction change.
- Do not tune the MDBM hyperparameters to force a number; the SGD-mask variance is characterized, not hidden.
- All code, configs, per-SAE raw JSON, and these tolerances are released.

*Locked 2026-07-07. Any later change to §1–§4 must be recorded as a dated amendment.*

### Amendment 2026-07-10 — 4k reproduced within the pre-registered bar

The full 42-SAE `gemma-2-2b_4k` suite ran on GPU and **reproduces published SAEBench**: per-arch
`disentanglement_score` within |Δ| ≤ **0.0088** (bar 0.05) and architecture ranking Spearman **ρ = +1.000**
(bar ≥ 0.9), 42/42 ok. Both committed §4 checks pass. The published column was verified bit-for-bit against
a fresh HuggingFace re-fetch (not self-referential). The pre-committed qualitative tests (§4): (i) ReLU/
Standard is outperformed — **confirmed** (standard is the lowest arch); (ii) MatryoshkaBatchTopK leads —
**not observed at 4k** (5th of 7), but this matches the published 4k numbers and is consistent with the
paper's stated width-dependence ("advantage grows with dictionary width"), so it is deferred to the 16k/65k
scaling widths rather than recorded as a failure. Evidence: `docs/findings/ravel_reproduction.md`; run
record `docs/run_records/ravel/gemma-2-2b_4k_20260708T091349Z/`.