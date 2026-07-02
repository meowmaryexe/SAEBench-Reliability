# Metric notes

Per-metric methodology, status, and findings. See `preregistration.md` for locked tolerances and
`logs/` for the dated work record.

---

## Core / Loss Recovered  ✅ (reproduce-only) — owner: Ari

**Definition (SAEBench Eq. 4).** `Loss Recovered = (H* − H0) / (H_orig − H0)`, where H_orig is the
model's next-token cross-entropy, H* is CE with `resid_post(layer L)` replaced by the SAE
reconstruction, and H0 is CE with that activation zero-ablated. Identical to the released code's
`frac_recovered`. We also report **L0** (mean non-zero latents/token).

**Implementation.** `src/saebench_audit/metrics/core_loss_recovered.py` — independent reimplementation
using HuggingFace `transformers` for the model and `saebench_audit.sae_models` for the SAE forward.
Imports nothing from SAEBench / dictionary_learning; consumes only released SAE weights. The runner is
checkpoint-resumable (one JSONL line per batch) to survive strict per-process wall-clock limits.

**Two tokenization paths (a real code-vs-paper distinction).** The metric is computed two different ways
inside the authors' own codebase, and they are **only comparable in Loss Recovered, not absolute CE**:

| path | tokenization | produced | config variant |
|---|---|---|---|
| `dictionary_learning/evaluation.py::loss_recovered` | **per-document**, truncate to ctx, dynamic pad | the bundled `eval_results.json` (ctx 1024, the Pile) | `bundle_exact` |
| SAEBench `core/main.py` (transformer_lens ActivationsStore) | **packed** concatenated tokens | paper Table 4 / Neuronpedia (ctx 128, OpenWebText) | `paper` |

Packing inserts document-boundary predictions → inflates absolute CE; the **ratio** (Loss Recovered) is
invariant. This is worth a sentence in the write-up and a question to the authors.

**Full-suite reproduction (4k Pythia-160M L8, all 7 architectures × 6 sparsities = 42 SAEs)** — exact
per-document path, 12 docs, shared baseline (`scripts/run_suite.py`):

| Metric | Result |
|---|---|
| max \|Δ Loss Recovered\| vs released | **0.0059** (mean 0.0013) — all within the 0.01 band ✅ |
| L0 relative error | max **3.8%**, mean 1.8% — all within 5% ✅ |
| mine vs released Loss Recovered | **Pearson r = 0.9984, Spearman ρ = 0.9966** ✅ |

Per-architecture max \|ΔLR\|: Standard 0.0010 · TopK 0.0038 · BatchTopK 0.0059 · JumpRelu 0.0038 ·
GatedSAE 0.0019 · Matryoshka 0.0042 · PAnneal 0.0036. TopK-family L0 reproduces the exact k
(20/40/80/160/320/640). The sparsity–fidelity frontier and architecture ranking reproduce (figures
`core_lr_frontier_4k.svg`, `core_lr_reproduction_4k.svg`). Full detail: `logs/2026-06-22_07`.

**Exact single-point anchor (Standard 4k trainer_0, 128 docs):** Loss Recovered 0.9866 vs 0.9872 (0.1%),
L0 463.97 vs 465.59 (0.3%), H0 12.470 vs 12.979 (3.9%); H_orig 2.656 vs 2.591 — the small H_orig/H* shift
is a uniform data-order offset that cancels out of the ratio. Under the **paper** procedure (OpenWebText
ctx128, packed) Loss Recovered = 0.9789.

**Loaders:** all 7 architecture forwards verified against released values. Matryoshka required a distinct
loader (`W_enc`/`W_dec` + threshold, unlike BatchTopK's Linear layout). See `sae_models.py`.

**Methodology equivalence — proven identical to `core/main.py` (no discrepancy).** There are two SAEBench
eval paths: `dictionary_learning::loss_recovered` (→ bundled `eval_results.json`, what the suite above was
validated against) and `sae_bench/evals/core/main.py` (→ paper Fig. 2 / Neuronpedia, the canonical Core
eval). We audited `core/main.py` line-by-line and implemented it exactly in
`saebench_core_eval` (packed BOS tokens, `sae.decode(sae.encode)`, zero-ablate full resid_post, per-token
CE excluding `{bos,eos,pad}` via `mask[:,:-1]`, `(ce_abl−ce_sae)/(ce_abl−ce_orig)`, L0 `(acts!=0)`
excluding special tokens). An **oracle test** (`tests/test_core_oracle.py`) runs SAEBench's *verbatim*
code on the real `transformer_lens` model and matches ours to **Δ ≤ 2.4e-5 Loss Recovered for all 7
architectures** (Standard 4.1e-7). 8 unit tests pin each primitive. Full detail: `logs/2026-06-22_08`.
Note: the 42-SAE suite numbers above used the `dictionary_learning` reference; the canonical `core/main.py`
methodology is reproduced exactly (oracle) and is ready to run at suite scale via `saebench_core_eval`.

**Configs:** `configs/reproduce/core.yaml` (+ `configs/registry.yaml`).
**Results:** `results/raw/core_loss_recovered/*.jsonl` (per batch) → `results/processed/core_loss_recovered/*.json`.
**Figures:** `figures/core_lr_mine_vs_bundle.svg`, `figures/core_lr_convergence.svg`.
**Full history:** `logs/2026-06-22_01..05`.

**Done:** all 7 architectures × 6 sparsities at **4k** on Pythia-160M (the frontier).

**Full Core metric set + Neuronpedia comparison + baselines (2026-06-22, log #10).** Implemented every
Core metric (`metrics/core_full.py`), oracle-verified against SAEBench's verbatim code (~1e-7). Re-ran all
42 4k SAEs under `saebench_core` and compared **every metric** to the published values
(`adamkarvonen/sae_bench_results_0125`): 11/15 within <1% (Pearson ≈1.0), weight metrics **exact**, density
metrics rank-correct (sampling-limited magnitude). PCA + residual-stream baselines reproduce the paper
exactly (Loss Recovered 1.0, L0 = d_model = 768). Subtleties handled: published `explained_variance` =
legacy formula; `mse` batch-size-normalized (batch 16); `l1` needs decoder normalization (added);
TopK L0 bounded at k (sampling); KL not computed in published run. Figure
`figures/core_full_metrics_vs_neuronpedia.svg`; data `results/processed/.../full_metrics_vs_neuronpedia.json`,
`baselines_pca_residual.json`.

**Next (GPU only):** widths 16k & 65k; Gemma-2-2B layer 12 — `configs/gpu/` is set up; the density metrics
and TopK L0 reach exact agreement at the full 3200/32000-sequence scale.

---

## AutoInterp (Automated Interpretability)  ✅ — owner: Ari

**Definition (paper Table 5 / Paulo et al. 2024 detection score).** An LLM judge (gpt-4o-mini) writes a
natural-language explanation of an SAE latent from its top-activating sequences, then — given the
explanation plus a shuffled mix of activating + random sequences — predicts which activate. Score =
**detection accuracy** over the test set (2 top + 2 importance-weighted + 10 random = 14 sequences);
`autointerp_score` = mean over ~1000 non-dead latents.

**Implementation.** `src/saebench_audit/metrics/autointerp.py` — faithful port of
`autointerp/main.py` + the indexing/activation/dataset utils (verbatim prompts, top-k/IW/random example
construction, `<<token>>` marking, detection-accuracy scoring). Runner `scripts/run_autointerp.py` caches
the SAE-independent residual activations once and runs the gpt-4o-mini judge resumably. **8/8 unit tests**
(`tests/test_autointerp_units.py`).

**Result (gpt-4o-mini, Pythia-160M L8, Standard 4k t0):** the score **converges to published** as the
activation-token budget grows — 24k tokens → 0.710 (at the 0.714 null floor), 96k → **0.748**, paper 2M →
**0.780** (published). The remaining gap is token budget + latent-sample noise + judge stochasticity, not
methodology (an LLM-judge metric is noisy by design — the paper itself flags this). Figure
`figures/autointerp_convergence.svg`; data `results/processed/autointerp/`; log #11.

**Next (GPU/scale):** 2M tokens / 1000 latents across the suites (`configs/gpu/autointerp_gpu.yaml`).

---

## Absorption (first-letter feature absorption) 🚧 (reproduce-only) — owner: Alor

**Definition (paper App. D / Table 8).** For each first letter, a ground-truth LR probe is trained on
the residual stream over an ICL spelling task; k-sparse probing finds the SAE's main/split latents.
**Absorption** is when, on a probe true-positive token, the main latents don't fire but a probe-aligned
latent carries the concept. Two headline scores reported: `mean_absorption_fraction_score` and
`mean_full_absorption_score` (+ `mean_num_split_features` and the three `std_dev_*`).

**Implementation — wrap upstream.** `src/saebench_audit/metrics/absorption.py` is a thin wrapper around
`sae_bench.evals.absorption` (sae-bench 0.6.0): we run the authors' code end-to-end inside this repo's
resumable per-SAE `run_absorption.py → aggregate_results.py --metric absorption` flow, exposing the four
hardcoded thresholds as config for the Stage-2 audit toggle. (Matches the Probe-Rig "wrap, don't
reimplement" spec; independent reimplementation deferred.) **Runs under a dedicated pinned venv**
(`/Users/alor/saebench-absorption-env/.venv`): `sae_bench` + `sae_lens` + `transformer_lens 2.16.1`
with **`transformers` pinned `<5`** — transformers 5 removed `GPTNeoXConfig.rotary_pct`, which
transformer_lens 2.16.1 still reads when loading Pythia (Mary's shared venv has transformers 5.x and
currently cannot load Pythia via transformer_lens). Kept isolated so the shared venv is untouched.

**Two code-vs-paper faithfulness facts (we adopt the shipped code to match numbers):**
1. **Shipped thresholds ≠ Table 8.** Cosine gate **0.1** (fraction) / **0.025** (full), projection-
   proportion **0.4**, max-absorbing-latents **3** — hardcoded module constants
   (`feature_absorption.py:34-44`), vs Table 8's τ_ps=−1, τ_pa=0, A_max=dict. Stage-1 uses shipped;
   toggling to Table 8 is a Stage-2 audit action (forces a rerun — the raw df is cached by
   (layer, sae_name) only).
2. **`random_seed=42` is declared but never applied** → run-to-run drift. We report both scores + their
   std devs and characterize drift explicitly; the reproduction bar is drift-aware (see preregistration).

**SAE loading.** Released dictionary_learning SAEs (`ae.pt`+`config.json`) are loaded into sae_lens-
compatible objects via upstream `TRAINER_LOADERS` and handed to `run_eval` as `[(name, sae)]`.

**Status.** Wrapper + resumable runner + aggregation + 8/8 unit tests
(`tests/test_absorption_units.py`) landed. Green pipeline on Pythia-160M 4k Standard trainer_0 (CPU) →
then 42-SAE suite + published comparison. Feasibility watch: the `min_feats_for_eval=20` /
`min_GT_probe_f1=0.6` guard may trip on small-model SAEs (documented as a finding if it does — the paper
does report Pythia-160M absorption). **Configs:** `configs/reproduce/absorption.yaml`. **Pre-reg:**
`docs/preregistration.md` (Absorption).

---

## SCR · TPP · Sparse Probing 🚧 (owner: Mary)

### Current Status

The benchmark execution paths for all three audit-priority metrics have been validated.

Completed:

- Reproduced the official SAEBench CUDA acceptance tests for:
  - TPP
  - SCR
  - Sparse Probing
- Established a reproducible Google Colab CUDA workflow for benchmark execution.
- Archived smoke-test artifacts under `results/raw/smoke_tests/`.
- Investigated the Pythia-160M loading-path issue and identified the distinction between SAE Lens registry loading and the released dictionary-learning SAE repositories.
- Enumerated all 42 released Pythia-160M SAEs from the benchmark repository.
- Completed the first successful Pythia-160M TPP benchmark run using the released dictionary-learning SAE suite.
- Documented execution paths, loading behavior, and benchmark outputs in the corresponding run logs.

### Reliability Motivation

These metrics are the primary targets of the reliability audit because they contain stochastic components that may introduce benchmark variance, including:

- Probe training
- Dataset sampling
- Negative-class sampling
- Feature-selection procedures
- Random minibatch ordering

Code inspection identified multiple sources of randomness within the released implementations, particularly in probe optimization and dataset construction.

### Current Questions

1. Which benchmark components are actually controlled by the published random-seed configuration?
2. How sensitive are metric values to seed variation?
3. Do architecture rankings remain stable across repeated runs?
4. Is variance concentrated at particular sparsity levels?
5. How strongly correlated are SCR and TPP across architectures and sparsities?

### Next Steps

#### Faithful Reproduction

- Reproduce TPP on released Pythia-160M SAEs.
- Reproduce SCR on released Pythia-160M SAEs.
- Reproduce Sparse Probing on released Pythia-160M SAEs.
- Extend reproduction to Gemma-2-2B benchmark suites.

#### Reliability Audit

- Single-SAE seed sweeps.
- Multi-SAE seed sweeps.
- Architecture-ranking stability analysis.
- SCR/TPP correlation analysis.
- Additional variance-source investigations as needed.

### Documentation

- `docs/scr_tpp_run_log.md`
- `docs/sparse_probing_run_log.md`
- `docs/pythia160m_loading_notes.md`