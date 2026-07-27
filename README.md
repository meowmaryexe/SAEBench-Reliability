## SAEBench Reliability 

Reproducibility and reliability study of Sparse Autoencoder (SAE) evaluation benchmarks.

## Overview

This project aims to reproduce key results from:

* Karvonen et al., SAEBench (2025)
* Chanin et al., Are Sparse Autoencoder Benchmarks Reliable? (2026)

Our current focus is:

1. Reproducing core SAEBench evaluation results on released SAEs.
2. Building a clean and fully reproducible evaluation pipeline.
3. Auditing the reliability and stability of selected SAE benchmark metrics.
4. Investigating the robustness and generalizability of benchmark conclusions across models and evaluation settings.

The project is being conducted as a reproducibility study targeting submission to TMLR and consideration for the NeurIPS 2026 Machine Learning Reproducibility Challenge (MLRC).

## Repository Structure
- `configs/` - Experiment configurations (`reproduce/`, `audit/`) + `registry.yaml`
- `docs/` - Project notes, pre-registration, per-metric notes, dated logs
- `figures/` - Generated figures and visualizations (SVG, via `scripts/make_figures.py`)
- `results/` - Raw per-batch (`raw/`) and aggregated (`processed/`) outputs, by metric
- `scripts/` - Entry-point scripts (`run_metric.py`, `aggregate_results.py`, `make_figures.py`)
- `src/` - Core source: the `saebench_audit` package (`io`, `schema`, `statistics`, `plotting`, `sae_models`, `metrics/`)

## Status (2026-07-26)

| Metric (owner) | Status |
|---|---|
| **Core / Loss Recovered** (Ari) | ✅ **Fully reproduced — all 6 suites, 252 SAEs** (Pythia-160M L8 + Gemma-2-2b L12 × 4k/16k/65k, 7 arch × 6 sparsities). **252/252 within the pre-registered ±0.01** on Loss Recovered vs the published `sae_bench_results_0125`, max \|Δ\| **0.0045**, Pearson **r = 0.9973–0.9999**. Methodology **oracle-proven identical** to `core/main.py` (Δ≈3e-06, all 7 archs). Key findings: the per-SAE bundled `eval_results.json` is the **wrong reference** (per-doc/ctx1024/Pile vs Table-4 packed/ctx128/OWT) and manufactures 5 false outliers; **L0 rel-err 6.85%/8.83% on Gemma 16k/65k exceeds the 5% band while Loss Recovered passes**; ~93% of GPU work was redundant (**8.4–10.6× speedup**, bitwise identical). Run on 4×L4, ~5.5 h, ~$60. See branch **`alor/core-baseline-cache`** → [`docs/findings/core_reproduction.md`](https://github.com/meowmaryexe/SAEBench-Reliability/blob/alor/core-baseline-cache/docs/findings/core_reproduction.md), [log #12](https://github.com/meowmaryexe/SAEBench-Reliability/blob/alor/core-baseline-cache/docs/logs/2026-07-26_12_core-full-sweep-252-saes.md), [`results/processed/core_loss_recovered/`](https://github.com/meowmaryexe/SAEBench-Reliability/tree/alor/core-baseline-cache/results/processed/core_loss_recovered). |
| **AutoInterp** (Ari) | ⏸️ **Validated, not run at scale.** Pipeline reproduced on Pythia-160M 4k with the paper's **gpt-4o-mini** judge (8/8 unit tests, verbatim prompts, deterministic oracle passes); score converges to published with token budget: 24k→0.710 (null floor 0.714), 96k→0.748, paper 2M→0.780. The full 252-SAE sweep is **blocked on an OpenAI key (~$54)** that nobody on the team currently holds — all CPU tests pass, so it is ready the moment one exists. See `docs/metric_notes.md`, log #11, `figures/autointerp_convergence.svg`. |
| **TPP / SCR / Sparse Probing** (Mary) | 🚧 Evaluation paths validated. Official SAEBench acceptance tests reproduced successfully on CUDA for TPP, SCR, and Sparse Probing. Smoke-test artifacts archived. Investigated the Pythia-160M loading-path issue, verified the dictionary-learning SAE loading path, enumerated all 42 released Pythia-160M SAEs, and completed the first successful Pythia-160M TPP benchmark run. Faithful reproduction runs and reliability audits are in progress. |
| **Absorption** (Alor) | ✅ **Reproduced on Pythia-160M 4k** (42 SAEs, reproduce-only), wrapping upstream `sae_bench.evals.absorption`. Key finding — a **code-vs-published version drift**: `mean_full_absorption_score` reproduces under **both** sae-bench versions, but `mean_absorption_fraction_score` reproduces **only under 0.3.2** (0.164 vs published 0.155, within the drift-aware bar) and collapses ~10× under 0.6.0 (the fraction metric was redefined in PR #62) — not a harness bug. _Full code, findings & run record are on branch `alor/ravel-abs-unlearning` (not yet merged to main)._ |
| **RAVEL** (Alor) | ✅ **Reproduced on Gemma-2-2b 4k** (42 SAEs, base model, reproduce-only): all 7 archs within the ±0.05 band, **Spearman ρ = +1.000** (perfect rank match), disentanglement max\|Δ\| = 0.0088 — the tightest of the three. Published re-fetched from HF **bit-for-bit** (independent). ReLU outperformed on RAVEL ✓; Matryoshka "best on RAVEL" **not** seen at 4k but *matches published* — a width-scaling claim to verify at 16k/65k. _Full code, findings & run record on branch `alor/ravel-abs-unlearning`._ |
| **Unlearning** (Alor) | ✅ **Reproduced on Gemma-2-2b-it 4k** (42 SAEs, WMDP-bio, reproduce-only): all 7 archs within the ±0.05 band, **Spearman ρ = +0.929**, HF-verified bit-for-bit. Scores near-zero (0.026–0.075) = the paper's pre-conceded **"degenerate on Gemma-2-2B"**; ReLU ties mid-pack ✓. _Full code, findings & run record on branch `alor/ravel-abs-unlearning`._ |

### Where the work lives (branches)

`main` carries the shared scaffolding and this status table; each owner's full work sits on their branch.

| branch | contents |
|---|---|
| [`alor/core-baseline-cache`](https://github.com/meowmaryexe/SAEBench-Reliability/tree/alor/core-baseline-cache) | **Core**: the 252-SAE results, the cross-check vs published, the per-suite caching (~10× speedup, bitwise identical), the `bundle_exact` control, and Ari's harness fixes folded in |
| [`alor/gpu-orchestration-fixes`](https://github.com/meowmaryexe/SAEBench-Reliability/tree/alor/gpu-orchestration-fixes) | the four harness fixes on their own, one commit each, for review in isolation: per-SAE checkpointing, per-suite trainer indices, `curl` error handling, sweep failure reporting (+ SAE-path preflight) |
| [`ari/core-loss-recovered`](https://github.com/meowmaryexe/SAEBench-Reliability/tree/ari/core-loss-recovered) | Ari's original Core + AutoInterp implementation (the base both branches build on) |
| [`alor/ravel-abs-unlearning`](https://github.com/meowmaryexe/SAEBench-Reliability/tree/alor/ravel-abs-unlearning) | Absorption, RAVEL, Unlearning |
| [`mary-scr-tpp`](https://github.com/meowmaryexe/SAEBench-Reliability/tree/mary-scr-tpp) | TPP / SCR / Sparse Probing |

### Run the Core metric (CPU; resumable — re-invoke until `ALL_BATCHES_DONE`)

```bash
python scripts/run_metric.py --metric core --variant bundle_exact \
  --local_model <PYTHIA_DIR> --local_sae_dir <SAE_DIR> --arch standard --layer 8 \
  --n_seqs 128 --batch 2 --workdir results/raw/core_loss_recovered/standard_4k_t0_bundle_exact
python scripts/aggregate_results.py \
  --workdir results/raw/core_loss_recovered/standard_4k_t0_bundle_exact \
  --out results/processed/core_loss_recovered/standard_4k_t0_bundle_exact.json
python scripts/make_figures.py
```

On GPU, the `paper` variant scales to the full Table-4 counts (`--n_seqs 3200 --batch 16`); switch the
model to `google/gemma-2-2b --layer 12` for the headline reproduction. Models/SAEs: `configs/registry.yaml`.

# References 
Karvonen et al. (2025). SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability.

Chanin et al. (2026). Are Sparse Autoencoder Benchmarks Reliable?
