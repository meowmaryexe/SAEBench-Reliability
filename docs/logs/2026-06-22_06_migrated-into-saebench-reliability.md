# Log 2026-06-22 #06 — Migrated Core / Loss Recovered into SAEBench-Reliability

The Core / Loss Recovered work (built in the scratch repo `SAEReproducibility`) was moved into the
canonical `SAEBench-Reliability` repo and **refactored to conform to its scaffold** (the
`saebench_audit` package with `io` / `schema` / `statistics` / `plotting` + `metrics/<name>.py`,
`configs/{reproduce,audit}`, `results/{raw,processed}`). No results changed — the relocated runner
reproduces the identical per-batch numbers (verified: bi=0 Horig=1.953, frac=0.9940, …).

## Mapping onto the scaffold

| Concern | Home in SAEBench-Reliability |
|---|---|
| Config schema (`CoreConfig`) | `src/saebench_audit/schema.py` |
| Model/SAE/data loading, tokenization windows, JSONL checkpoints | `src/saebench_audit/io.py` |
| Aggregation + bundle comparison | `src/saebench_audit/statistics.py` |
| Independent SAE forwards | `src/saebench_audit/sae_models.py` |
| SVG figure builders | `src/saebench_audit/plotting.py` |
| The metric (intervention, CE, resumable runner) | `src/saebench_audit/metrics/core_loss_recovered.py` |
| Run / aggregate / plot entry points | `scripts/run_metric.py`, `aggregate_results.py`, `make_figures.py` |
| Configs | `configs/reproduce/core.yaml`, `configs/registry.yaml` |
| Pre-registration + methodology + logs | `docs/preregistration.md`, `docs/metric_notes.md`, `docs/logs/` |
| Per-batch raw / aggregated results | `results/raw/core_loss_recovered/`, `results/processed/core_loss_recovered/` |
| Figures | `figures/core_lr_*.svg` |

## Consolidation of the two earlier runners

`eval_resumable.py` (packed, paper/bundle configs) and `eval_bundle_exact.py` (per-document, exact) were
merged into one mode-aware, resumable `metrics/core_loss_recovered.run_core()`, selected by
`CoreConfig.tokenize_mode`. `process_batch()` uses dynamic padding + an attention mask, so packed
(all-ones mask) and per-document (ragged) share one code path. Three CLI variants in
`scripts/run_metric.py`: `paper`, `bundle_match`, `bundle_exact`.

## Pipeline verified end-to-end

`run_metric.py` (resumable → `results/raw/.../raw.jsonl` + `run_meta.json`) →
`aggregate_results.py` (→ `results/processed/.../*.json` with `comparison_to_bundle`) →
`make_figures.py` (→ `figures/*.svg`). Package + all three scripts import cleanly; `pyproject.toml`
populated (`name = saebench-audit`, `packages = src`).

## Note on the scratch repo

Source content was **copied** (this mount blocks deletion without explicit permission, and
`SAEReproducibility` is a separate git repo). It still exists as a redundant scratch copy; deleting it is
a destructive whole-repo action left for explicit confirmation.
