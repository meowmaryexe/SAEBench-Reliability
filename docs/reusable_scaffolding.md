# Reusable scaffolding for wrap-upstream metrics (Absorption · Unlearning · RAVEL)

The probe-family metrics **wrap** upstream SAEBench code (we run the authors' eval, we don't reimplement
it) and share one shape: enumerate every released SAE in a suite, run the upstream eval per SAE, aggregate
by architecture, compare to the published numbers, and emit a durable run record — all resumable and
attributable. That shape now lives in shared modules under `src/saebench_audit/`, so a new metric is a thin
adapter, not a copy of the whole pipeline. (Ari's Core/AutoInterp predate this and keep their own runners;
they can adopt the shared base later.)

**Three adapters exist today** and are the worked examples to copy: **Absorption**
(`metrics/absorption.py` + `scripts/*absorption*`, two-version drift story), **Unlearning**
(`metrics/unlearning.py` + `scripts/*unlearning*`, single scalar score, single-version, GPU/Gemma-only with
a gated corpus + a pure-python score-reduction oracle in `metrics/unlearning_score.py`), and **RAVEL**
(`metrics/ravel.py` + `scripts/*ravel*`, three scores per SAE, single-version, GPU, base `gemma-2-2b`, no
gated data — code done, GPU run deferred). Between them they show the single-score, multi-score,
single-version, and multi-version shapes, plus the base-model vs instruct-model split.

## What's shared (reuse verbatim)

| Module | What you get |
|---|---|
| `provenance.py` | `ensure_torch_load_shim()` (torch≥2.6 vs sae-bench 0.3.2), `installed_version(pkg)`, `environment_provenance(device, packages)`, and `build_upstream_config(cls, wanted, extra_attrs)` — the **version-safe** config builder (passes only fields the installed upstream dataclass has, so one runner works across pinned versions). |
| `suites.py` | `resolve_suite(name)` (registry-driven model/layer/repo), `discover_locations(repo, layer)`, `arch_from_location`, `sae_name`, `load_released_sae(...)` (dictionary_learning → sae_lens, incl. the Matryoshka typo workaround), `sae_result_path`. |
| `runner.py` | `run_sae_suite(...)` — the resumable per-SAE loop: JSONL ledger, skip-if-done, resume-from-upstream-output, wall-clock timebox, **per-SAE error isolation**. Plus `write_run_meta(...)` and `announce(...)` (the `ALL_SAES_DONE` / `PROGRESS n/N` sentinel the suite shell greps). |
| `statistics.py` | `summary`, `ranks`, `spearman`, `aggregate_by_arch(rows, score_keys, per_sae_keys, ...)`, `compare_by_arch_to_reference(agg, reference, score_keys)`. |
| `run_record.py` | `write_run_record(...)` — the durable `run_record.{json,md}` writer, generic over the metric's headline score columns. |

## What each new metric supplies (the thin adapter)

Mirror `metrics/absorption.py` + `scripts/run_absorption.py` (and the `absorption_*` scripts):

1. **`metrics/<metric>.py`** — a `@dataclass <Metric>Config` (repo-side, mirrors the upstream eval config +
   exposes any hardcoded knobs); a `build_eval_config` that fills a `wanted` dict and calls
   `build_upstream_config`; a `_flatten_output(out)` pulling the headline scores from the upstream result;
   and a `run_<metric>(cfg, sae, name, workdir, ...)` single-SAE wrapper. Import + re-export the shared
   provenance/suite helpers so `<metric>.<name>` stays the one import surface.
2. **`scripts/run_<metric>.py`** — build the config from argparse, then hand three closures to
   `runner.run_sae_suite`: `name_fn` (= `suites.sae_name`), `process_one(location, name)` (load + eval),
   and optionally `resume_row_fn` (record an on-disk upstream result without reloading the model). Finish
   with `runner.write_run_meta(...)` + `runner.announce(...)`.
3. **`statistics.py`** — a two-line `aggregate_<metric>` / `compare_<metric>_to_published` wrapping
   `aggregate_by_arch` / `compare_by_arch_to_reference` with the metric's score keys; add a `--metric`
   branch in `scripts/aggregate_results.py`.
4. **`scripts/fetch_published_<metric>.py`** — copy the absorption fetcher; change the results-repo prefix
   and the metric's score-key names. **The published-results layout differs per metric — verify the keys.**
5. **`scripts/<metric>_run_record.py`** — a thin CLI over `write_run_record` supplying `score_keys` +
   version labels (see `absorption_run_record.py`).
6. **`scripts/run_<metric>_suite.sh`** — copy `run_absorption_suite.sh`; it's already generic (resumable
   `until … grep ALL_SAES_DONE`, both-versions loop, aggregate → report → run-record, S3/git auto-save,
   auto-shutdown). Only the script names + `--metric` change.

## The test bar

Follow the house convention (see `tests/README.md`): the shared scaffolding is pinned by
`tests/test_scaffolding_units.py` (torch-free). For the metric itself add `tests/test_<metric>_units.py`
(config defaults, flatten, aggregation — plain interpreter) and, since we wrap upstream rather than
reimplement, a version/behavior fixture test rather than a live verbatim oracle (mirror
`tests/test_absorption_version_drift.py`). Keep all `sae_bench`/`torch` imports lazy so the fast tests run
without a venv.
