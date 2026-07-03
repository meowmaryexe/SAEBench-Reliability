# Finding: SAEBench `absorption_fraction` is version-dependent (~10×)

**Owner:** Alor · **Metric:** Feature Absorption · **Model:** Pythia-160M-deduped L8, 4k Standard trainer_0 (CPU)

## Summary

SAEBench reports two absorption scores. On the released Pythia-160M 4k Standard SAE:

- **`mean_full_absorption_score`** reproduces the published value under **both** sae-bench versions we tried.
- **`mean_absorption_fraction_score`** reproduces the published value **only under the version that
  generated the published results** (sae-bench 0.3.2). Under current code (0.6.0) it drops **~10×**,
  because the fraction metric was *redefined* between the two releases.

This is a code-vs-published-results version drift, confirmed at the source level and against the SAEBench
PR history — not a bug in our harness (the full score, per-letter full rates, and probe true-positive
counts all line up).

## The numbers

| source | `sae_bench` / `sae_lens` | fraction | full |
|---|---|---|---|
| **published** (`adamkarvonen/sae_bench_results_0125`) | 0.3.2-era / 5.3.1 | **0.1553** | **0.0075** |
| ours @ **0.3.2** (`git @141aff72`, the release that produced the published results) | 0.3.2 / 5.3.1 | **0.1644** | 0.0095 |
| ours @ **0.6.0** (current) | 0.6.0 / 6.44.4 | **0.0163** | 0.0077 |

Under 0.3.2 both scores are within the drift band (upstream applies **no seed** — `random_seed=42` is
declared but never used — so run-to-run drift of a few % is expected; our N=3 drift on 0.6.0 was
fraction 0.0163 ± 0.0005, full 0.0077 ± 0.0008). Under 0.6.0 the fraction is ~10× low.

## Why: the fraction metric was redefined (PR #48 → PR #62)

From the SAEBench git history for `evals/absorption/`:

- **PR #48** (2025-01-13, Demian Till) — *added* `absorption_fraction` alongside Chanin's original full
  metric. No thresholds: it counts the projection of **all** non-main latents. This is what the published
  `0125` results used. PR #48's own note: *"the two metrics sometimes differ in their relative rankings of
  models' absorption scores… the new metric better handles SAEs with high L0s."*
- **PR #62** (2025-03-08, Demian Till) — *"Support for using thresholds with the absorption fraction
  metric."* Added a cosine gate, a max-absorbing-latents cap, and reused the proportion floor, "to control
  what the absorption fraction metric considers as cases of absorption." This shipped in 0.6.0.

The written README definition is **identical** across versions; only the operationalization changed.

### Constants + formulas

| | 0.3.2 (published) | 0.6.0 (current) |
|---|---|---|
| fraction cos gate | none | **0.1** |
| max absorbing latents | unbounded | **3** |
| proportion floor | on `all` non-main | **0.4** on top-3 |
| denominator | `all_feats_proj` | `absorbers + main` |
| full-absorption cos | 0.025 | 0.025 (**unchanged**) |

- **0.3.2:** `frac = (all_feats_proj − main_proj) / all_feats_proj` (0 if `main ≥ act` or `all ≤ 0`).
- **0.6.0:** keep only non-main latents with `cos ≥ 0.1` and `proj > 0` → top-3 → if their share of
  `act_proj` `< 0.4` then 0, else `min(top_total, act−main) / (top_total + main)`.

Every 0.6.0 change is restrictive (drops diffuse, weakly-aligned leakage the 0.3.2 form counted), hence
the ~10× drop. Both are ported as pure functions in
[`src/saebench_audit/metrics/absorption_fraction.py`](../../src/saebench_audit/metrics/absorption_fraction.py).

## Reproduce it

**Quick (seconds, no model/GPU/venv)** — mechanism on synthetic inputs + assertions on committed result
JSONs:
```bash
python tests/test_absorption_version_drift.py       # or: pytest -q tests/test_absorption_version_drift.py
```
Fixtures: `tests/fixtures/absorption/{published_0125,ours_v0.3.2,ours_v0.6.0}_standard_4k_t0.json`.

**From scratch (~9 min/run, CPU)** — run the real eval under each pinned venv:
```bash
# published version
python3.11 -m venv /Users/alor/saebench-0125-env/.venv
/Users/alor/saebench-0125-env/.venv/bin/pip install \
  "sae-bench @ git+https://github.com/adamkarvonen/SAEBench.git@141aff72928f7588c1451bed47c401e1d565d471" \
  "sae_lens==5.3.1" "transformers>=4.40,<5"
/Users/alor/saebench-0125-env/.venv/bin/python scripts/reproduce_absorption_drift.py \
  --workdir results/raw/absorption/standard_4k_t0_v0125          # -> fraction ~0.164, full ~0.0095

# current version
python3.11 -m venv /Users/alor/saebench-absorption-env/.venv
/Users/alor/saebench-absorption-env/.venv/bin/pip install "sae-bench" "transformers>=4.51,<5"
/Users/alor/saebench-absorption-env/.venv/bin/python scripts/reproduce_absorption_drift.py \
  --workdir results/raw/absorption/standard_4k_t0_v060           # -> fraction ~0.016, full ~0.0077
```
(The `transformers<5` pin is required: transformers 5 removed `GPTNeoXConfig.rotary_pct`, which
transformer_lens reads when loading Pythia.)

## Open question (the one that matters for the paper)

The absolute fraction number is version-dependent, but the paper's Absorption *claim* is an **architecture
ranking** (Matryoshka low, ReLU high, inverse width-scaling). PR #48 explicitly warns the two absorption
definitions "sometimes differ in their relative rankings." So the live question is: **does the architecture
ranking survive the redefinition?** Run a representative subset (Matryoshka + ReLU + a mid-pack arch, a few
sparsities) under both fraction definitions and compare the *ordering*, not the magnitude. If it holds, the
claim is robust to the definition change; if it flips, that's a headline reproducibility finding — with the
authors' own note as corroboration. Deferred to the audit stage.
