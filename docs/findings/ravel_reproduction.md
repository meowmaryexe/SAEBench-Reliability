# Finding — RAVEL (disentanglement) reproduces published SAEBench on Gemma-2-2b 4k

**Owner:** Alor · **Metric:** RAVEL (disentanglement) · **Model:** base gemma-2-2b, layer 12 · **Suite:**
gemma-2-2b_4k (7 arch × 6 trainers = 42 SAEs) · **sae-bench:** 0.6.0 · **git:** `7a1c4af` · **HW:** AWS A10G.

## Summary

The full 42-SAE RAVEL suite was run on GPU (see `docs/aws_ravel_runbook.md`) and **reproduces the published
SAEBench numbers** within the pre-registered bar (`docs/preregistration.md` — RAVEL): all 7 architectures
within the absolute band, Spearman **ρ = +1.000** on the architecture ranking (a *perfect* rank match),
42/42 SAEs ok. This is the **tightest** of the three metrics in this owner's set (cf. absorption, unlearning)
— see "Why so clean" below.

## Per-architecture — `disentanglement_score` (primary; = mean of cause + isolation)

| arch | published | ours | Δ |
|---|---|---|---|
| gatedsae | 0.6834 | 0.6775 | −0.0060 |
| panneal | 0.6543 | 0.6475 | −0.0068 |
| jumprelu | 0.6484 | 0.6420 | −0.0063 |
| batchtopk | 0.6311 | 0.6373 | +0.0062 |
| matryoshkabatchtopk | 0.6288 | 0.6299 | +0.0010 |
| topk | 0.6186 | 0.6213 | +0.0028 |
| standard (ReLU) | 0.5802 | 0.5714 | −0.0088 |

**max |Δ| = 0.0088 (standard) < 0.05 band → reproduces.** ρ = +1.000 (identical ordering). The two
component scores agree just as tightly: **max |Δ cause| = 0.0221**, **max |Δ isolation| = 0.0192**.

## Cross-check against the paper (Karvonen et al.)

- **"ReLU baseline is outperformed" (load-bearing claim #3, RAVEL is one of the 5/8): ✓ reproduces.**
  Standard/ReLU is the **lowest** architecture on RAVEL disentanglement (0.571 ours / 0.580 published), by a
  clear margin over the next arch (topk 0.621). ReLU loses on RAVEL, as claimed.

- **"MatryoshkaBatchTopK is best on RAVEL" (load-bearing claim #2): not observed at 4k — and this is
  consistent with the paper, not a reproduction failure.** At 4k, averaged over the six L0 trainers,
  MatryoshkaBatchTopK sits **5th of 7** (0.630), with gatedsae leading (0.678). Crucially our ordering
  **matches the published 4k numbers exactly** (ρ = +1.000) — the published 4k results *also* do not put
  Matryoshka on top. The paper's claim is explicitly scoped to the **L0 40–200 window** and states the
  Matryoshka disentanglement advantage **"grows with dictionary width"** (§ claim #2, #4). 4k is the
  *smallest* width, and our per-arch means average over the full L0 sweep (including high-L0 trainers), so
  Matryoshka not leading here is exactly what "advantage grows with width" predicts. **Verifying the
  Matryoshka-on-RAVEL claim therefore requires the 16k/65k scaling widths** (deferred) — we flag it as a
  width-specific claim to test there, not as a contradiction of the benchmark. (Second-Look principle: no
  ghost-hunting; report the ranking that reproduced, and scope the unverified claim honestly.)

- **cause vs isolation decomposition.** Across every architecture, `isolation_score` (0.64–0.72) is
  uniformly higher than `cause_score` (0.50–0.64): the trained MDBM mask preserves the *other* attributes
  better than it flips the *target* one. This is a structural property of the metric (isolation is the
  easier half), stable across archs, and reproduces in both columns.

## Why it reproduces so cleanly (vs unlearning's topk noise)

RAVEL disentanglement scores are large (0.57–0.68) and the MDBM masks are SGD-trained under an applied
`random_seed=42`, so run-to-run variance is small — hence per-arch agreement to ±0.009 and a perfect rank
correlation. This contrasts with unlearning, whose near-zero/degenerate scores on Gemma made per-arch means
noise-sensitive. RAVEL is a genuinely discriminative, low-noise metric at this width.

## Independent verification (this repo, read-only)

The comparison is not self-referential:
- The **"published" column was re-fetched directly from HuggingFace** (`adamkarvonen/sae_bench_results_0125`,
  `ravel/saebench_gemma-2-2b_width-2pow12_date-0108/`, 42 files) and the per-arch means match the run's
  `published_ref.json` **bit-for-bit** (max diff 0.0 over 7 archs × 3 scores: disentanglement 0.6311 /
  0.6834 / 0.5802 / …, cause, isolation).
- **"ours" was recomputed independently** from the raw 42-row ledger (`ravel.jsonl`) via
  `saebench_audit.statistics.aggregate_ravel` and matches the processed JSON's `by_arch` exactly; ρ and
  max |Δ| were recomputed with `statistics.spearman`.

## Reproduce it

- **Full run (GPU):** `docs/aws_ravel_runbook.md` → `scripts/run_ravel_suite.sh` (suite `gemma-2-2b_4k`).
  Base gemma-2-2b (license only; RAVEL prompt data is public). ~45 min/SAE.
- **Aggregate/compare:** `scripts/aggregate_results.py --metric ravel` + `scripts/ravel_suite_report.py`.
- **Result artifacts:** run record `docs/run_records/ravel/gemma-2-2b_4k_20260708T091349Z/`; processed
  `results/processed/ravel/gemma-2-2b_4k.json`.

_No known sae-bench version drift for RAVEL (single current-version run is the reproduction). 16k/65k
scaling widths — where the Matryoshka-on-RAVEL claim would be verified — remain to run._
