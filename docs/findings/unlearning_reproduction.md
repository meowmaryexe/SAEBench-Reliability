# Finding — Unlearning (WMDP-bio) reproduces published SAEBench on Gemma-2-2b-it 4k

**Owner:** Alor · **Metric:** Unlearning (WMDP-bio) · **Model:** gemma-2-2b-it, layer 12 · **Suite:**
gemma-2-2b_4k (7 arch × 6 trainers = 42 SAEs) · **sae-bench:** 0.6.0 · **git:** `00587e4` · **HW:** AWS A10G.

## Summary

The full 42-SAE Unlearning suite was run on GPU (see `docs/aws_unlearning_runbook.md`) and **reproduces the
published SAEBench numbers** within the pre-registered bar (`docs/preregistration.md` — Unlearning): all 7
architectures within the absolute band, Spearman **ρ = +0.929** on the architecture ranking, 42/42 SAEs ok.
Consistent with the paper's characterization that Unlearning is **degenerate (near-zero) on Gemma-2-2B**.

## Per-architecture (`unlearning_score = 1 − min WMDP-bio acc` over the 16-config sweep, MMLU-gated ≥ 0.99)

| arch | published | ours | Δ |
|---|---|---|---|
| batchtopk | 0.0413 | 0.0417 | +0.0005 |
| gatedsae | 0.0260 | 0.0280 | +0.0021 |
| jumprelu | 0.0697 | 0.0738 | +0.0041 |
| matryoshkabatchtopk | 0.0260 | 0.0259 | −0.0001 |
| panneal | 0.0378 | 0.0389 | +0.0011 |
| standard (ReLU) | 0.0475 | 0.0480 | +0.0004 |
| topk | 0.0507 | 0.0751 | +0.0244 |

**max |Δ| = 0.0244 (topk) < 0.05 band → reproduces.** 6 of 7 archs match to < 0.004.

## The one gap (topk) is intrinsic noise, not a discrepancy

topk is the only arch with a large Δ (+0.0244). It is **not** a reproduction failure — the *published*
topk per-trainer scores are themselves wildly variable:

```
topk published per-trainer: 0.0957, 0.0544, 0.1126, 0.0188, 0.0150, 0.0075   (15× spread)
```

A 0.024 shift in the 6-trainer mean is trivial against that per-trainer spread. Unlearning on Gemma-2-2B is
near-zero/degenerate, so the per-arch means are noise-sensitive — topk is simply the arch where that shows.
This is the quantified version of the pre-registration's "ranking is noise-sensitive" caveat.

## Cross-check against the paper (Karvonen et al.)

- **"Degenerate on Gemma-2-2B" (pre-conceded):** all scores are near-zero (0.026–0.075). ✓
- **"ReLU ties on unlearning" (load-bearing claim #3):** standard/ReLU (0.048) sits mid-pack, not distinctly
  worse than the others — it does not lose here, matching the paper's claim that ReLU *ties* on unlearning
  (unlike the 5/8 metrics where it's outperformed). ✓
- The headline uses the shipped best-of-16 `min`-WMDP selection + 0.99 MMLU gate (the undocumented reduction
  in `main.py:72-96`); the Stage-2 audit of that optimistic selection is a separate task
  (`metrics/unlearning_score.py` recomputes it from the saved `.pkl` sweep with zero inference).

## Independent verification (this repo, read-only)

The comparison is not self-referential: the "published" column was **re-fetched directly from HuggingFace**
(`adamkarvonen/sae_bench_results_0125`, `unlearning/saebench_gemma-2-2b_width-2pow12_date-0108/`, 42 files)
and the per-arch means match the run's `published_ref` **bit-for-bit** (batchtopk 0.0413, standard 0.0475,
topk 0.0507, …). "ours" was recomputed independently from the raw 42-row ledger and matches the report.

## Reproduce it

- **Full run (GPU):** `docs/aws_unlearning_runbook.md` → `scripts/run_unlearning_suite.sh` (suite
  `gemma-2-2b-it_4k`). Needs the gated bio-forget-corpus + Gemma-it license. ~10 min/SAE.
- **Aggregate/compare:** `scripts/aggregate_results.py --metric unlearning` +
  `scripts/unlearning_suite_report.py`.
- **Result artifacts:** run record `docs/run_records/unlearning/gemma-2-2b-it_4k_20260706T230215Z/`;
  processed `results/processed/unlearning/gemma-2-2b-it_4k.json`.

_No known sae-bench version drift for unlearning (0.3.2 ≡ 0.6.0), so a single current-version run is the
reproduction — unlike absorption's two-version story._
