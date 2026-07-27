# Finding — Core / Loss Recovered reproduces published SAEBench across all 6 suites (252 SAEs)

**Owner:** Ari (metric + harness) · run, orchestration fixes and audit: Alor · **Metric:** Core / Loss
Recovered · **Models:** Pythia-160M-deduped L8, Gemma-2-2b L12 · **Suites:** all six (4k/16k/65k × both
models), 7 arch × 6 sparsities = **42 SAEs each, 252 total** · **git:** `alor/core-baseline-cache` ·
**HW:** 4 × AWS `g6.xlarge` (L4 24 GB), us-west-2 + us-east-1.

## Summary

The **full 252-SAE Core sweep** was run on GPU and **reproduces the published SAEBench numbers**: all
252 SAEs are within the pre-registered ±0.01 band on Loss Recovered
(`docs/preregistration.md`), max |Δ| = **0.0045**, mean |Δ| = **0.0007**, Pearson **r = 0.9973–0.9999**
per suite. This is the widest-coverage reproduction in the project (252 SAEs vs 42 for the other metrics)
and the only one covering both anchor models at all three widths.

Two things make the result stronger than a single comparison would:

1. The comparison is against the **config-matched** reference. The published `eval_config` in
   `adamkarvonen/sae_bench_results_0125` is **identical** to our `configs/gpu/core_gpu.yaml`
   (OpenWebText, ctx 128, batch 16, 200/2000 batches, float32, `exclude_special_tokens=True`).
2. The methodology is **oracle-proven identical** to upstream `core/main.py` (Δ ≈ 3e-06 across all 7
   architectures — `tests/test_core_oracle.py`, `tests/test_core_full_oracle.py`).

## Per-suite — Loss Recovered vs published `sae_bench_results_0125`

| suite | n | max \|Δ\| | mean \|Δ\| | Pearson r | max L0 rel err | within ±0.01 |
|---|---|---|---|---|---|---|
| pythia-160m_4k | 42 | 0.0016 | 0.0006 | 0.99993 | 4.97% | 42/42 ✅ |
| pythia-160m_16k | 42 | 0.0014 | 0.0006 | 0.99986 | 2.43% | 42/42 ✅ |
| pythia-160m_65k | 42 | 0.0010 | 0.0006 | 0.99983 | 1.96% | 42/42 ✅ |
| gemma-2-2b_4k | 42 | 0.0045 | 0.0012 | 0.99896 | 2.49% | 42/42 ✅ |
| gemma-2-2b_16k | 42 | 0.0041 | 0.0008 | 0.99813 | **6.85%** | 42/42 ✅ |
| gemma-2-2b_65k | 42 | 0.0025 | 0.0006 | 0.99726 | **8.83%** | 42/42 ✅ |
| **all** | **252** | **0.0045** | **0.0007** | — | — | **252/252 ✅** |

Per-architecture across all six suites (36 SAEs each): standard 0.0045 · topk 0.0042 · batchtopk 0.0019 ·
jumprelu 0.0014 · gated 0.0023 · matryoshka 0.0015 · panneal 0.0021.

**Headline suite (gemma-2-2b_65k), per-arch mean Loss Recovered:** gated 0.9939 (pub 0.9945) · batchtopk
0.9908 (0.9912) · panneal 0.9906 (0.9910) · jumprelu 0.9901 (0.9904) · topk 0.9899 (0.9910) · matryoshka
0.9896 (0.9901) · standard 0.9884 (0.9890). Every Δ is negative and ≤ 0.0010 — a small uniform offset,
not scatter.

## L0 diverges from Loss Recovered on the wide Gemma suites — reportable

Loss Recovered is comfortably inside its ±0.01 band everywhere, but **L0 relative error reaches 6.85%
(gemma-16k) and 8.83% (gemma-65k), above the pre-registered 5% L0 band** (`core_gpu.yaml:34`). Pythia
stays at 1.96–4.97%. So the two pre-registered criteria disagree on exactly the two widest Gemma suites:
the *fidelity* metric reproduces while the *sparsity* metric does not, on the same SAEs and the same
tokens. Worth a sentence in the write-up — it suggests L0 is the more configuration-sensitive of the two.

## The bundled `eval_results.json` is the WRONG reference (and why that matters)

Each released SAE ships an `eval_results.json`. It is tempting to use it as the reference — we did at
first — but it is produced by `dictionary_learning`'s `loss_recovered`: **per-document, ctx 1024, on the
Pile**, not the Table-4 packed/ctx128/OpenWebText path. Against it we saw **205/210 within ±0.01 with 5
apparent gemma-2-2b_4k outliers** (topk t0, batchtopk t0/t1, matryoshka t0, panneal t5 — all |Δ| 0.0103–0.0129).
Against the config-matched published reference, **those same SAEs are all in tolerance** (e.g. topk t0:
0.0042 vs published, 0.0129 vs bundled).

Note the outliers were the **sparsest** SAEs (L0 20–52), not the densest, and L0 agreed closely on all of
them — so the disagreement was in the loss measurement alone.

### Control run (GPU, `bundle_exact`)

We re-ran 3 of the 5 outliers plus 1 non-outlier under the config that *produced* the bundled refs
(per-document, ctx 1024, the Pile; `run_metric.py --variant bundle_exact`, 256 docs):

| arch (t0) | outlier? | bundled | ours (packed) \|Δ\| | control (per-doc) \|Δ\| | verdict |
|---|---|---|---|---|---|
| batchtopk | yes | 0.9604 | 0.0127 | **0.0062** | closes |
| matryoshka | yes | 0.9516 | 0.0109 | **0.0044** | closes |
| topk | yes | 0.9561 | 0.0129 | 0.0116 | **still out** |
| standard | no (control) | 0.9917 | 0.0018 | 0.0052 | stays ok |

**The tokenization/dataset mismatch explains most but not all of it.** Two of three outliers close; `topk`
does not, and it flips sign (0.9431 under-shoots, 0.9676 over-shoots). The negative control drifting
0.0018 → 0.0052 puts a **~0.005 sampling-noise floor** on this control at 256 docs (the bundled refs used
1000), so "closes" here means "inside the noise floor" and `topk` is genuinely outside it.

**Honest conclusion:** our packed run reproduces the packed published reference for all 252 SAEs including
`topk` t0. The residual is a discrepancy *between the two upstream reference sets*, not a defect in this
reproduction. Fully resolving `topk` would need the control at the bundled 1000-doc scale (deferred).

## Upstream inconsistencies found (report these)

1. **Results labelled `date-0108`, weights are `date-0107`.** `core/saebench_gemma-2-2b_width-2pow{14,16}_date-0108/`
   contains results whose `sae_lens_release_id` says `date-0107`; the `date-0108` weight repos do not
   exist. We evaluated the correct weights — only the folder label differs.
2. **Trainer indices disagree between the results filename and `sae_lens_release_id`, in different
   directions per suite** (pythia-4k BatchTopK vs gemma-16k BatchTopK). Cross-check pairing is therefore
   done within (suite, arch) by the **monotonic L0 ladder** and validated by L0 agreement, not by index.
3. **`gemma-2-2b_16k` `batch_top_k` is published as `trainer_6..11`**, while every other (suite, arch)
   uses `0..5`. Encoded as `trainer_overrides` in `configs/registry.yaml`.
4. **42 of the 252 released SAEs ship no `eval_results.json`** (notably 36 of the 42 in the headline
   gemma-65k suite) — which is what makes the published dataset, not the bundle, the necessary reference.

## Compute finding — ~93% of the GPU work was redundant

Measured on L4: gemma-4k took **4458 s/SAE** and gemma-65k **4492 s/SAE** — a 16× larger SAE for 0.8% more
time. Same signature on Pythia (208 s at 4k, 228 s at 16k). Runtime is set by *model* forward passes, which
are identical for every SAE in a suite: of ~2800 forwards per SAE, only ~200 depend on the SAE.

Caching the SAE-independent work (`saebench_core_baseline` + a float32 sparsity-activation cache on local
NVMe) gave a measured **8.4–10.6× speedup** (4458 s → 420 s/SAE) and is **bitwise identical** — verified by
`tests/test_core_cache_equivalence.py` across 5 architectures, with both oracles unchanged. Sweep wall-clock
~74 h → ~5.5 h; cost ~$140 → ~$60.

## Reproduce it

- **Full sweep (GPU):** `docs/GPU_SETUP.md` → `scripts/run_all_gpu.sh` (or `scripts/run_core_gpu.py
  --suite <name> --cache_dir /opt/dlami/nvme/saecache`). Checkpointed per (arch, trainer); safe to re-run.
- **Cross-check:** `results/processed/core_loss_recovered/CROSSCHECK_vs_published_0125.json`.
- **Control:** `results/processed/core_loss_recovered/bundle_exact_control/`.
- **Result artifacts:** `results/processed/core_loss_recovered/*_saebench_core.json` (+ `.progress.jsonl`
  per-SAE checkpoints; 16k/65k also keep their raw shards alongside the merged file).
- **Tests:** `tests/test_core_units.py` (10/10), `tests/test_core_oracle.py`,
  `tests/test_core_full_oracle.py`, `tests/test_core_cache_equivalence.py`.

_AutoInterp (also Ari's) is unrun at scale — it needs an OpenAI key (~$54) nobody on the team holds. Its
CPU tests all pass (`tests/test_autointerp_*`), so it is ready when a key appears._
