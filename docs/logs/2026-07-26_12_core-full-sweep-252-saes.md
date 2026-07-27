# Log 12 — Core full sweep: 252 SAEs, 6 suites, both anchor models (2026-07-26)

Continues Ari's Core work (logs 07–10). Ari built and CPU-validated the metric and the GPU harness but
never ran the sweep. This log covers the audit of that harness, the fixes it needed, the sweep itself, and
the cross-check.

## 1. Pre-flight audit of the harness (no GPU)

`tests/test_core_units.py` passed 10/10 — but 3 of the 10 tests **silently no-op'd**: their bodies skip
when `_fixtures/arch_saes` is absent, and the runner counts a skip as a pass. That left 6 of 7
architectures with **zero loader coverage**. After fetching one SAE per architecture: still 10/10, now
genuinely. Both oracles pass (Δ = 3.04e-06; full metric set 1e-7–1e-6), including a per-architecture run
confirming `tests/README.md`'s "Δ ≤ 2.4e-5 across all 7 architectures" (max 2.32e-05, BatchTopK).

Four orchestration defects found and fixed on `alor/gpu-orchestration-fixes`:

1. **No checkpointing.** `run_core_gpu.py` wrote `--out` once at the end; a crash lost the whole suite
   (~11 h for gemma-65k). `docs/GPU_SETUP.md` claimed resumability — true of the AutoInterp runner, never
   of this one. Now a per-`(arch, trainer)` JSONL sidecar + atomic `--out` rewrite.
2. **`gemma-2-2b_16k` `batch_top_k` is published as `trainer_6..11`**, not `0..5` — a guaranteed crash ~12
   SAEs in. Now `trainer_overrides` in `configs/registry.yaml`.
3. **`curl` without `-f`.** A 404 wrote `Entry not found` (15 bytes) into `ae.pt`, which then died
   unpickling; the poisoned file was never re-fetched, so re-runs failed identically.
4. **`run_all_gpu.sh` never checked exit status**, so a crashed suite printed the DONE banner and exited 0.

Plus a preflight that HEADs every SAE URL before any GPU work (all 252 verified present, ~21 s), and a
fix to `make_figures.py`, which crashed on a fresh clone *and* read filenames the sweep never writes.

## 2. The sweep

4 × `g6.xlarge` (L4 24 GB), split across us-west-2 and us-east-1 because the G-family vCPU quota is
per-region and 8 vCPU each. `g6.xlarge` (4 vCPU) rather than `g6.2xlarge` (8 vCPU): same L4, so the same
quota buys **two** GPUs instead of one, at a lower $/GPU-hour.

**fp32 Gemma fits on a 24 GB L4** (17.2–20.4 GB observed) — so the `bfloat16` fallback `GPU_SETUP.md` §10
suggests was never needed. Just as well: that path currently **crashes**, because three call sites
hardcode `.to(torch.float32)` against what would be bf16 SAE weights.

**The compute finding.** First timings showed gemma-4k at 4458 s/SAE and gemma-65k at 4492 s/SAE — 16×
the SAE for 0.8% the time. Runtime is dominated by *model* forwards, which are identical across SAEs:
~2600 of ~2800 per SAE are SAE-independent. Caching them (`saebench_core_baseline` + float32 sparsity
cache on local NVMe) gave a measured **8.4–10.6×** and is **bitwise identical**
(`tests/test_core_cache_equivalence.py`, both oracles unchanged). Projected 74 h → actual ~5.5 h;
~$140 → ~$60.

One failure worth recording: box A's Gemma shard 401'd because work was rebalanced onto a box that had
never been given the HF token. The auto-stop watcher caught it and stopped the box rather than let it
idle; nothing was lost, because shutdown-behaviour was *stop* and the checkpoints persist on EBS.

## 3. Cross-check

`adamkarvonen/sae_bench_results_0125`'s `eval_config` is **identical** to `configs/gpu/core_gpu.yaml`, so
it — not the per-SAE bundled `eval_results.json` — is the right reference. **252/252 within ±0.01**,
max |Δ| 0.0045. Against the bundled refs instead: 205/210 with 5 apparent outliers, which a GPU control
run showed to be *mostly* a reference-config artefact (2 of 3 close; `topk` does not).

See `docs/findings/core_reproduction.md` for the full result, the L0-vs-Loss-Recovered divergence on the
wide Gemma suites, and the upstream labelling/indexing inconsistencies.

## 4. State

- **Core: done.** 252/252, committed under `results/processed/core_loss_recovered/`.
- **AutoInterp: unrun at scale** — needs an OpenAI key (~$54). All CPU tests pass; ready when one exists.
- Branches: `alor/gpu-orchestration-fixes` (harness fixes) → `alor/core-baseline-cache` (caching + results).
