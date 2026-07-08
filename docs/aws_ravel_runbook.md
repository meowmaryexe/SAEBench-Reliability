# AWS runbook — SAEBench RAVEL (disentanglement) on base Gemma-2-2b

Reproduce the RAVEL suite (42 SAEs/width, base `gemma-2-2b`, layer 12) on an AWS GPU box. This mirrors
`docs/aws_absorption_runbook.md` for the generic AWS bits (launch, budget alarm, auto-shutdown, S3/git
auto-save) — read that for the cloud mechanics. This file covers the **RAVEL-specific** setup, which is
notably **simpler** than unlearning (no gated corpus) but **more expensive** (it's the priciest metric).

## Use an ON-DEMAND box
RAVEL is the suite's most expensive metric (~45 min/SAE per the plan doc's Appendix A timing table, ~41% of
the all-8 per-SAE cost). A single 42-SAE width is a long, uninterrupted run — use an **on-demand** g5.xlarge
(A10G, 24 GB), not spot, to avoid mid-run reclaims (spot thrashed the unlearning run). Cost per width ≈ ~32
GPU-h ≈ **~$33–38** on-demand (@ ~$1.006/hr; treat ±50% as the real band). 65k is OOM-prone on 24 GB —
consider `--llm_batch_size 4` or a g5.2xlarge.

## What RAVEL needs (vs unlearning)
- **GPU required** (gemma-2-2b MDBM forward+backward passes; no CPU path).
- **Gemma license** — `google/gemma-2-2b` (the **BASE** model, not `-it`) is HF-gated.
- **NO gated corpus.** RAVEL's prompts (`adamkarvonen/ravel_prompts`) are a **public** HF dataset that
  auto-downloads to `artifacts/ravel/base/` on first use. Nothing to request or place manually.
- **One-time per-model dataset generation** on the first SAE: the model generates completions over all
  (template × entity) prompts to find the correctly-answered subset, then caches the filtered dataset under
  `artifact_dir` (`artifacts/ravel`), reused across every SAE, resume, and width. Budget ~1 GPU-h for it.

## Prerequisites (before launching the suite)

1. **Gemma license.** On the box: `huggingface-cli login` (a token with access), then accept the license at
   huggingface.co/google/gemma-2-2b. Verify: `huggingface-cli download google/gemma-2-2b config.json`.

2. **Venv.** Reuse the dedicated sae-bench venv (transformers<5) — the same `saebench-absorption-env` used
   for absorption/unlearning (RAVEL lives in the same `sae_bench` package). One version is enough (no drift).

3. **Launch from the repo root** so the `artifacts/ravel` dataset cache is created there and persists across
   SAEs/resumes (it is gitignored under `artifacts/`).

## Run

From the **repo root**, on the on-demand GPU box, inside the venv:

```bash
# 4k green milestone first
VENV=/path/to/.venv/bin/python \
DEVICE=cuda LLM_DTYPE=bfloat16 LLM_BATCH=8 AUTO_SHUTDOWN=1 \
bash scripts/run_ravel_suite.sh
```

The driver is resumable (`until … grep ALL_SAES_DONE`), fetches published values, aggregates, writes a
report + durable run record under `docs/run_records/ravel/`, and (with `AUTO_SHUTDOWN=1`) stops the box on
success. Set `S3_DEST=s3://…` and/or `GIT_PUSH=1` to auto-save results off the box (recommended). To add
16k/65k, uncomment them in the `SUITES=(...)` line — no code change (they're just registry suite keys); for
65k drop `LLM_BATCH=4` if it OOMs.

**Single-SAE smoke first** (strongly recommended before the full 42 — it also triggers the one-time dataset
gen so you can measure real per-SAE wall-clock):
```bash
$VENV scripts/run_ravel.py --suite gemma-2-2b_4k --device cuda \
  --sae_location Standard_gemma-2-2b__0108/resid_post_layer_12/trainer_0 \
  --workdir results/raw/ravel/gemma-2-2b_4k
```

## Published comparison — VERIFY
`fetch_published_ravel.py` derives the results prefix `ravel/<sae_repo_basename>/` and reads each score
under `eval_result_metrics.ravel.{disentanglement,cause,isolation}_score`. **Confirm this layout exists** in
`adamkarvonen/sae_bench_results_0125` (the driver logs a warning if the fetch misses). If RAVEL results use
a different width naming, pass `--results_prefix`; if they aren't published there at all, fall back to
Neuronpedia / the paper's Table 7 as the comparison source and note it in the run record.

## After completion
- Grab the run record from `docs/run_records/ravel/<suite>_<TS>/` (scp or S3/git).
- Keep `artifacts/ravel` if you plan to run 16k/65k next — the filtered dataset cache is model-keyed and
  reused across widths.
- Terminate (or stop to reuse for the next width).

## Cost + guardrails
On-demand g5.xlarge, `--instance-initiated-shutdown-behavior stop`, AWS Budgets alarm, `AUTO_SHUTDOWN=1`,
dead-man's-switch `sudo shutdown`. One width ≈ ~32 GPU-h (~$33–38 on-demand); the one-time dataset gen (~1
GPU-h) is amortized across the 42 SAEs. All three widths ≈ ~95 GPU-h (~$100–130).

## Stage-2 audit (deferred — not part of this Stage-1 reproduction)
RAVEL's undocumented choices need source edits + reruns (no compute-free recompute): the dropped
reconstruction-error term (`mdbm.py` `add_error=False`, un-toggleable), the dead `top_n_templates=90` filter
(defined but never applied), and entity ranking by raw correct count vs accuracy rate. See
`configs/gpu/ravel_gpu.yaml` and the RAVEL preregistration section.
