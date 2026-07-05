# AWS runbook — SAEBench Unlearning (WMDP-bio) on Gemma-2-2b-it

Reproduce the Unlearning suite (42 SAEs/width, gemma-2-2b-it, layer 12) on an AWS GPU box. This mirrors
`docs/aws_absorption_runbook.md` for the generic AWS bits (launch, spot, budget alarm, auto-shutdown,
S3/git auto-save) — read that for the cloud mechanics. This file covers the **unlearning-specific**
prerequisites, which are the whole reason it's more involved than absorption.

## Why unlearning is heavier than absorption
- **GPU required** (Gemma-2-2b-it forward passes; no CPU/Pythia path — unlearning is Gemma-only).
- **Gemma license** — `google/gemma-2-2b-it` is HF-gated.
- **Gated forget corpus** — `bio-forget-corpus.jsonl` needs a Google-form request; must be placed manually.
- **One-time question-id generation** (~20 min) on first run for a model, then cached and reused per SAE.
- ~10 min/SAE → ~7 GPU-h per width (~$7–14 spot for one width).

## Prerequisites (do these before launching the suite)

1. **Gemma license.** On the box: `huggingface-cli login` (a token with access), then accept the license at
   huggingface.co/google/gemma-2-2b-it. Verify: `huggingface-cli download google/gemma-2-2b-it config.json`.

2. **Gated bio-forget-corpus.** Request access via the `cais/wmdp-corpora` Google form (linked from the
   upstream `sae_bench/evals/unlearning/README.md`). You receive `bio-forget-corpus.jsonl`. Place it where
   upstream reads it — a **CWD-relative** path — i.e. **launch the runner from the repo root** and put the
   file at:
   ```
   <repo>/sae_bench/evals/unlearning/data/bio-forget-corpus.jsonl
   ```
   The runner (`scripts/run_unlearning.py`) preflights this and fails fast with an actionable message if
   it's missing (`unlearning.require_forget_corpus`). If your `sae_bench` is only installed in the venv
   (not vendored in the repo), create the dir under the repo root and drop the file there, or symlink it;
   the upstream path is resolved relative to the process CWD, so run from the repo root either way.

3. **Venv.** Reuse the dedicated sae-bench venv (transformers<5), same recipe as absorption's
   `saebench-absorption-env` — see `docs/aws_absorption_runbook.md`. Unlearning needs **one** version
   (no version drift), so a single `VENV` is enough.

## Run

From the **repo root**, on the GPU box, inside the venv:

```bash
# one suite at a time; 4k first (cheapest; SAE repo matches published-results naming)
VENV=/path/to/.venv/bin/python \
DEVICE=cuda LLM_DTYPE=bfloat16 AUTO_SHUTDOWN=1 \
bash scripts/run_unlearning_suite.sh
```

The driver is resumable (`until … grep ALL_SAES_DONE`), fetches published values, aggregates, writes a
report + durable run record under `docs/run_records/unlearning/`, and (with `AUTO_SHUTDOWN=1`) stops the
box on success. Set `S3_DEST=s3://…` and/or `GIT_PUSH=1` to auto-save results off the box (recommended, so
a late return never loses data). To add 16k/65k, uncomment them in the `SUITES=(...)` line — those widths
need `fetch_published_unlearning.py --results_prefix unlearning/saebench_gemma-2-2b_width-2pow{14,16}
_date-0108` because their published results use `date-0108` naming while the SAE weights are canrager
`date-0107` (the driver logs a warning if the plain fetch misses).

Single-SAE smoke first (recommended before the full 42):
```bash
$VENV scripts/run_unlearning.py --suite gemma-2-2b-it_4k --device cuda \
  --sae_location Standard_gemma-2-2b__0108/resid_post_layer_12/trainer_0 \
  --workdir results/raw/unlearning/gemma-2-2b-it_4k
```

## After completion
- Grab the run record from `docs/run_records/unlearning/<suite>_<TS>/` (scp or S3/git).
- The per-config `.pkl` sweep is kept under `results/raw/unlearning/<suite>/artifacts/` — needed for the
  Stage-2 audit recompute (`metrics/unlearning_score.py`), no re-inference.
- Terminate (or stop to reuse for the next width).

## Cost + guardrails
Same as the absorption runbook: spot instance, `--instance-initiated-shutdown-behavior stop`, AWS Budgets
alarm, `AUTO_SHUTDOWN=1`, dead-man's-switch `sudo shutdown`. One width ≈ ~7 GPU-h (~$7–14 spot); all three
widths ≈ ~21 GPU-h. The one-time question-id generation (~20 min) is amortized across the 42 SAEs.
