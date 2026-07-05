#!/usr/bin/env bash
# Unlearning suite driver — runs the full 42-SAE unlearning suite under the current sae-bench version,
# aggregates, compares to published, and prints the report. Resumable (safe to re-run / after a spot
# interruption). GPU + Gemma-it license + gated bio-forget-corpus required. See docs/aws_unlearning_runbook.md.
#
#   VENV=/path/.venv AUTO_SHUTDOWN=1 bash scripts/run_unlearning_suite.sh
#
# PREREQS (fail fast without these):
#   - `huggingface-cli login` + accept the google/gemma-2-2b-it license.
#   - Gated forget corpus at ./sae_bench/evals/unlearning/data/bio-forget-corpus.jsonl (relative to CWD;
#     request via the cais/wmdp-corpora Google form). MUST launch this script from the repo root.
#   - Single sae-bench version suffices — unlearning has no known version drift (unlike absorption).
#
# Env:
#   VENV       python of the sae-bench venv (GPU-capable)                 [required]
#   DEVICE     cuda|cpu            (default: cuda)
#   LLM_DTYPE  bfloat16|float32    (default: bfloat16)
#   AUTO_SHUTDOWN=1   `sudo shutdown -h now` on success  (THE cost safeguard — off by default)
set -euo pipefail
cd "$(dirname "$0")/.."

VENV="${VENV:?set VENV to the sae-bench venv python (GPU)}"
DEVICE="${DEVICE:-cuda}"
LLM_DTYPE="${LLM_DTYPE:-bfloat16}"

# ---- Suites to run. Start with 4k (cheapest, repo matches published naming); add 16k/65k when ready. ----
SUITES=( "gemma-2-2b-it_4k" )
# SUITES=( "gemma-2-2b-it_4k" "gemma-2-2b-it_16k" "gemma-2-2b-it_65k" )

run_one () {  # $1=suite → echoes the processed JSON path on its last line
  local suite="$1"
  local raw="results/raw/unlearning/${suite}"
  local proc="results/processed/unlearning/${suite}.json"
  mkdir -p "$raw" results/processed/unlearning
  echo ">>> $suite : running (resumable) under $VENV" 1>&2
  until "$VENV" scripts/run_unlearning.py --suite "$suite" --device "$DEVICE" \
        --llm_dtype "$LLM_DTYPE" --workdir "$raw" 2>&1 | tee -a "${raw}.log" | grep -q "ALL_SAES_DONE"; do
    echo ">>> $suite : progress checkpoint, continuing…" 1>&2
  done
  # 16k/65k need an explicit --results_prefix (date-0108) — see fetch script header.
  "$VENV" scripts/fetch_published_unlearning.py --suite "$suite" --workdir "$raw" 1>&2 || \
    echo "!! published fetch failed (16k/65k may need --results_prefix)" 1>&2
  "$VENV" scripts/aggregate_results.py --metric unlearning --workdir "$raw" --out "$proc" 1>&2
  echo "$proc"
}

TS="$(date -u +%Y%m%dT%H%M%SZ)"
for suite in "${SUITES[@]}"; do
  PROC="$(run_one "$suite" | tail -1)"

  REC="docs/run_records/unlearning/${suite}_${TS}"
  mkdir -p "$REC"
  echo "=================  REPORT: $suite  ================="
  "$VENV" scripts/unlearning_suite_report.py \
     --ours "$PROC" --published "results/raw/unlearning/${suite}/published_ref.json" \
     | tee "$REC/report.txt" || echo "!! report skipped (no published_ref?)"
  "$VENV" scripts/unlearning_run_record.py --suite "$suite" --record_dir "$REC" \
     --workdir "results/raw/unlearning/${suite}" --processed "$PROC" --report "$REC/report.txt"
  echo ">>> run record: $REC/ (run_record.md, run_record.json, report.txt)"
done

echo ">>> ALL SUITES DONE — records under docs/run_records/unlearning/*_${TS}/"

# ---- auto-save results OFF the box, so a late return never loses anything (both optional) ----
if [[ -n "${S3_DEST:-}" ]]; then
  echo ">>> auto-save: syncing to ${S3_DEST}"
  aws s3 sync docs/run_records   "${S3_DEST%/}/run_records/" || echo "!! S3 sync (run_records) FAILED"
  aws s3 sync results/processed  "${S3_DEST%/}/processed/"   || echo "!! S3 sync (processed) FAILED"
fi
if [[ "${GIT_PUSH:-0}" == "1" ]]; then
  echo ">>> auto-save: commit + push run record"
  git add docs/run_records/unlearning 2>/dev/null || true
  git commit -m "unlearning run record ${TS}" && git push || echo "!! git push FAILED (GitHub creds not set up?)"
fi

if [[ "${AUTO_SHUTDOWN:-0}" == "1" ]]; then
  echo ">>> AUTO_SHUTDOWN=1 → 'sudo shutdown -h now' in 60s (Ctrl-C to cancel)."
  echo "    Launch with '--instance-initiated-shutdown-behavior stop' so results stay on disk."
  sleep 60; sudo shutdown -h now
fi
