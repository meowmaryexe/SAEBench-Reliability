#!/usr/bin/env bash
# RAVEL suite driver — runs the full 42-SAE RAVEL suite under the current sae-bench version, aggregates,
# compares to published, and prints the report. Resumable (safe to re-run / after an interruption).
# GPU + gemma-2-2b license required. RAVEL's prompt data is PUBLIC (no gated corpus). See
# docs/aws_ravel_runbook.md. Plan for an ON-DEMAND box: RAVEL is the most expensive metric (~45 min/SAE).
#
#   VENV=/path/.venv AUTO_SHUTDOWN=1 bash scripts/run_ravel_suite.sh
#
# PREREQS (fail fast without these):
#   - `huggingface-cli login` + accept the google/gemma-2-2b (BASE, not -it) license.
#   - Single sae-bench version suffices — RAVEL has no known version drift.
#   - Launch from the repo root so the artifact_dir cache (artifacts/ravel) persists across SAEs/resumes.
#
# Env:
#   VENV       python of the sae-bench venv (GPU-capable)                 [required]
#   DEVICE     cuda|cpu            (default: cuda)
#   LLM_DTYPE  bfloat16|float32    (default: bfloat16)
#   LLM_BATCH  MDBM batch size     (default: 8; drop to 4 if 65k OOMs on a 24 GB card)
#   AUTO_SHUTDOWN=1   `sudo shutdown -h now` on success  (THE cost safeguard — off by default)
set -euo pipefail
cd "$(dirname "$0")/.."

VENV="${VENV:?set VENV to the sae-bench venv python (GPU)}"
DEVICE="${DEVICE:-cuda}"
LLM_DTYPE="${LLM_DTYPE:-bfloat16}"
LLM_BATCH="${LLM_BATCH:-8}"

# ---- Suites to run. Start with 4k (green milestone). 16k/65k are trivial follow-ons: just add them here. ----
SUITES=( "gemma-2-2b_4k" )
# SUITES=( "gemma-2-2b_4k" "gemma-2-2b_16k" "gemma-2-2b_65k" )

run_one () {  # $1=suite → echoes the processed JSON path on its last line
  local suite="$1"
  local raw="results/raw/ravel/${suite}"
  local proc="results/processed/ravel/${suite}.json"
  mkdir -p "$raw" results/processed/ravel
  echo ">>> $suite : running (resumable) under $VENV" 1>&2
  until "$VENV" scripts/run_ravel.py --suite "$suite" --device "$DEVICE" \
        --llm_dtype "$LLM_DTYPE" --llm_batch_size "$LLM_BATCH" --workdir "$raw" \
        2>&1 | tee -a "${raw}.log" | grep -q "ALL_SAES_DONE"; do
    echo ">>> $suite : progress checkpoint, continuing…" 1>&2
  done
  # If RAVEL results aren't published under the derived prefix, this warns; 16k/65k may need --results_prefix.
  "$VENV" scripts/fetch_published_ravel.py --suite "$suite" --workdir "$raw" 1>&2 || \
    echo "!! published fetch failed (verify RAVEL results layout / pass --results_prefix)" 1>&2
  "$VENV" scripts/aggregate_results.py --metric ravel --workdir "$raw" --out "$proc" 1>&2
  echo "$proc"
}

TS="$(date -u +%Y%m%dT%H%M%SZ)"
for suite in "${SUITES[@]}"; do
  PROC="$(run_one "$suite" | tail -1)"

  REC="docs/run_records/ravel/${suite}_${TS}"
  mkdir -p "$REC"
  echo "=================  REPORT: $suite  ================="
  "$VENV" scripts/ravel_suite_report.py \
     --ours "$PROC" --published "results/raw/ravel/${suite}/published_ref.json" \
     | tee "$REC/report.txt" || echo "!! report skipped (no published_ref?)"
  "$VENV" scripts/ravel_run_record.py --suite "$suite" --record_dir "$REC" \
     --workdir "results/raw/ravel/${suite}" --processed "$PROC" --report "$REC/report.txt"
  echo ">>> run record: $REC/ (run_record.md, run_record.json, report.txt)"
done

echo ">>> ALL SUITES DONE — records under docs/run_records/ravel/*_${TS}/"

# ---- auto-save results OFF the box, so a late return never loses anything (both optional) ----
if [[ -n "${S3_DEST:-}" ]]; then
  echo ">>> auto-save: syncing to ${S3_DEST}"
  aws s3 sync docs/run_records   "${S3_DEST%/}/run_records/" || echo "!! S3 sync (run_records) FAILED"
  aws s3 sync results/processed  "${S3_DEST%/}/processed/"   || echo "!! S3 sync (processed) FAILED"
fi
if [[ "${GIT_PUSH:-0}" == "1" ]]; then
  echo ">>> auto-save: commit + push run record"
  git add docs/run_records/ravel 2>/dev/null || true
  git commit -m "ravel run record ${TS}" && git push || echo "!! git push FAILED (GitHub creds not set up?)"
fi

if [[ "${AUTO_SHUTDOWN:-0}" == "1" ]]; then
  echo ">>> AUTO_SHUTDOWN=1 → 'sudo shutdown -h now' in 60s (Ctrl-C to cancel)."
  echo "    Launch with '--instance-initiated-shutdown-behavior stop' so results stay on disk."
  sleep 60; sudo shutdown -h now
fi
