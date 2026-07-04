#!/usr/bin/env bash
# Absorption suite driver — runs the full 42-SAE absorption suite under BOTH sae-bench versions,
# aggregates, compares to published, and prints the report. Resumable (safe to re-run / after a spot
# interruption). Designed for the AWS box; see docs/aws_absorption_runbook.md.
#
#   VENV_032=/path/.venv VENV_060=/path/.venv AUTO_SHUTDOWN=1 bash scripts/run_absorption_suite.sh
#
# Env:
#   VENV_032   python of the sae-bench 0.3.2 venv (matches published)      [required]
#   VENV_060   python of the sae-bench 0.6.0 venv (current code)           [required]
#   DEVICE     cuda|cpu            (default: cuda)
#   LLM_DTYPE  float32|bfloat16    (default: float32; Gemma: bfloat16)
#   AUTO_SHUTDOWN=1   `sudo shutdown -h now` on success  (THE cost safeguard — off by default)
set -euo pipefail
cd "$(dirname "$0")/.."

VENV_032="${VENV_032:?set VENV_032 to the 0.3.2 venv python}"
VENV_060="${VENV_060:?set VENV_060 to the 0.6.0 venv python}"
DEVICE="${DEVICE:-cuda}"
LLM_DTYPE="${LLM_DTYPE:-float32}"

# ---- Suites to run. RUN NOW = Pythia only. Uncomment Gemma when ready (Gemma is HF-gated: hf login). ----
SUITES=( "pythia-160m_4k" )
# SUITES=( "pythia-160m_4k" "gemma-2-2b_4k" "gemma-2-2b_16k" "gemma-2-2b_65k" )

run_one () {  # $1=version tag  $2=venv python  $3=suite
  local tag="$1" py="$2" suite="$3"
  local raw="results/raw/absorption/${suite}_${tag}"
  local proc="results/processed/absorption/${suite}_${tag}.json"
  mkdir -p "$raw" results/processed/absorption   # ensure log/output dirs exist before tee/aggregate
  echo ">>> [$tag] $suite : running (resumable) under $py"
  # re-invoke until ALL_SAES_DONE (the runner is per-SAE resumable + time-boxed)
  until "$py" scripts/run_absorption.py --suite "$suite" --device "$DEVICE" \
        --llm_dtype "$LLM_DTYPE" --workdir "$raw" 2>&1 | tee -a "${raw}.log" | grep -q "ALL_SAES_DONE"; do
    echo ">>> [$tag] $suite : progress checkpoint, continuing…"
  done
  "$py" scripts/fetch_published_absorption.py --suite "$suite" --workdir "$raw"
  "$py" scripts/aggregate_results.py --metric absorption --workdir "$raw" --out "$proc"
  echo "$proc"
}

TS="$(date -u +%Y%m%dT%H%M%SZ)"
for suite in "${SUITES[@]}"; do
  P032="$(run_one v0.3.2 "$VENV_032" "$suite" | tail -1)"
  P060="$(run_one v0.6.0 "$VENV_060" "$suite" | tail -1)"

  # ---- durable, git-tracked run record for recordkeeping (self-contained; raw stays referenced) ----
  REC="docs/run_records/absorption/${suite}_${TS}"
  mkdir -p "$REC"
  echo "=================  REPORT: $suite  ================="
  "$VENV_060" scripts/absorption_suite_report.py \
     --v032 "$P032" --v060 "$P060" \
     --published "results/raw/absorption/${suite}_v0.3.2/published_ref.json" \
     | tee "$REC/report.txt"
  "$VENV_060" scripts/absorption_run_record.py --suite "$suite" --record_dir "$REC" \
     --v032_workdir "results/raw/absorption/${suite}_v0.3.2" \
     --v060_workdir "results/raw/absorption/${suite}_v0.6.0" \
     --v032_processed "$P032" --v060_processed "$P060" --report "$REC/report.txt"
  echo ">>> run record: $REC/ (run_record.md, run_record.json, report.txt)"
done

echo ">>> ALL SUITES DONE — records under docs/run_records/absorption/*_${TS}/"

# ---- auto-save results OFF the box, so a late return never loses anything (both optional) ----
#   S3_DEST=s3://bucket/prefix   -> sync run records + processed results to S3
#   GIT_PUSH=1                   -> commit + push the run record (needs GitHub creds on the box)
if [[ -n "${S3_DEST:-}" ]]; then
  echo ">>> auto-save: syncing to ${S3_DEST}"
  aws s3 sync docs/run_records   "${S3_DEST%/}/run_records/" || echo "!! S3 sync (run_records) FAILED"
  aws s3 sync results/processed  "${S3_DEST%/}/processed/"   || echo "!! S3 sync (processed) FAILED"
fi
if [[ "${GIT_PUSH:-0}" == "1" ]]; then
  echo ">>> auto-save: commit + push run record"
  git add docs/run_records/absorption 2>/dev/null || true
  git commit -m "absorption run record ${TS}" && git push || echo "!! git push FAILED (GitHub creds not set up?)"
fi

if [[ "${AUTO_SHUTDOWN:-0}" == "1" ]]; then
  echo ">>> AUTO_SHUTDOWN=1 → 'sudo shutdown -h now' in 60s (Ctrl-C to cancel)."
  echo "    This STOPS or TERMINATES per the instance's shutdown-behavior chosen at launch."
  echo "    Tip: launch with '--instance-initiated-shutdown-behavior stop' so results stay on disk"
  echo "    (and/or set S3_DEST above) — then a late return can never lose data."
  sleep 60; sudo shutdown -h now
fi
