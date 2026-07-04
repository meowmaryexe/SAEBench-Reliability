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

for suite in "${SUITES[@]}"; do
  P032="$(run_one v0.3.2 "$VENV_032" "$suite" | tail -1)"
  P060="$(run_one v0.6.0 "$VENV_060" "$suite" | tail -1)"
  echo "=================  REPORT: $suite  ================="
  "$VENV_060" scripts/absorption_suite_report.py \
     --v032 "$P032" --v060 "$P060" \
     --published "results/raw/absorption/${suite}_v0.3.2/published_ref.json"
done

echo ">>> ALL SUITES DONE"
if [[ "${AUTO_SHUTDOWN:-0}" == "1" ]]; then
  echo ">>> AUTO_SHUTDOWN=1 → shutting down in 60s (Ctrl-C to cancel)…"; sleep 60; sudo shutdown -h now
fi
