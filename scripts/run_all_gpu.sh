#!/usr/bin/env bash
# Run the FULL paper-scale reproduction (Core + AutoInterp) across all 6 SAE suites.
#
#   bash scripts/run_all_gpu.sh              # everything (252 SAEs x 2 metrics)
#   METRICS=core bash scripts/run_all_gpu.sh # Core only (no OpenAI cost)
#   SUITES="gemma-2-2b_65k" bash scripts/run_all_gpu.sh   # headline only
#
# SAFE TO RE-RUN: every runner is checkpoint-resumable (per-SAE and per-latent). If the box dies,
# SSH drops, or a Spot instance is reclaimed, just run this again — it picks up where it left off.
# Recommended: run under tmux/screen so an SSH drop doesn't kill it.
set -uo pipefail
cd "$(dirname "$0")/.."

SUITES="${SUITES:-pythia-160m_4k pythia-160m_16k pythia-160m_65k gemma-2-2b_4k gemma-2-2b_16k gemma-2-2b_65k}"
METRICS="${METRICS:-core autointerp}"
DEVICE="${DEVICE:-cuda}"
JUDGE_WORKERS="${JUDGE_WORKERS:-10}"
LOGDIR="${LOGDIR:-logs_run}"
mkdir -p "$LOGDIR" results/processed/core_loss_recovered results/processed/autointerp

echo "suites : $SUITES"
echo "metrics: $METRICS   device: $DEVICE"
echo "logs   : $LOGDIR/    (tail -f to watch)"
echo

# A failing suite must not be silently skipped: record each outcome and report it at the end.
# We deliberately do NOT abort the sweep on one failure — continuing is correct, hiding it is not.
OK_RUNS=(); FAILED_RUNS=(); SKIPPED_RUNS=()

for SUITE in $SUITES; do
  for M in $METRICS; do
    if [ "$M" = "core" ]; then
      OUT="results/processed/core_loss_recovered/${SUITE}_saebench_core.json"
      LOG="$LOGDIR/core_${SUITE}.log"
      echo "▶ CORE  $SUITE  -> $OUT"
      python scripts/run_core_gpu.py \
        --config configs/gpu/core_gpu.yaml \
        --suite "$SUITE" --out "$OUT" 2>&1 | tee "$LOG" | tail -3
      RC=${PIPESTATUS[0]}          # exit status of python, not of tail
      if [ "$RC" -eq 0 ]; then
        OK_RUNS+=("core/$SUITE"); echo "  ✅ core $SUITE"
      else
        FAILED_RUNS+=("core/$SUITE (exit $RC, see $LOG)"); echo "  ❌ core $SUITE — exit $RC (see $LOG)"
      fi
    else
      if [ ! -f openai_api_key.txt ]; then
        echo "⚠  skipping AUTOINTERP $SUITE — openai_api_key.txt not found"
        SKIPPED_RUNS+=("autointerp/$SUITE (no openai_api_key.txt)"); continue
      fi
      OUT="results/processed/autointerp/${SUITE}_autointerp.json"
      LOG="$LOGDIR/autointerp_${SUITE}.log"
      echo "▶ AUTOINTERP  $SUITE  -> $OUT"
      python scripts/run_autointerp_gpu.py \
        --config configs/gpu/autointerp_gpu.yaml \
        --registry configs/registry.yaml \
        --suite "$SUITE" --device "$DEVICE" --judge_workers "$JUDGE_WORKERS" \
        --keyfile openai_api_key.txt --out "$OUT" 2>&1 | tee "$LOG" | tail -3
      RC=${PIPESTATUS[0]}
      if [ "$RC" -eq 0 ]; then
        OK_RUNS+=("autointerp/$SUITE"); echo "  ✅ autointerp $SUITE"
      else
        FAILED_RUNS+=("autointerp/$SUITE (exit $RC, see $LOG)"); echo "  ❌ autointerp $SUITE — exit $RC (see $LOG)"
      fi
    fi
    echo
  done
done

echo "▶ figures"
python scripts/make_figures.py
FIG_RC=$?
[ "$FIG_RC" -eq 0 ] || echo "  ⚠  make_figures.py exited $FIG_RC (does not affect the sweep result)"

echo
echo "=========================================="
echo " SUMMARY: ${#OK_RUNS[@]} ok, ${#FAILED_RUNS[@]} failed, ${#SKIPPED_RUNS[@]} skipped"
for r in ${OK_RUNS+"${OK_RUNS[@]}"};           do echo "   ✅ $r"; done
for r in ${SKIPPED_RUNS+"${SKIPPED_RUNS[@]}"}; do echo "   ⚠  $r"; done
for r in ${FAILED_RUNS+"${FAILED_RUNS[@]}"};   do echo "   ❌ $r"; done
echo
echo " Results:"
ls -1 results/processed/core_loss_recovered/*.json 2>/dev/null | sed 's/^/   /'
ls -1 results/processed/autointerp/*.json 2>/dev/null | sed 's/^/   /'
echo " Figures in figures/ ; per-suite logs in $LOGDIR/"
echo " Remember to STOP the EC2 instance when you're done."
if [ ${#FAILED_RUNS[@]} -ne 0 ]; then
  echo " ⚠️  ${#FAILED_RUNS[@]} RUN(S) FAILED — the sweep is INCOMPLETE."
  echo "=========================================="
  exit 1
fi
echo " ✅ ALL RUNS COMPLETED."
echo "=========================================="
