#!/usr/bin/env bash
# Smoke test — proves the whole chain works BEFORE you burn GPU hours.
# Checks: python deps, GPU visible, HF auth (incl. gated Gemma), SAE download, OpenAI key,
# then runs Core + AutoInterp on ONE small SAE end-to-end (~3-5 min).
#
#   bash scripts/smoke_test.sh
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
PASS=0; FAIL=0
ok(){ echo "  ✅ $1"; PASS=$((PASS+1)); }
bad(){ echo "  ❌ $1"; FAIL=$((FAIL+1)); }

echo "=============================================="
echo " SAEBench-Reliability smoke test"
echo "=============================================="

echo "[1/6] python deps"
python -c "import torch, transformers, datasets, yaml, einops, openai" 2>/dev/null \
  && ok "core deps import" || bad "missing deps -> pip install -r requirements.txt"

echo "[2/6] GPU"
python - <<'PY'
import torch
if torch.cuda.is_available():
    print(f"  ✅ CUDA: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB)")
else:
    print("  ⚠️  no CUDA — will run on CPU (fine for this smoke test, NOT for the full sweep)")
PY

echo "[3/6] HuggingFace auth (Pythia public, Gemma gated)"
python - <<'PY' 2>/dev/null
import sys
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained("EleutherAI/pythia-160m-deduped")
PY
[ $? -eq 0 ] && ok "Pythia tokenizer (public)" || bad "Pythia download failed (network?)"
python - <<'PY' 2>/dev/null
import sys
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained("google/gemma-2-2b")
PY
[ $? -eq 0 ] && ok "Gemma access (gated model OK)" \
  || bad "Gemma GATED-ACCESS FAILED -> accept license at hf.co/google/gemma-2-2b then 'huggingface-cli login' (Gemma suites will not run)"

echo "[4/6] SAE download from the registry"
python - <<'PY' 2>/dev/null
import sys; sys.path.insert(0, "tests")
import _fixtures
sys.exit(0 if _fixtures.sae_dir() else 1)
PY
[ $? -eq 0 ] && ok "reference SAE downloads from HF" || bad "SAE download failed (network?)"

echo "[5/6] OpenAI key (AutoInterp judge)"
if [ -f openai_api_key.txt ]; then
  python - <<'PY' 2>/dev/null
from openai import OpenAI
c = OpenAI(api_key=open("openai_api_key.txt").read().strip())
c.chat.completions.create(model="gpt-4o-mini",
    messages=[{"role":"user","content":"Reply with exactly: OK"}], max_tokens=5)
PY
  [ $? -eq 0 ] && ok "gpt-4o-mini reachable" || bad "OpenAI call failed (bad key? no credit?)"
else
  bad "openai_api_key.txt missing -> AutoInterp cannot run (Core still works)"
fi

echo "[6/6] fast unit + methodology tests"
python tests/test_core_units.py        >/dev/null 2>&1 && ok "core unit tests"        || bad "core unit tests"
python tests/test_autointerp_units.py  >/dev/null 2>&1 && ok "autointerp unit tests"  || bad "autointerp unit tests"
python tests/test_autointerp_prompts.py>/dev/null 2>&1 && ok "prompt verbatim tests"  || bad "prompt verbatim tests"
python tests/test_autointerp_oracle.py >/dev/null 2>&1 && ok "autointerp oracle"      || bad "autointerp oracle"

echo "=============================================="
echo " passed: $PASS   failed: $FAIL"
if [ "$FAIL" -eq 0 ]; then
  echo " ✅ ALL GOOD — you're clear to run:  bash scripts/run_all_gpu.sh"
else
  echo " ⚠️  fix the ❌ items above before the full run (see docs/GPU_SETUP.md §10)"
fi
echo "=============================================="
exit $([ "$FAIL" -eq 0 ] && echo 0 || echo 1)
