#!/usr/bin/env bash
# Smoke test — proves the whole chain works BEFORE you burn GPU hours.
# Checks: python deps, GPU visible, HF auth (incl. gated Gemma), SAE download, SAE path preflight,
# OpenAI key,
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

echo "[1/7] python deps"
python -c "import torch, transformers, datasets, yaml, einops, openai" 2>/dev/null \
  && ok "core deps import" || bad "missing deps -> pip install -r requirements.txt"

echo "[2/7] GPU"
python - <<'PY'
import torch
if torch.cuda.is_available():
    print(f"  ✅ CUDA: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB)")
else:
    print("  ⚠️  no CUDA — will run on CPU (fine for this smoke test, NOT for the full sweep)")
PY

echo "[3/7] HuggingFace auth (Pythia public, Gemma gated)"
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

echo "[4/7] SAE download from the registry"
python - <<'PY' 2>/dev/null
import sys; sys.path.insert(0, "tests")
import _fixtures
sys.exit(0 if _fixtures.sae_dir() else 1)
PY
[ $? -eq 0 ] && ok "reference SAE downloads from HF" || bad "SAE download failed (network?)"

echo "[5/7] SAE path preflight (all suites x 7 archs x 6 sparsities, HEAD only)"
python - <<'PY'
import sys, yaml
sys.path.insert(0, "scripts")
from run_core_gpu import resolve_folder, trainers_for, preflight
reg = yaml.safe_load(open("configs/registry.yaml"))
bad = 0
for name, suite in reg["sae_suites"].items():
    work = [(a, tr, f"{resolve_folder(reg, suite, a)}/resid_post_layer_{suite['layer']}/trainer_{tr}")
            for a in reg["architectures"] for tr in trainers_for(reg, name, a, None)]
    miss, unknown = preflight(suite["hf_repo"], work)
    tag = f"{name} ({len(work)} SAEs)"
    if miss:
        bad += len(miss)
        print(f"  ❌ {tag}: {len(miss)} missing, e.g. {miss[0][0]} trainer_{miss[0][1]}")
    elif unknown:
        print(f"  ⚠️  {tag}: {len(unknown)} indeterminate (network?)")
    else:
        print(f"  ✅ {tag}")
sys.exit(1 if bad else 0)
PY
[ $? -eq 0 ] && ok "all SAE paths resolve on HuggingFace" || bad "some SAE paths are missing -> fix trainer_overrides in configs/registry.yaml"

echo "[6/7] OpenAI key (AutoInterp judge)"
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

echo "[7/7] fast unit + methodology tests"
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
