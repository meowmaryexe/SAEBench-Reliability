"""
From-scratch regeneration of the absorption version-drift finding: run the released Pythia-160M 4k
Standard SAE (trainer_0) through whichever `sae_bench` is installed in the CURRENT interpreter, and print
both scores. Run it under each pinned venv and compare:

  # published version (sae-bench 0.3.2 @141aff72 + sae_lens 5.3.1) -> fraction ~0.164, full ~0.0095
  /Users/alor/saebench-0125-env/.venv/bin/python scripts/reproduce_absorption_drift.py \
      --workdir results/raw/absorption/standard_4k_t0_v0125

  # current version (sae-bench 0.6.0)                              -> fraction ~0.016, full ~0.0077
  /Users/alor/saebench-absorption-env/.venv/bin/python scripts/reproduce_absorption_drift.py \
      --workdir results/raw/absorption/standard_4k_t0_v060

Setup for the two venvs (one-time):
  python3.11 -m venv <dir>/.venv && <dir>/.venv/bin/pip install \
      "sae-bench @ git+https://github.com/adamkarvonen/SAEBench.git@141aff72928f7588c1451bed47c401e1d565d471" \
      "sae_lens==5.3.1" "transformers>=4.40,<5"        # published (0.3.2)
  <dir>/.venv/bin/pip install "sae-bench" "transformers>=4.51,<5"                   # current (0.6.0)

The quick, no-compute version of this comparison is tests/test_absorption_version_drift.py (uses the
committed result JSONs). Full write-up: docs/findings/absorption_version_drift.md.

~9 min/run on CPU (Pythia-160M is transformer_lens-cached after the first run).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from saebench_audit import suites
from saebench_audit.metrics import absorption as absorp

DEFAULT_REPO = "adamkarvonen/saebench_pythia-160m-deduped_width-2pow12_date-0108"
DEFAULT_LOC = "Standard_pythia-160m-deduped__0108/resid_post_layer_8/trainer_0"


def main():
    ap = argparse.ArgumentParser(description="Regenerate one absorption SAE result under the current sae_bench")
    ap.add_argument("--sae_repo", default=DEFAULT_REPO)
    ap.add_argument("--sae_location", default=DEFAULT_LOC)
    ap.add_argument("--model_name", default="pythia-160m-deduped")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--llm_dtype", default="float32")
    ap.add_argument("--llm_batch_size", type=int, default=32)
    ap.add_argument("--workdir", required=True)
    args = ap.parse_args()

    import importlib.metadata as md
    print(f"[env] sae_bench={md.version('sae-bench')}  sae_lens={md.version('sae_lens')}  "
          f"transformers={md.version('transformers')}", flush=True)

    # sae-bench 0.3.2 predates torch 2.6's weights_only=True default; force weights_only=False so it loads.
    absorp.ensure_torch_load_shim()
    os.makedirs(args.workdir, exist_ok=True)
    sae = absorp.load_released_sae(args.sae_repo, args.sae_location, model_name=args.model_name,
                                   device=args.device, dtype=args.llm_dtype)
    from sae_bench.evals.absorption.main import run_eval
    from sae_bench.evals.absorption.eval_config import AbsorptionEvalConfig
    from sae_bench.sae_bench_utils import general_utils

    # Only the fields common to both 0.3.2 and 0.6.0 configs (we run each version's native defaults).
    cfg = AbsorptionEvalConfig(model_name=args.model_name, random_seed=42,
                               llm_batch_size=args.llm_batch_size, llm_dtype=args.llm_dtype)
    name = suites.sae_name(args.sae_repo, args.sae_location)
    res = run_eval(cfg, [(name, sae)], args.device, args.workdir, force_rerun=True)

    out = res.get(f"{name}_custom_sae")
    if out is None:
        out = json.load(open(general_utils.get_results_filepath(args.workdir, name, "custom_sae")))
    m = out["eval_result_metrics"]["mean"]
    print(f"[result] mean_absorption_fraction_score = {m['mean_absorption_fraction_score']:.4f}")
    print(f"[result] mean_full_absorption_score     = {m['mean_full_absorption_score']:.4f}")
    print("ABSORPTION_DRIFT_RUN_DONE", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
