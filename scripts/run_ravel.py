"""
Entry point: RAVEL (disentanglement), Stage-1 faithful reproduction.

Wraps the upstream SAEBench eval (sae_bench.evals.ravel) in this repo's resumable per-SAE
run -> aggregate flow. Re-invoke until it prints ALL_SAES_DONE, then aggregate:

  # MUST run under the sae_bench venv on a GPU, from the repo root, with `huggingface-cli login`
  # (accept the google/gemma-2-2b license). RAVEL's prompt data is public and auto-downloads.
  python scripts/run_ravel.py --suite gemma-2-2b_4k --device cuda \
    --workdir results/raw/ravel/gemma-2-2b_4k
  python scripts/aggregate_results.py --metric ravel \
    --workdir results/raw/ravel/gemma-2-2b_4k \
    --out results/processed/ravel/gemma-2-2b_4k.json

Omit --sae_location to enumerate every SAE in the repo (7 archs x 6 sparsities = 42; checkpoint SAEs are
filtered out). RAVEL runs on the BASE gemma-2-2b (not the instruct model), and reads its intervention layer
from each SAE's own cfg.hook_layer — so there is no base/instruct split (unlike unlearning).

The generic scaffolding (suite resolution, SAE discovery, the resumable loop, run_meta, ALL_SAES_DONE)
lives in saebench_audit.{suites,runner}; this file is the thin RAVEL adapter.
"""
import argparse
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from saebench_audit import runner, suites
from saebench_audit.metrics import ravel as rvl

# Default suite: base gemma-2-2b, 4k width (2^12), layer 12 (registry: gemma-2-2b_4k) — the green milestone.
DEFAULT_REPO = "adamkarvonen/saebench_gemma-2-2b_width-2pow12_date-0108"

# Shared helpers, aliased at module scope so existing importers (tests) keep resolving r._arch_from_location.
_resolve_suite = suites.resolve_suite
_arch_from_location = suites.arch_from_location
_sae_name = suites.sae_name
_discover_locations = suites.discover_locations


def main():
    ap = argparse.ArgumentParser(description="RAVEL (SAEBench) — resumable per-SAE runner")
    ap.add_argument("--suite", default=None,
                    help="registry suite name (e.g. gemma-2-2b_4k); sets model/layer/repo")
    ap.add_argument("--model_name", default="gemma-2-2b", help="base model RAVEL runs on (NOT instruct)")
    ap.add_argument("--sae_repo", default=DEFAULT_REPO, help="HuggingFace dictionary_learning SAE repo")
    ap.add_argument("--sae_location", action="append", default=None,
                    help="in-repo SAE folder (repeatable). Omit to enumerate the whole repo.")
    ap.add_argument("--arch", default=None, help="architecture label (default: parsed from location)")
    ap.add_argument("--layer", type=int, default=12, help="resid_post layer (used to filter/enumerate SAEs)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--llm_dtype", default="bfloat16")
    ap.add_argument("--llm_batch_size", type=int, default=8,
                    help="24 GB default; upstream default (2048) is NOT auto-reduced by run_eval")
    ap.add_argument("--artifact_dir", default="artifacts/ravel",
                    help="filtered-dataset cache root (reused across SAEs + widths; keep it persistent)")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--max_seconds", type=float, default=1e9, help="wall-clock budget; checked between SAEs")
    ap.add_argument("--force_rerun", action="store_true")
    args = ap.parse_args()

    if args.suite:  # registry-driven: sets model/layer/repo (base-gemma widths, no code change)
        r = _resolve_suite(args.suite)
        args.model_name, args.layer, args.sae_repo = r["model_name"], r["layer"], r["sae_repo"]

    os.makedirs(args.workdir, exist_ok=True)

    locations = args.sae_location or _discover_locations(args.sae_repo, args.layer)
    if not locations:
        print(f"No SAE locations found in {args.sae_repo} for layer {args.layer}", flush=True)
        sys.stdout.flush(); os._exit(1)

    cfg = rvl.RavelConfig(
        model_name=args.model_name, llm_dtype=args.llm_dtype, llm_batch_size=args.llm_batch_size,
        device=args.device, artifact_dir=args.artifact_dir)

    def _row_extra(location, name):
        return {"arch": args.arch or _arch_from_location(location), "location": location}

    def _resume_row(location, name):
        out_path = rvl.sae_result_path(args.workdir, name)
        if os.path.exists(out_path):
            with open(out_path) as f:
                return {"status": "ok", **rvl._flatten_output(json.load(f))}
        return None

    def _process_one(location, name):
        # RAVEL runs on the base model directly; the SAE is loaded with that same base name.
        sae = suites.load_released_sae(args.sae_repo, location, model_name=args.model_name,
                                       device=args.device, dtype=args.llm_dtype)
        try:
            return rvl.run_ravel(cfg, sae, name, args.workdir, force_rerun=args.force_rerun)
        finally:
            del sae

    done = runner.run_sae_suite(
        locations, workdir=args.workdir, process_one=_process_one,
        name_fn=lambda loc: _sae_name(args.sae_repo, loc), row_extra_fn=_row_extra,
        resume_row_fn=_resume_row, ledger_name="ravel.jsonl", key="sae_name",
        max_seconds=args.max_seconds, force_rerun=args.force_rerun, tag="ravel")

    runner.write_run_meta(
        args.workdir, metric="ravel", suite=args.suite, sae_repo=args.sae_repo,
        model_name=args.model_name, layer=args.layer,
        sae_bench_version=rvl.installed_sae_bench_version(),
        provenance=rvl.environment_provenance(args.device),
        config=cfg.to_dict(), n_saes=len(locations))

    runner.announce(done, len(locations))
    os._exit(0)


if __name__ == "__main__":
    main()
