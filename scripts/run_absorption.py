"""
Entry point: Absorption (first-letter feature absorption), Stage-1 faithful reproduction.

Wraps the upstream SAEBench eval (sae_bench.evals.absorption) in this repo's resumable per-SAE
run -> aggregate flow. Re-invoke until it prints ALL_SAES_DONE, then aggregate:

  # MUST run under the dedicated pinned absorption venv (transformers<5 + transformer_lens 2.16.1):
  /Users/alor/saebench-absorption-env/.venv/bin/python scripts/run_absorption.py \
    --sae_repo adamkarvonen/saebench_pythia-160m-deduped_width-2pow12_date-0108 \
    --sae_location Standard_pythia-160m-deduped__0108/resid_post_layer_8/trainer_0 \
    --arch standard --layer 8 --device cpu \
    --workdir results/raw/absorption/standard_4k_t0
  python scripts/aggregate_results.py --metric absorption \
    --workdir results/raw/absorption/standard_4k_t0 \
    --out results/processed/absorption/standard_4k_t0.json

Omit --sae_location to enumerate every SAE in the repo (all 7 archs x 6 sparsities) for the suite run.
Thresholds default to the SHIPPED upstream constants (faithful Stage 1); overriding any of them is an
audit-phase action and forces a rerun. See src/saebench_audit/metrics/absorption.py + docs.

The generic scaffolding (suite resolution, SAE discovery, the resumable loop, run_meta, the
ALL_SAES_DONE protocol) lives in saebench_audit.{suites,runner}; this file is the thin absorption adapter.
"""
import argparse
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from saebench_audit import runner, suites
from saebench_audit.metrics import absorption as absorp

# Default suite: Pythia-160M-deduped, 4k width (2^12), layer 8 (configs/registry.yaml: pythia-160m_4k).
DEFAULT_REPO = "adamkarvonen/saebench_pythia-160m-deduped_width-2pow12_date-0108"

# Shared helpers, aliased at module scope so existing importers (tests) keep resolving r._arch_from_location.
_resolve_suite = suites.resolve_suite
_arch_from_location = suites.arch_from_location
_sae_name = suites.sae_name
_discover_locations = suites.discover_locations


def main():
    ap = argparse.ArgumentParser(description="Absorption (SAEBench) — resumable per-SAE runner")
    ap.add_argument("--suite", default=None,
                    help="registry suite name (e.g. pythia-160m_4k, gemma-2-2b_4k); sets model/layer/repo")
    ap.add_argument("--model_name", default="pythia-160m-deduped", help="transformer_lens model name")
    ap.add_argument("--sae_repo", default=DEFAULT_REPO, help="HuggingFace dictionary_learning SAE repo")
    ap.add_argument("--sae_location", action="append", default=None,
                    help="in-repo SAE folder (repeatable). Omit to enumerate the whole repo.")
    ap.add_argument("--arch", default=None, help="architecture label (default: parsed from location)")
    ap.add_argument("--layer", type=int, default=8, help="resid_post layer (used to filter/enumerate)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--llm_dtype", default="float32")
    ap.add_argument("--llm_batch_size", type=int, default=32, help="inference batching only (no metric effect)")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--max_seconds", type=float, default=1e9, help="wall-clock budget; checked between SAEs")
    ap.add_argument("--force_rerun", action="store_true")
    # Audit-phase threshold overrides (default = shipped). Setting any of these forces a rerun.
    ap.add_argument("--cos_frac", type=float, default=absorp.SHIPPED_ABSORPTION_FRACTION_COS)
    ap.add_argument("--cos_full", type=float, default=absorp.SHIPPED_FULL_ABSORPTION_COS)
    ap.add_argument("--proj_prop", type=float, default=absorp.SHIPPED_PROJECTION_PROPORTION)
    ap.add_argument("--max_absorb", type=int, default=absorp.SHIPPED_MAX_ABSORBING_LATENTS)
    args = ap.parse_args()

    if args.suite:  # registry-driven: sets model/layer/repo (Pythia now, Gemma later, no code change)
        r = _resolve_suite(args.suite)
        args.model_name, args.layer, args.sae_repo = r["model_name"], r["layer"], r["sae_repo"]

    os.makedirs(args.workdir, exist_ok=True)
    locations = args.sae_location or _discover_locations(args.sae_repo, args.layer)
    if not locations:
        print(f"No SAE locations found in {args.sae_repo} for layer {args.layer}", flush=True)
        sys.stdout.flush(); os._exit(1)

    # Config depends only on args (not the per-SAE location), so build it once and reuse.
    cfg = absorp.AbsorptionConfig(
        model_name=args.model_name, llm_dtype=args.llm_dtype, llm_batch_size=args.llm_batch_size,
        device=args.device,
        absorption_fraction_probe_cos_sim_threshold=args.cos_frac,
        full_absorption_probe_cos_sim_threshold=args.cos_full,
        probe_projection_proportion_threshold=args.proj_prop,
        absorption_fraction_max_absorbing_latents=args.max_absorb,
    )

    def _row_extra(location, name):
        return {"arch": args.arch or _arch_from_location(location), "location": location}

    def _resume_row(location, name):
        # Resume without reloading the model if upstream already wrote this SAE's result JSON.
        out_path = absorp.sae_result_path(args.workdir, name)
        if os.path.exists(out_path):
            with open(out_path) as f:
                return {"status": "ok", **absorp._flatten_output(json.load(f))}
        return None

    def _process_one(location, name):
        sae = absorp.load_released_sae(args.sae_repo, location, model_name=args.model_name,
                                       device=args.device, dtype=args.llm_dtype)
        try:
            return absorp.run_absorption(cfg, sae, name, args.workdir, force_rerun=args.force_rerun)
        finally:
            del sae

    done = runner.run_sae_suite(
        locations, workdir=args.workdir, process_one=_process_one, name_fn=lambda loc: _sae_name(args.sae_repo, loc),
        row_extra_fn=_row_extra, resume_row_fn=_resume_row, ledger_name="absorption.jsonl",
        key="sae_name", max_seconds=args.max_seconds, force_rerun=args.force_rerun, tag="absorption")

    runner.write_run_meta(
        args.workdir, metric="absorption", suite=args.suite, sae_repo=args.sae_repo,
        model_name=args.model_name, layer=args.layer,
        sae_bench_version=absorp.installed_sae_bench_version(),
        provenance=absorp.environment_provenance(args.device),
        config=cfg.to_dict(), n_saes=len(locations))

    runner.announce(done, len(locations), extra_counts={"insufficient_features": "insufficient_features"})
    os._exit(0)


if __name__ == "__main__":
    main()
