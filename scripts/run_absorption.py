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
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from saebench_audit.metrics import absorption as absorp

# Default suite: Pythia-160M-deduped, 4k width (2^12), layer 8 (configs/registry.yaml: pythia-160m_4k).
DEFAULT_REPO = "adamkarvonen/saebench_pythia-160m-deduped_width-2pow12_date-0108"


def _arch_from_location(location: str) -> str:
    """Label a SAE by the leading token of its top folder, e.g. 'Standard_..._0108/...' -> 'standard'."""
    top = location.strip("/").split("/")[0]
    return top.split("_")[0].lower()


def _sae_name(repo_id: str, location: str) -> str:
    return f"{repo_id.split('/')[-1]}_{location.strip('/').replace('/', '_')}"


def _discover_locations(repo_id: str, layer: int):
    """All SAE folders in the repo (containing ae.pt + config.json), filtered to `layer`, no checkpoints."""
    from sae_bench.custom_saes.run_all_evals_dictionary_learning_saes import (
        get_all_hf_repo_autoencoders,
    )
    from sae_bench.sae_bench_utils import general_utils

    locs = get_all_hf_repo_autoencoders(repo_id)
    locs = general_utils.filter_keywords(locs, exclude_keywords=["checkpoints"], include_keywords=[])
    return sorted(loc for loc in locs if f"resid_post_layer_{layer}" in loc)


def main():
    ap = argparse.ArgumentParser(description="Absorption (SAEBench) — resumable per-SAE runner")
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

    os.makedirs(args.workdir, exist_ok=True)
    locations = args.sae_location or _discover_locations(args.sae_repo, args.layer)
    if not locations:
        print(f"No SAE locations found in {args.sae_repo} for layer {args.layer}", flush=True)
        sys.stdout.flush(); os._exit(1)

    summary_path = os.path.join(args.workdir, "absorption.jsonl")
    done = {}
    if os.path.exists(summary_path):
        for line in open(summary_path):
            line = line.strip()
            if line:
                r = json.loads(line)
                done[r["sae_name"]] = r

    def _record(row):
        with open(summary_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        done[row["sae_name"]] = row

    t0, processed, n_total = time.time(), 0, len(locations)
    for location in locations:
        name = _sae_name(args.sae_repo, location)
        arch = args.arch or _arch_from_location(location)
        if name in done:
            continue
        # Resume without reloading the model if the upstream output already exists.
        out_path = absorp.sae_result_path(args.workdir, name)
        if os.path.exists(out_path) and not args.force_rerun:
            with open(out_path) as f:
                _record({"sae_name": name, "arch": arch, "location": location, "status": "ok",
                         **absorp._flatten_output(json.load(f))})
            continue
        if processed > 0 and time.time() - t0 > args.max_seconds:
            break

        cfg = absorp.AbsorptionConfig(
            model_name=args.model_name, llm_dtype=args.llm_dtype, llm_batch_size=args.llm_batch_size,
            device=args.device,
            absorption_fraction_probe_cos_sim_threshold=args.cos_frac,
            full_absorption_probe_cos_sim_threshold=args.cos_full,
            probe_projection_proportion_threshold=args.proj_prop,
            absorption_fraction_max_absorbing_latents=args.max_absorb,
        )
        print(f"[absorption] {name} (arch={arch})", flush=True)
        sae = absorp.load_released_sae(args.sae_repo, location, model_name=args.model_name,
                                       device=args.device, dtype=args.llm_dtype)
        res = absorp.run_absorption(cfg, sae, name, args.workdir, force_rerun=args.force_rerun)
        _record({"sae_name": name, "arch": arch, "location": location, **res})
        del sae
        processed += 1

    json.dump({"metric": "absorption", "sae_repo": args.sae_repo, "model_name": args.model_name,
               "layer": args.layer, "config": absorp.AbsorptionConfig(
                   model_name=args.model_name, llm_dtype=args.llm_dtype,
                   llm_batch_size=args.llm_batch_size, device=args.device,
                   absorption_fraction_probe_cos_sim_threshold=args.cos_frac,
                   full_absorption_probe_cos_sim_threshold=args.cos_full,
                   probe_projection_proportion_threshold=args.proj_prop,
                   absorption_fraction_max_absorbing_latents=args.max_absorb).to_dict(),
               "n_saes": n_total},
              open(os.path.join(args.workdir, "run_meta.json"), "w"), indent=2)

    n_done = len([r for r in done.values() if r.get("status") == "ok"])
    n_insuf = len([r for r in done.values() if r.get("status") == "insufficient_features"])
    if len(done) >= n_total:
        print(f"ALL_SAES_DONE ({n_done} ok, {n_insuf} insufficient_features of {n_total})", flush=True)
    else:
        print(f"PROGRESS {len(done)}/{n_total}", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
