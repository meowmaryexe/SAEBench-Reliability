"""
Minimal SCR/TPP smoke runner.

This mirrors the official SAEBench acceptance tests, but runs as a script
instead of through pytest.

Run from the SAEBench repo root, for example:

python /content/SAEBench-Reliability/scripts/run_scr_tpp_smoke.py --metric tpp
python /content/SAEBench-Reliability/scripts/run_scr_tpp_smoke.py --metric scr
"""

import argparse
import json
from pathlib import Path

import torch

import sae_bench.evals.scr_and_tpp.main as scr_and_tpp
from sae_bench.evals.scr_and_tpp.eval_config import ScrAndTppEvalConfig
from sae_bench.sae_bench_utils.sae_selection_utils import select_saes_multiple_patterns


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_config(metric: str) -> ScrAndTppEvalConfig:
    config = ScrAndTppEvalConfig()

    config.dataset_names = ["LabHC/bias_in_bios_class_set1"]
    config.model_name = "pythia-70m-deduped"
    config.n_values = [10]
    config.sae_batch_size = 250
    config.llm_batch_size = 500
    config.llm_dtype = "float32"

    if metric == "scr":
        config.perform_scr = True
        config.random_seed = 48
        config.lower_vram_usage = True
        config.column1_vals_lookup = {
            "LabHC/bias_in_bios_class_set1": [
                ("professor", "nurse"),
            ],
        }
    elif metric == "tpp":
        config.perform_scr = False
        config.random_seed = 44
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return config


def select_test_sae():
    layer = 4
    sae_regex_patterns = [
        r"(sae_bench_pythia70m_sweep_topk_ctx128_0730).*",
    ]
    sae_block_pattern = [
        rf".*blocks\.([{layer}])\.hook_resid_post__trainer_(10)$",
    ]

    selected_saes = select_saes_multiple_patterns(
        sae_regex_patterns,
        sae_block_pattern,
    )

    if len(selected_saes) != 1:
        raise RuntimeError(f"Expected 1 SAE, found {len(selected_saes)}: {selected_saes}")

    return selected_saes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metric",
        choices=["scr", "tpp"],
        required=True,
        help="Which metric smoke test to run.",
    )
    parser.add_argument(
        "--output-dir",
        default="evals/scr_and_tpp/smoke_results",
        help="Output directory relative to the SAEBench repo root.",
    )
    args = parser.parse_args()

    device = get_device()
    config = build_config(args.metric)
    selected_saes = select_test_sae()

    print(f"Running {args.metric.upper()} smoke test")
    print(f"Device: {device}")
    print(f"Selected SAEs: {selected_saes}")
    print(f"Output dir: {args.output_dir}")

    results = scr_and_tpp.run_eval(
        config,
        selected_saes,
        device,
        output_path=args.output_dir,
        force_rerun=True,
        clean_up_activations=True,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / f"{args.metric}_smoke_results_summary.json"
    with summary_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()