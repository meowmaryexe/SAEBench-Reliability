"""
Chunkable faithful reproduction runner for Mary-owned SAEBench metrics on
Pythia-160M dictionary-learning SAEs.

This script must be executed from the upstream SAEBench repository root so that
the `sae_bench` package and its evaluation outputs resolve exactly as expected.

Example from the SAEBench repository root:

    python ../SAEBench-Reliability/scripts/run_pythia160m_reproduction.py \
        --metric tpp \
        --start 0 \
        --end 6 \
        --seed 42

The selected SAE interval follows Python slicing semantics: [start, end).
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from sae_bench.custom_saes.run_all_evals_dictionary_learning_saes import (
    MODEL_CONFIGS,
    get_all_hf_repo_autoencoders,
    load_dictionary_learning_sae,
)
from sae_bench.evals.scr_and_tpp.eval_config import ScrAndTppEvalConfig
from sae_bench.evals.scr_and_tpp.main import run_eval as run_scr_or_tpp
from sae_bench.evals.sparse_probing.eval_config import SparseProbingEvalConfig
from sae_bench.evals.sparse_probing.main import run_eval as run_sparse_probing


REPO_ID = (
    "adamkarvonen/"
    "saebench_pythia-160m-deduped_width-2pow14_date-0108"
)
MODEL_NAME = "pythia-160m-deduped"
EXPECTED_SAE_COUNT = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_DIR = (
    PROJECT_ROOT / "results" / "raw" / "reproduction_manifests"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a faithful, chunkable Pythia-160M reproduction for "
            "TPP, SCR, or Sparse Probing."
        )
    )
    parser.add_argument(
        "--metric",
        choices=("tpp", "scr", "sparse_probing"),
        required=True,
        help="Official SAEBench metric to reproduce.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Inclusive start index in the deterministically sorted SAE list.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help=(
            "Exclusive end index in the sorted SAE list. "
            "Defaults to the total number of SAEs."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed passed to the official SAEBench evaluation config.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="Evaluation device. Full reproduction runs should use CUDA.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("eval_results"),
        help=(
            "Output root relative to the upstream SAEBench repository. "
            "Defaults to eval_results."
        ),
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=DEFAULT_MANIFEST_DIR,
        help="Directory in which run manifests are recorded.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Recompute results even when official output JSONs already exist.",
    )
    parser.add_argument(
        "--keep-activations",
        action="store_true",
        help="Keep cached activation artifacts after the selected chunk finishes.",
    )
    parser.add_argument(
        "--lower-vram-usage",
        action="store_true",
        help="Enable SAEBench's lower-VRAM execution behavior.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print and record the selected chunk without loading models or SAEs.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but it is unavailable.")
        return requested

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def git_commit(repo: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def verify_upstream_root() -> Path:
    upstream_root = Path.cwd().resolve()
    expected_package = upstream_root / "sae_bench"
    expected_runner = (
        expected_package
        / "custom_saes"
        / "run_all_evals_dictionary_learning_saes.py"
    )

    if not expected_package.is_dir() or not expected_runner.is_file():
        raise RuntimeError(
            "Run this script from the upstream SAEBench repository root.\n"
            "For example:\n"
            "  cd /content/SAEBench\n"
            "  python /content/SAEBench-Reliability/scripts/"
            "run_pythia160m_reproduction.py ..."
        )

    return upstream_root


def enumerate_saes() -> list[str]:
    locations = sorted(get_all_hf_repo_autoencoders(REPO_ID))

    if len(locations) != EXPECTED_SAE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_SAE_COUNT} Pythia-160M SAEs, "
            f"but found {len(locations)}."
        )

    unexpected = [
        location
        for location in locations
        if "resid_post_layer_8" not in location
    ]
    if unexpected:
        raise RuntimeError(
            "Found SAE locations outside the expected layer-8 collection:\n"
            + "\n".join(unexpected)
        )

    return locations


def validate_slice(
    start: int,
    end: int | None,
    total: int,
) -> tuple[int, int]:
    resolved_end = total if end is None else end

    if start < 0:
        raise ValueError("--start must be nonnegative.")
    if resolved_end < 0:
        raise ValueError("--end must be nonnegative.")
    if start >= resolved_end:
        raise ValueError(
            f"Invalid empty slice [{start}, {resolved_end}). "
            "--start must be smaller than --end."
        )
    if resolved_end > total:
        raise ValueError(
            f"--end={resolved_end} exceeds the total SAE count of {total}."
        )

    return start, resolved_end


def unique_sae_id(location: str) -> str:
    repo_name = REPO_ID.split("/", maxsplit=1)[1]
    normalized_location = location.replace("/", "_")
    return f"{repo_name}_{normalized_location}"


def manifest_path(
    manifest_dir: Path,
    metric: str,
    start: int,
    end: int,
    seed: int,
) -> Path:
    filename = (
        f"pythia160m_{metric}_"
        f"saes-{start:02d}-{end:02d}_seed-{seed}.json"
    )
    return manifest_dir / filename


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")

    temporary_path.replace(path)


def build_manifest(
    *,
    args: argparse.Namespace,
    upstream_root: Path,
    device: str,
    all_locations: list[str],
    selected_locations: list[str],
    start: int,
    end: int,
) -> dict[str, Any]:
    model_config = MODEL_CONFIGS[MODEL_NAME]

    return {
        "schema_version": 1,
        "status": "planned",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_at_utc": None,
        "metric": args.metric,
        "model_name": MODEL_NAME,
        "sae_repository": REPO_ID,
        "expected_total_sae_count": EXPECTED_SAE_COUNT,
        "observed_total_sae_count": len(all_locations),
        "slice": {
            "start_inclusive": start,
            "end_exclusive": end,
            "count": len(selected_locations),
        },
        "selected_sae_locations": selected_locations,
        "selected_sae_ids": [
            unique_sae_id(location) for location in selected_locations
        ],
        "random_seed": args.seed,
        "device": device,
        "llm_batch_size": model_config["batch_size"],
        "llm_dtype": model_config["dtype"],
        "force_rerun": args.force_rerun,
        "save_activations": True,
        "clean_up_activations": not args.keep_activations,
        "lower_vram_usage": args.lower_vram_usage,
        "output_root": str(args.output_root),
        "upstream_saebench_root": str(upstream_root),
        "upstream_saebench_commit": git_commit(upstream_root),
        "reliability_repository_root": str(PROJECT_ROOT),
        "reliability_repository_commit": git_commit(PROJECT_ROOT),
        "dry_run": args.dry_run,
        "error": None,
    }


def load_selected_saes(
    locations: list[str],
    device: str,
    dtype: torch.dtype,
) -> list[tuple[str, Any]]:
    selected_saes: list[tuple[str, Any]] = []

    for offset, location in enumerate(locations, start=1):
        print(f"[{offset}/{len(locations)}] Loading SAE: {location}")

        sae = load_dictionary_learning_sae(
            repo_id=REPO_ID,
            location=location,
            layer=None,
            model_name=MODEL_NAME,
            device=device,
            dtype=dtype,
        )
        selected_saes.append((unique_sae_id(location), sae))

    return selected_saes


def run_metric(
    *,
    metric: str,
    selected_saes: list[tuple[str, Any]],
    device: str,
    seed: int,
    output_root: Path,
    force_rerun: bool,
    clean_up_activations: bool,
    lower_vram_usage: bool,
) -> dict[str, Any]:
    model_config = MODEL_CONFIGS[MODEL_NAME]
    llm_batch_size = model_config["batch_size"]
    llm_dtype = model_config["dtype"]

    if metric in {"tpp", "scr"}:
        config = ScrAndTppEvalConfig(
            model_name=MODEL_NAME,
            random_seed=seed,
            perform_scr=(metric == "scr"),
            llm_batch_size=llm_batch_size,
            llm_dtype=llm_dtype,
        )
        config.lower_vram_usage = lower_vram_usage

        return run_scr_or_tpp(
            config=config,
            selected_saes=selected_saes,
            device=device,
            output_path=str(output_root),
            force_rerun=force_rerun,
            clean_up_activations=clean_up_activations,
            save_activations=True,
        )

    if metric == "sparse_probing":
        config = SparseProbingEvalConfig(
            model_name=MODEL_NAME,
            random_seed=seed,
            llm_batch_size=llm_batch_size,
            llm_dtype=llm_dtype,
        )
        config.lower_vram_usage = lower_vram_usage

        return run_sparse_probing(
            config=config,
            selected_saes=selected_saes,
            device=device,
            output_path=str(output_root / "sparse_probing"),
            force_rerun=force_rerun,
            clean_up_activations=clean_up_activations,
            save_activations=True,
        )

    raise ValueError(f"Unsupported metric: {metric}")


def release_loaded_saes(selected_saes: list[tuple[str, Any]]) -> None:
    selected_saes.clear()
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def main() -> None:
    args = parse_args()
    upstream_root = verify_upstream_root()
    device = resolve_device(args.device)

    all_locations = enumerate_saes()
    start, end = validate_slice(args.start, args.end, len(all_locations))
    selected_locations = all_locations[start:end]

    manifest_file = manifest_path(
        args.manifest_dir,
        args.metric,
        start,
        end,
        args.seed,
    )
    manifest = build_manifest(
        args=args,
        upstream_root=upstream_root,
        device=device,
        all_locations=all_locations,
        selected_locations=selected_locations,
        start=start,
        end=end,
    )
    write_manifest(manifest_file, manifest)

    print("=" * 72)
    print("Pythia-160M SAEBench reproduction")
    print(f"Metric:       {args.metric}")
    print(f"Seed:         {args.seed}")
    print(f"Device:       {device}")
    print(f"SAE slice:    [{start}, {end})")
    print(f"SAE count:    {len(selected_locations)}")
    print(f"Manifest:     {manifest_file}")
    print(f"Output root:  {args.output_root}")
    print("=" * 72)

    for index, location in enumerate(
        selected_locations,
        start=start,
    ):
        print(f"{index:02d}: {location}")

    if args.dry_run:
        manifest["status"] = "dry_run_complete"
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_manifest(manifest_file, manifest)
        print("\nDry run complete; no models or SAEs were loaded.")
        return

    if device != "cuda":
        print(
            "\nWARNING: Full reproduction runs are intended for CUDA. "
            f"The resolved device is {device!r}."
        )

    dtype = torch.float32
    selected_saes: list[tuple[str, Any]] = []

    try:
        selected_saes = load_selected_saes(
            selected_locations,
            device,
            dtype,
        )

        manifest["status"] = "running"
        manifest["started_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_manifest(manifest_file, manifest)

        results = run_metric(
            metric=args.metric,
            selected_saes=selected_saes,
            device=device,
            seed=args.seed,
            output_root=args.output_root,
            force_rerun=args.force_rerun,
            clean_up_activations=not args.keep_activations,
            lower_vram_usage=args.lower_vram_usage,
        )

        manifest["status"] = "completed"
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["returned_result_keys"] = sorted(results.keys())
        manifest["returned_result_count"] = len(results)
        write_manifest(manifest_file, manifest)

        print("\nReproduction chunk completed successfully.")
        print(f"Official result objects returned: {len(results)}")
        print(f"Updated manifest: {manifest_file}")

        if len(results) < len(selected_locations):
            print(
                "Note: fewer results were returned than SAEs selected. "
                "This is expected when existing official JSONs were skipped."
            )

    except Exception as error:
        manifest["status"] = "failed"
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        write_manifest(manifest_file, manifest)
        raise

    finally:
        release_loaded_saes(selected_saes)


if __name__ == "__main__":
    main()
