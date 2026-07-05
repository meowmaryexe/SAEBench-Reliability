"""
SAE suite resolution + discovery + release loading, shared by the wrap-upstream metrics.

Given a registry suite name (e.g. "pythia-160m_4k") this resolves the model/layer/HF-repo, enumerates
every released SAE in the repo, labels each by architecture, and loads it into a sae_lens-compatible
object for an upstream `run_eval`. `configs/registry.yaml` is the single source of truth.

`sae_bench` / `huggingface_hub` / `yaml` are imported lazily inside functions so this module imports
under the plain repo interpreter (the metric unit tests rely on that).
"""
from __future__ import annotations

import os

# repo root = two levels up from src/saebench_audit/suites.py
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def resolve_suite(name: str, root: str = REPO_ROOT) -> dict:
    """Resolve a registry suite name to {model_name, layer, sae_repo}.

    This is how a runner switches models/widths — Pythia now, Gemma later — with one `--suite` flag; the
    Gemma suites are already defined in configs/registry.yaml, so no code change is needed to run them.
    """
    import yaml

    reg = yaml.safe_load(open(os.path.join(root, "configs", "registry.yaml")))
    suites = reg.get("sae_suites", {})
    if name not in suites:
        raise SystemExit(f"Unknown suite {name!r}. Available: {sorted(suites)}")
    s = suites[name]
    model = s.get("model") or reg.get("models", {}).get(s.get("modeltag"), {}).get("tl_name")
    if not model:
        raise SystemExit(f"Suite {name!r} has no `model` field in registry.yaml; add one to run it.")
    return {"model_name": model, "layer": int(s["layer"]), "sae_repo": s["hf_repo"]}


def arch_from_location(location: str) -> str:
    """Label a SAE by the leading token of its top folder, e.g. 'Standard_..._0108/...' -> 'standard'."""
    top = location.strip("/").split("/")[0]
    return top.split("_")[0].lower()


def sae_name(repo_id: str, location: str) -> str:
    """Stable, filesystem-safe SAE name from repo + in-repo folder (matches published filename stems)."""
    return f"{repo_id.split('/')[-1]}_{location.strip('/').replace('/', '_')}"


def discover_locations(repo_id: str, layer: int):
    """All SAE folders in the repo (containing ae.pt + config.json), filtered to `layer`, no checkpoints."""
    from sae_bench.custom_saes.run_all_evals_dictionary_learning_saes import (
        get_all_hf_repo_autoencoders,
    )
    from sae_bench.sae_bench_utils import general_utils

    locs = get_all_hf_repo_autoencoders(repo_id)
    locs = general_utils.filter_keywords(locs, exclude_keywords=["checkpoints"], include_keywords=[])
    return sorted(loc for loc in locs if f"resid_post_layer_{layer}" in loc)


def load_released_sae(
    repo_id: str,
    location: str,
    model_name: str = "pythia-160m-deduped",
    device: str = "cpu",
    dtype: str = "float32",
    download_location: str = "downloaded_saes",
):
    """Load a released dictionary_learning SAE into a sae_lens-compatible object for `run_eval`.

    `location` is the in-repo folder holding `ae.pt` + `config.json`
    (e.g. "Standard_.../resid_post_layer_8/trainer_0"). Reads the config to pick the right
    upstream trainer loader (TRAINER_LOADERS), which downloads + formats the weights (HF-cached).
    """
    import json

    from huggingface_hub import hf_hub_download

    from sae_bench.custom_saes.run_all_evals_dictionary_learning_saes import TRAINER_LOADERS
    from sae_bench.sae_bench_utils import general_utils

    location = location.rstrip("/")
    cfg_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"{location}/config.json",
        force_download=False,
        local_dir=download_location,
    )
    with open(cfg_path) as f:
        config = json.load(f)
    trainer_class = config["trainer"]["trainer_class"]
    # Tolerate the sae-bench 0.3.2 loader typo ('Matroyshka' vs 'Matryoshka') and minor renames:
    # pick the loader whose key matches the reported name or a Matryoshka<->Matroyshka respelling.
    candidates = [trainer_class,
                  trainer_class.replace("Matryoshka", "Matroyshka"),
                  trainer_class.replace("Matroyshka", "Matryoshka")]
    key = next((c for c in candidates if c in TRAINER_LOADERS), None)
    if key is None:
        raise ValueError(
            f"Unknown trainer_class {trainer_class!r}; known: {sorted(TRAINER_LOADERS)}"
        )
    if key != trainer_class:
        # 0.3.2's Matryoshka loader *also* asserts the (typo'd) trainer_class name internally after
        # building the SAE, raising otherwise. The SAE weights are identical; only this metadata string
        # differs. Patch the cached config in place so the loader accepts it — verified the loader re-reads
        # this cached file (force_download=False) without clobbering the edit.
        config["trainer"]["trainer_class"] = key
        with open(cfg_path, "w") as f:
            json.dump(config, f)
    return TRAINER_LOADERS[key](
        repo_id=repo_id,
        filename=f"{location}/ae.pt",
        layer=None,
        model_name=model_name,
        device=device,
        dtype=general_utils.str_to_dtype(dtype),
    )


def sae_result_path(workdir: str, name: str) -> str:
    """Path where upstream writes this SAE's result JSON (used for resume/skip)."""
    from sae_bench.sae_bench_utils import general_utils

    return general_utils.get_results_filepath(workdir, name, "custom_sae")
