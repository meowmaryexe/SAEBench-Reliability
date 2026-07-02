"""
Absorption (first-letter feature absorption) — Stage-1 faithful reproduction. Owner: Alor.

This is a thin **wrapper** around the upstream SAEBench eval (`sae_bench.evals.absorption`,
sae-bench 0.6.0): we run the authors' code end-to-end and package it in this repo's resumable
`run -> aggregate` flow, exposing the metric's undocumented thresholds as config so the audit
phase can toggle them. (This matches the Probe-Rig "wrap, don't reimplement" design spec for the
probe-family metrics; independent reimplementation is deferred to the audit stage.)

Faithfulness (Stage 1) — we use the SHIPPED module constants:
  - absorption-fraction cosine gate = 0.1   (Table 8 says tau_ps = -1)
  - full-absorption cosine gate     = 0.025  (a separate gate, used only for the full score)
  - projection-proportion gate      = 0.4    (Table 8 says tau_pa = 0)
  - max-absorbing-latents           = 3      (Table 8 says A_max = dict size)
These live as module constants in `feature_absorption.py:34-44` and are copied into the
`FeatureAbsorptionCalculator` at `feature_absorption.py:214-217` — so overriding them means
monkeypatching those module names (exactly the audit recipe). The raw per-word DataFrame is cached
by (layer, sae_name) ONLY (not by threshold), so any override must force a rerun.

Upstream declares `random_seed=42` but never applies it (no seed call anywhere; probing.py:276 uses
`random.sample`, DataLoaders/ICL shuffle), so expect **run-to-run drift**. We report both mean scores
and their per-letter std devs.

We report both `mean_absorption_fraction_score` and `mean_full_absorption_score`
(`eval_output.py:19-27`), plus `mean_num_split_features` and the three `std_dev_*` fields.

Runs under the sae_bench venv (sae_bench, sae_lens, transformer_lens). `sae_bench` is imported
lazily inside functions so this module (and its config) import cleanly in the plain repo interpreter.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass

METRIC_NAME = "absorption"

# Shipped upstream thresholds (feature_absorption.py:34-44). Stage-1 uses these verbatim.
SHIPPED_ABSORPTION_FRACTION_COS = 0.1
SHIPPED_FULL_ABSORPTION_COS = 0.025
SHIPPED_PROJECTION_PROPORTION = 0.4
SHIPPED_MAX_ABSORBING_LATENTS = 3

# Field names of the six mean/std metrics upstream emits under eval_result_metrics.mean.
ABSORPTION_MEAN_KEYS = [
    "mean_absorption_fraction_score",
    "mean_full_absorption_score",
    "mean_num_split_features",
    "std_dev_absorption_fraction_score",
    "std_dev_full_absorption_score",
    "std_dev_num_split_features",
]


@dataclass
class AbsorptionConfig:
    """Mirrors upstream `AbsorptionEvalConfig`, plus the four thresholds it hardcodes as module
    constants (exposed here so the audit phase can toggle them). Defaults reproduce the paper."""

    # model / SAE (upstream loads the model itself via transformer_lens `from_pretrained_no_processing`)
    model_name: str = "pythia-160m-deduped"   # transformer_lens name for the anchor Pythia model
    llm_dtype: str = "float32"
    llm_batch_size: int = 32                   # inference batching only — does NOT affect the metric
    device: str = "cpu"                        # the CPU green-pipeline milestone

    # upstream AbsorptionEvalConfig fields (defaults = eval_config.py)
    random_seed: int = 42                       # declared but INERT upstream -> run-to-run drift
    f1_jump_threshold: float = 0.03
    max_k_value: int = 10
    prompt_template: str = "{word} has the first letter:"
    prompt_token_pos: int = -6
    min_GT_probe_f1: float = 0.6
    min_feats_for_eval: int = 20
    k_sparse_probe_l1_decay: float = 0.01
    k_sparse_probe_batch_size: int = 4096
    k_sparse_probe_num_epochs: int = 50
    eval_k_sparse_probe_batch_size: int = 24

    # the four hardcoded absorption thresholds (SHIPPED defaults; the audit phase overrides these)
    absorption_fraction_probe_cos_sim_threshold: float = SHIPPED_ABSORPTION_FRACTION_COS
    full_absorption_probe_cos_sim_threshold: float = SHIPPED_FULL_ABSORPTION_COS
    probe_projection_proportion_threshold: float = SHIPPED_PROJECTION_PROPORTION
    absorption_fraction_max_absorbing_latents: int = SHIPPED_MAX_ABSORBING_LATENTS

    def to_dict(self) -> dict:
        return asdict(self)

    def uses_shipped_thresholds(self) -> bool:
        """True iff all four thresholds are at the shipped values (Stage-1 faithful reproduction)."""
        return (
            self.absorption_fraction_probe_cos_sim_threshold == SHIPPED_ABSORPTION_FRACTION_COS
            and self.full_absorption_probe_cos_sim_threshold == SHIPPED_FULL_ABSORPTION_COS
            and self.probe_projection_proportion_threshold == SHIPPED_PROJECTION_PROPORTION
            and self.absorption_fraction_max_absorbing_latents == SHIPPED_MAX_ABSORBING_LATENTS
        )


def build_eval_config(cfg: AbsorptionConfig):
    """Translate our AbsorptionConfig into the upstream `AbsorptionEvalConfig`."""
    from sae_bench.evals.absorption.eval_config import AbsorptionEvalConfig

    eval_cfg = AbsorptionEvalConfig(
        model_name=cfg.model_name,
        random_seed=cfg.random_seed,
        f1_jump_threshold=cfg.f1_jump_threshold,
        max_k_value=cfg.max_k_value,
        prompt_template=cfg.prompt_template,
        prompt_token_pos=cfg.prompt_token_pos,
        min_GT_probe_f1=cfg.min_GT_probe_f1,
        min_feats_for_eval=cfg.min_feats_for_eval,
        k_sparse_probe_l1_decay=cfg.k_sparse_probe_l1_decay,
        k_sparse_probe_batch_size=cfg.k_sparse_probe_batch_size,
        k_sparse_probe_num_epochs=cfg.k_sparse_probe_num_epochs,
        eval_k_sparse_probe_batch_size=cfg.eval_k_sparse_probe_batch_size,
    )
    eval_cfg.llm_batch_size = cfg.llm_batch_size
    eval_cfg.llm_dtype = cfg.llm_dtype
    return eval_cfg


def apply_threshold_overrides(cfg: AbsorptionConfig) -> bool:
    """Monkeypatch the four hardcoded thresholds onto the upstream `feature_absorption` module
    (they are copied into the calculator at feature_absorption.py:214-217). Returns True if any
    value differs from the shipped defaults, in which case the caller MUST force a rerun because
    the raw absorption DataFrame is cached by (layer, sae_name) only, not by threshold."""
    from sae_bench.evals.absorption import feature_absorption as fa

    fa.ABSORPTION_FRACTION_PROBE_COS_THRESHOLD = cfg.absorption_fraction_probe_cos_sim_threshold
    fa.FULL_ABSORPTION_PROBE_COS_THRESHOLD = cfg.full_absorption_probe_cos_sim_threshold
    fa.ABSORPTION_PROBE_PROJECTION_PROPORTION_THRESHOLD = cfg.probe_projection_proportion_threshold
    fa.ABSORPTION_FRACTION_MAX_ABSORBING_LATENTS = cfg.absorption_fraction_max_absorbing_latents
    return not cfg.uses_shipped_thresholds()


def _flatten_output(out: dict) -> dict:
    """Pull the six mean/std metrics + per-letter details out of an AbsorptionEvalOutput dict."""
    mean = out["eval_result_metrics"]["mean"]
    flat = {k: mean.get(k) for k in ABSORPTION_MEAN_KEYS}
    flat["eval_result_details"] = out.get("eval_result_details", [])
    return flat


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
        trainer_class = json.load(f)["trainer"]["trainer_class"]
    if trainer_class not in TRAINER_LOADERS:
        raise ValueError(
            f"Unknown trainer_class {trainer_class!r}; known: {sorted(TRAINER_LOADERS)}"
        )
    return TRAINER_LOADERS[trainer_class](
        repo_id=repo_id,
        filename=f"{location}/ae.pt",
        layer=None,
        model_name=model_name,
        device=device,
        dtype=general_utils.str_to_dtype(dtype),
    )


def sae_result_path(workdir: str, sae_name: str) -> str:
    """Path where upstream writes this SAE's result JSON (used for resume/skip)."""
    from sae_bench.sae_bench_utils import general_utils

    return general_utils.get_results_filepath(workdir, sae_name, "custom_sae")


def run_absorption(
    cfg: AbsorptionConfig,
    sae,
    sae_name: str,
    workdir: str,
    force_rerun: bool = False,
    verbose: bool = True,
) -> dict:
    """Run the upstream absorption eval on a single (already-loaded) SAE, writing its result JSON
    into `workdir`. Returns a flat summary dict with both mean scores + std devs + per-letter
    details, or `{"status": "insufficient_features"}` if the SAE trips the min-features guard."""
    from sae_bench.evals.absorption import main as absorption_main

    os.makedirs(workdir, exist_ok=True)

    if apply_threshold_overrides(cfg) and not force_rerun:
        force_rerun = True  # cached raw df is keyed by (layer, sae_name) only, not by threshold
        if verbose:
            print("[absorption] non-shipped thresholds set -> forcing rerun (audit mode)")

    eval_cfg = build_eval_config(cfg)
    results = absorption_main.run_eval(
        eval_cfg,
        [(sae_name, sae)],
        device=cfg.device,
        output_path=workdir,
        force_rerun=force_rerun,
    )

    key = f"{sae_name}_custom_sae"
    if key in results:
        return {"status": "ok", "sae_name": sae_name, **_flatten_output(results[key])}

    # run_eval either skipped an already-complete SAE or `break`d on insufficient features.
    out_path = sae_result_path(workdir, sae_name)
    if os.path.exists(out_path):
        import json

        with open(out_path) as f:
            return {"status": "ok", "sae_name": sae_name, **_flatten_output(json.load(f))}
    return {"status": "insufficient_features", "sae_name": sae_name}
