"""
RAVEL (disentanglement) — Stage-1 faithful reproduction. Owner: Alor.

Thin **wrapper** around the upstream SAEBench eval (`sae_bench.evals.ravel`): we run the authors'
code end-to-end and package it in this repo's resumable `run -> aggregate` flow, reusing the shared
scaffolding (`saebench_audit.{provenance,suites,runner,run_record}`). This matches the "wrap, don't
reimplement" spec for the probe-family metrics; independent reimplementation is deferred to the audit.

What RAVEL measures (Appendix D / Table 7): for each entity type it trains a per-attribute **MDBM**
(a binary mask over SAE latents, SGD-trained — explicitly *not* the linear probe of the original Huang
method) that isolates one attribute (the "cause") from the others (the "iso"), then intervenes and
generates. The headline is the **disentanglement score = mean(cause_score, isolation_score)**, averaged
over attributes and entity types. `add_error` is hardcoded `False` upstream (the reconstruction-error term
is dropped — un-toggleable without a source edit; this is a Stage-2 audit item, not a reproduction knob).

Model is **base `gemma-2-2b`** (NOT the instruct model — RAVEL is a base-LM completion task, unlike
unlearning). The intervention layer is read from the SAE itself (`sae.cfg.hook_layer`), so there is no
layer field in the eval config. The prompt data (`adamkarvonen/ravel_prompts`) is **public** and
auto-downloaded — there is no gated corpus (unlike unlearning); only the `gemma-2-2b` model is gated
(accept its HF license + `huggingface-cli login`).

Runs under the sae_bench venv + GPU. A one-time per-model dataset generation/filter pass is cached under
`artifact_dir` and reused across every SAE and resume. `sae_bench` is imported lazily so this module
imports cleanly in the plain repo interpreter.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field

# Shared scaffolding, re-exported so `ravel.<name>` stays the single import surface (tests + scripts).
from ..provenance import build_upstream_config, ensure_torch_load_shim, environment_provenance, installed_version
from ..suites import load_released_sae, sae_result_path

# The headline metric(s) upstream emits under eval_result_metrics.ravel (eval_output.py: RAVELMetricResults).
# disentanglement_score = mean(cause_score, isolation_score) — the primary ranking column.
RAVEL_SCORE_KEYS = ("disentanglement_score", "cause_score", "isolation_score")

# Default entity/attribute selection = the shipped RAVEL config (eval_config.py): the two entity types the
# published eval scores, each with three attributes.
DEFAULT_ENTITY_ATTRIBUTES = {
    "city": ["Country", "Continent", "Language"],
    "nobel_prize_winner": ["Country of Birth", "Field", "Gender"],
}


@dataclass
class RavelConfig:
    """Mirrors upstream `RAVELEvalConfig`. Defaults reproduce the paper's shipped RAVEL settings.

    Note the upstream *default* `llm_batch_size` is 2048 (never auto-reduced by `run_eval`); we default to
    8 for a 24 GB card (the CLI helper reduces it the same way — LLM_NAME_TO_BATCH_SIZE//4 = 8 for gemma).
    `add_error` is NOT here — upstream hardcodes it False (the dropped error term is a Stage-2 audit item).
    There is no `layer` field: the intervention layer comes from the SAE's own cfg.hook_layer.
    """

    # model / SAE. Upstream loads the model itself (AutoModelForCausalLM, eager attention for gemma).
    model_name: str = "gemma-2-2b"              # BASE model (no "-it"); RAVEL is a base-LM completion task
    llm_dtype: str = "bfloat16"
    llm_batch_size: int = 8                     # 24 GB default (upstream default 2048 is not auto-reduced here)
    device: str = "cuda"

    # upstream RAVELEvalConfig fields (defaults = eval_config.py)
    entity_attribute_selection: dict = field(default_factory=lambda: {
        "city": ["Country", "Continent", "Language"],
        "nobel_prize_winner": ["Country of Birth", "Field", "Gender"]})
    top_n_entities: int = 500
    top_n_templates: int = 90                   # defined upstream but only entities are actually filtered
    full_dataset_downsample: int | None = None
    num_pairs_per_attribute: int = 5000
    train_test_split: float = 0.7
    force_dataset_recompute: bool = False
    learning_rate: float = 1e-3                 # MDBM Adam lr
    num_epochs: int = 2                         # MDBM training epochs (full fwd+bwd through the LLM per step)
    train_mdas: bool = False                    # False = mask the SAE (MDBM); True = the MDAS skyline baseline
    n_generated_tokens: int = 6                 # paper used 8; code ships 6
    random_seed: int = 42                       # applied upstream (numpy/torch)
    artifact_dir: str = "artifacts/ravel"       # dataset filtered-cache root (reused across SAEs + widths)

    def to_dict(self) -> dict:
        return asdict(self)

    def uses_shipped_settings(self) -> bool:
        """True iff the entity/attribute selection + core hyperparameters are the shipped defaults."""
        return (self.entity_attribute_selection == DEFAULT_ENTITY_ATTRIBUTES
                and self.top_n_entities == 500
                and self.num_pairs_per_attribute == 5000
                and self.num_epochs == 2
                and self.n_generated_tokens == 6
                and self.train_mdas is False)


def build_eval_config(cfg: RavelConfig):
    """Translate our RavelConfig into the upstream `RAVELEvalConfig` (version-safe: only fields present on
    the installed dataclass are passed). `llm_batch_size`/`llm_dtype` are real fields on the upstream
    config, so they go in `wanted` (no extra_attrs needed)."""
    from sae_bench.evals.ravel.eval_config import RAVELEvalConfig

    wanted = {
        "model_name": cfg.model_name,
        "llm_dtype": cfg.llm_dtype,
        "llm_batch_size": cfg.llm_batch_size,
        "entity_attribute_selection": dict(cfg.entity_attribute_selection),
        "top_n_entities": cfg.top_n_entities,
        "top_n_templates": cfg.top_n_templates,
        "full_dataset_downsample": cfg.full_dataset_downsample,
        "num_pairs_per_attribute": cfg.num_pairs_per_attribute,
        "train_test_split": cfg.train_test_split,
        "force_dataset_recompute": cfg.force_dataset_recompute,
        "learning_rate": cfg.learning_rate,
        "num_epochs": cfg.num_epochs,
        "train_mdas": cfg.train_mdas,
        "n_generated_tokens": cfg.n_generated_tokens,
        "random_seed": cfg.random_seed,
        "artifact_dir": cfg.artifact_dir,
    }
    return build_upstream_config(RAVELEvalConfig, wanted)


def installed_sae_bench_version() -> str:
    """The installed sae-bench version string (recorded in run_meta so runs are attributable)."""
    return installed_version("sae-bench")


def _flatten_output(out: dict) -> dict:
    """Pull the RAVEL headline scores out of a RAVELEvalOutput dict (eval_result_metrics.ravel)."""
    m = out["eval_result_metrics"]["ravel"]
    return {k: m[k] for k in RAVEL_SCORE_KEYS}


def _run_eval_version_safe(eval_cfg, sae_name, sae, cfg, workdir, force_rerun):
    """Call upstream `run_eval`, passing `artifacts_path` only if the installed signature has it. RAVEL's
    run_eval has no `clean_up_artifacts` kwarg (unlike unlearning). The filtered-dataset cache lives under
    `eval_cfg.artifact_dir`, which persists across SAEs/resumes independently of `artifacts_path`."""
    import inspect

    from sae_bench.evals.ravel import main as ravel_main

    kwargs = dict(force_rerun=force_rerun)
    if "artifacts_path" in inspect.signature(ravel_main.run_eval).parameters:
        kwargs["artifacts_path"] = os.path.join(workdir, "artifacts")
    return ravel_main.run_eval(eval_cfg, [(sae_name, sae)], cfg.device, workdir, **kwargs)


def run_ravel(
    cfg: RavelConfig,
    sae,
    sae_name: str,
    workdir: str,
    force_rerun: bool = False,
    verbose: bool = True,
) -> dict:
    """Run the upstream RAVEL eval on a single (already-loaded) SAE, writing its result JSON into `workdir`.
    Returns a flat summary dict with disentanglement/cause/isolation scores, or a non-ok status row."""
    ensure_torch_load_shim()
    os.makedirs(workdir, exist_ok=True)

    eval_cfg = build_eval_config(cfg)
    results = _run_eval_version_safe(eval_cfg, sae_name, sae, cfg, workdir, force_rerun)

    key = f"{sae_name}_custom_sae"
    if key in results:
        return {"status": "ok", "sae_name": sae_name, **_flatten_output(results[key])}

    # run_eval skipped an already-complete SAE (result already on disk) — read it back.
    out_path = sae_result_path(workdir, sae_name)
    if os.path.exists(out_path):
        import json

        with open(out_path) as f:
            return {"status": "ok", "sae_name": sae_name, **_flatten_output(json.load(f))}
    return {"status": "error", "sae_name": sae_name, "error": "run_eval produced no result and no output file"}
