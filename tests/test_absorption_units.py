"""
Unit tests for the Absorption wrapper (src/saebench_audit/metrics/absorption.py) + its aggregation.

Absorption wraps upstream SAEBench (we run the authors' code), so there is no verbatim oracle here.
These tests pin (a) our config defaults = the SHIPPED upstream constants, (b) that a threshold
override monkeypatches the upstream module correctly, (c) output flattening + per-SAE aggregation.
Tests needing `sae_bench` skip gracefully if it is not importable in this interpreter.
Run: python tests/test_absorption_units.py   (use the sae_bench venv to exercise the skip-guarded ones)
"""
import importlib.util
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from saebench_audit.metrics import absorption as absorp
from saebench_audit.statistics import aggregate_absorption, compare_absorption_to_published


def _has_sae_bench():
    return importlib.util.find_spec("sae_bench") is not None


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_absorption", os.path.join(ROOT, "scripts", "run_absorption.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- config

def test_config_defaults_are_shipped_constants():
    cfg = absorp.AbsorptionConfig()
    assert cfg.absorption_fraction_probe_cos_sim_threshold == 0.1
    assert cfg.full_absorption_probe_cos_sim_threshold == 0.025
    assert cfg.probe_projection_proportion_threshold == 0.4
    assert cfg.absorption_fraction_max_absorbing_latents == 3
    assert cfg.random_seed == 42          # declared but inert upstream
    assert cfg.uses_shipped_thresholds() is True


def test_config_detects_audit_override():
    # Table 8 values (the audit toggle) must be recognized as non-shipped.
    cfg = absorp.AbsorptionConfig(
        absorption_fraction_probe_cos_sim_threshold=-1.0,
        probe_projection_proportion_threshold=0.0,
        absorption_fraction_max_absorbing_latents=4096,
    )
    assert cfg.uses_shipped_thresholds() is False


# --------------------------------------------------------------------------- output + aggregation

def test_flatten_output_extracts_both_scores():
    out = {
        "eval_result_metrics": {"mean": {
            "mean_absorption_fraction_score": 0.12, "mean_full_absorption_score": 0.06,
            "mean_num_split_features": 1.5, "std_dev_absorption_fraction_score": 0.01,
            "std_dev_full_absorption_score": 0.02, "std_dev_num_split_features": 0.3}},
        "eval_result_details": [{"first_letter": "a"}],
    }
    flat = absorp._flatten_output(out)
    assert flat["mean_absorption_fraction_score"] == 0.12
    assert flat["mean_full_absorption_score"] == 0.06
    assert flat["mean_num_split_features"] == 1.5
    assert flat["eval_result_details"][0]["first_letter"] == "a"


def _row(name, arch, frac, full, status="ok"):
    return {"sae_name": name, "arch": arch, "location": f"{arch}/{name}", "status": status,
            "mean_absorption_fraction_score": frac, "mean_full_absorption_score": full}


def test_aggregate_absorption_by_arch_and_status():
    rows = [
        _row("s_t0", "standard", 0.10, 0.05),
        _row("s_t1", "standard", 0.20, 0.15),
        _row("m_t0", "matryoshka", 0.02, 0.01),
        _row("t_t0", "topk", None, None, status="insufficient_features"),
    ]
    agg = aggregate_absorption(rows)
    assert agg["n_saes"] == 4 and agg["n_ok"] == 3 and agg["n_insufficient_features"] == 1
    std = agg["by_arch"]["standard"]["mean_absorption_fraction_score"]
    assert std["n"] == 2 and abs(std["mean"] - 0.15) < 1e-9
    assert "topk" not in agg["by_arch"]           # guard-tripped SAEs excluded from score summaries
    assert len(agg["per_sae"]) == 4               # ...but still listed per-SAE for the record


def test_compare_absorption_to_published():
    agg = aggregate_absorption([_row("s0", "standard", 0.12, 0.06)])
    pub = {"standard": {"mean_absorption_fraction_score": 0.10, "mean_full_absorption_score": 0.05}}
    cmp = compare_absorption_to_published(agg, pub)
    assert abs(cmp["standard"]["mean_absorption_fraction_score"]["abs_delta"] - 0.02) < 1e-9
    assert abs(cmp["standard"]["mean_full_absorption_score"]["abs_delta"] - 0.01) < 1e-9
    assert compare_absorption_to_published(agg, None) == {}


# --------------------------------------------------------------------------- runner helpers

def test_runner_labels_arch_and_sae_name():
    r = _load_runner()
    assert r._arch_from_location("Standard_pythia-160m-deduped__0108/resid_post_layer_8/trainer_0") == "standard"
    assert r._arch_from_location("BatchTopK_x__0108/resid_post_layer_8/trainer_3") == "batchtopk"
    name = r._sae_name("adamkarvonen/saebench_pythia-160m-deduped_width-2pow12_date-0108",
                       "Standard_x__0108/resid_post_layer_8/trainer_0")
    assert name.endswith("Standard_x__0108_resid_post_layer_8_trainer_0")
    assert "/" not in name


# --------------------------------------------------------------------------- sae_bench-gated wiring

def test_threshold_override_monkeypatches_upstream():
    if not _has_sae_bench():
        print("  (skip threshold-override: sae_bench not importable)"); return
    from sae_bench.evals.absorption import feature_absorption as fa
    try:
        assert absorp.apply_threshold_overrides(absorp.AbsorptionConfig()) is False
        assert fa.ABSORPTION_FRACTION_PROBE_COS_THRESHOLD == 0.1
        cfg = absorp.AbsorptionConfig(
            absorption_fraction_probe_cos_sim_threshold=-1.0,
            probe_projection_proportion_threshold=0.0,
            absorption_fraction_max_absorbing_latents=4096)
        assert absorp.apply_threshold_overrides(cfg) is True
        assert fa.ABSORPTION_FRACTION_PROBE_COS_THRESHOLD == -1.0
        assert fa.ABSORPTION_PROBE_PROJECTION_PROPORTION_THRESHOLD == 0.0
        assert fa.ABSORPTION_FRACTION_MAX_ABSORBING_LATENTS == 4096
    finally:
        absorp.apply_threshold_overrides(absorp.AbsorptionConfig())   # restore shipped state


def test_build_eval_config_matches_fields():
    if not _has_sae_bench():
        print("  (skip build_eval_config: sae_bench not importable)"); return
    cfg = absorp.AbsorptionConfig(llm_batch_size=7)
    ec = absorp.build_eval_config(cfg)
    assert ec.model_name == cfg.model_name
    assert ec.llm_batch_size == 7
    assert ec.llm_dtype == cfg.llm_dtype
    assert ec.f1_jump_threshold == cfg.f1_jump_threshold
    assert ec.max_k_value == cfg.max_k_value
    assert ec.min_GT_probe_f1 == cfg.min_GT_probe_f1
    assert ec.min_feats_for_eval == cfg.min_feats_for_eval


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t(); print(f"PASS  {t.__name__}"); passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:
            print(f"ERROR {t.__name__}: {repr(e)[:160]}")
    print(f"\n{passed}/{len(tests)} absorption unit tests passed")
    sys.exit(0 if passed == len(tests) else 1)
