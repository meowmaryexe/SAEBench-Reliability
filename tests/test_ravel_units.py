"""
Unit tests for the RAVEL wrapper (src/saebench_audit/metrics/ravel.py) + its aggregation.

RAVEL wraps upstream SAEBench (we run the authors' code), so there is no verbatim oracle here for the full
eval. These tests pin (a) our config defaults = the shipped upstream settings, (b) output flattening of the
three scores, (c) per-SAE aggregation, (d) the compare-to-published deltas, and (e) runner label helpers.
Tests needing `sae_bench` skip gracefully if it is not importable in this interpreter.
Run: python tests/test_ravel_units.py   (use the sae_bench venv to exercise the skip-guarded one)
"""
import importlib.util
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from saebench_audit.metrics import ravel as rvl
from saebench_audit.statistics import aggregate_ravel, compare_ravel_to_published


def _has_sae_bench():
    return importlib.util.find_spec("sae_bench") is not None


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_ravel", os.path.join(ROOT, "scripts", "run_ravel.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- config
def test_config_defaults_are_shipped_settings():
    cfg = rvl.RavelConfig()
    assert cfg.model_name == "gemma-2-2b" and "it" not in cfg.model_name   # BASE model, not instruct
    assert cfg.top_n_entities == 500
    assert cfg.num_pairs_per_attribute == 5000
    assert cfg.num_epochs == 2
    assert cfg.n_generated_tokens == 6
    assert cfg.train_mdas is False
    assert cfg.random_seed == 42
    assert cfg.llm_batch_size == 8                     # 24 GB default (not upstream's 2048)
    assert set(cfg.entity_attribute_selection) == {"city", "nobel_prize_winner"}
    assert cfg.uses_shipped_settings() is True


def test_config_detects_settings_override():
    assert rvl.RavelConfig(num_epochs=10, train_mdas=True).uses_shipped_settings() is False
    assert rvl.RavelConfig(top_n_entities=100).uses_shipped_settings() is False


# --------------------------------------------------------------------------- output + aggregation
def test_flatten_output_extracts_all_three_scores():
    out = {"eval_result_metrics": {"ravel": {
        "disentanglement_score": 0.62, "cause_score": 0.71, "isolation_score": 0.53}},
        "eval_result_unstructured": {}}
    flat = rvl._flatten_output(out)
    assert flat == {"disentanglement_score": 0.62, "cause_score": 0.71, "isolation_score": 0.53}


def _row(name, arch, dis, cause=0.5, iso=0.5, status="ok"):
    return {"sae_name": name, "arch": arch, "location": f"{arch}/{name}", "status": status,
            "disentanglement_score": dis, "cause_score": cause, "isolation_score": iso}


def test_aggregate_ravel_by_arch_and_status():
    rows = [
        _row("s_t0", "standard", 0.40),
        _row("s_t1", "standard", 0.60),
        _row("m_t0", "matryoshkabatchtopk", 0.80),
        _row("t_t0", "topk", None, status="error"),
    ]
    agg = aggregate_ravel(rows)
    assert agg["n_saes"] == 4 and agg["n_ok"] == 3 and agg["n_failed"] == 1
    std = agg["by_arch"]["standard"]["disentanglement_score"]
    assert std["n"] == 2 and abs(std["mean"] - 0.50) < 1e-9
    assert "topk" not in agg["by_arch"]        # failed SAEs excluded from score summaries
    assert len(agg["per_sae"]) == 4            # ...but still listed per-SAE for the record
    # cause/isolation are summarized too
    assert agg["by_arch"]["standard"]["cause_score"]["n"] == 2


def test_compare_ravel_to_published():
    agg = aggregate_ravel([_row("s0", "standard", 0.55, cause=0.6, iso=0.5)])
    pub = {"standard": {"disentanglement_score": 0.50, "cause_score": 0.58, "isolation_score": 0.42}}
    cmp = compare_ravel_to_published(agg, pub)
    assert abs(cmp["standard"]["disentanglement_score"]["abs_delta"] - 0.05) < 1e-9
    assert abs(cmp["standard"]["isolation_score"]["abs_delta"] - 0.08) < 1e-9
    assert compare_ravel_to_published(agg, None) == {}


# --------------------------------------------------------------------------- runner helpers
def test_runner_labels_arch_and_sae_name():
    r = _load_runner()
    assert r._arch_from_location("Standard_gemma-2-2b__0108/resid_post_layer_12/trainer_0") == "standard"
    assert r._arch_from_location(
        "MatryoshkaBatchTopK_x__0108/resid_post_layer_12/trainer_3") == "matryoshkabatchtopk"
    name = r._sae_name("adamkarvonen/saebench_gemma-2-2b_width-2pow12_date-0108",
                       "Standard_x__0108/resid_post_layer_12/trainer_0")
    assert name.endswith("Standard_x__0108_resid_post_layer_12_trainer_0") and "/" not in name


def test_runner_default_model_is_base():
    r = _load_runner()
    # RAVEL runs on the base model — no base/instruct split logic (unlike run_unlearning).
    assert "gemma-2-2b" in r.DEFAULT_REPO


# --------------------------------------------------------------------------- sae_bench-gated wiring
def test_build_eval_config_matches_fields():
    if not _has_sae_bench():
        print("  (skip build_eval_config: sae_bench not importable)"); return
    from sae_bench.evals.ravel.eval_config import RAVELEvalConfig
    cfg = rvl.RavelConfig(llm_batch_size=7)
    ec = rvl.build_eval_config(cfg)
    assert ec.model_name == cfg.model_name
    assert ec.llm_batch_size == 7
    assert ec.llm_dtype == cfg.llm_dtype
    assert ec.top_n_entities == cfg.top_n_entities
    assert ec.num_epochs == cfg.num_epochs
    assert ec.n_generated_tokens == cfg.n_generated_tokens
    assert ec.random_seed == cfg.random_seed
    assert isinstance(ec, RAVELEvalConfig)


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
    print(f"\n{passed}/{len(tests)} ravel unit tests passed")
    sys.exit(0 if passed == len(tests) else 1)
