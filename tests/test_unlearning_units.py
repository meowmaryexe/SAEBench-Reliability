"""
Unit tests for the Unlearning wrapper (src/saebench_audit/metrics/unlearning.py) + its aggregation.

Unlearning wraps upstream SAEBench (we run the authors' code), so there is no verbatim oracle here for the
full eval. These tests pin (a) our config defaults = the shipped upstream sweep grid, (b) output
flattening, (c) per-SAE aggregation, (d) runner helpers, and (e) the gated-corpus preflight. Tests needing
`sae_bench` skip gracefully if it is not importable in this interpreter.
Run: python tests/test_unlearning_units.py   (use the sae_bench venv to exercise the skip-guarded ones)
"""
import importlib.util
import os
import sys
import tempfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from saebench_audit.metrics import unlearning as unl
from saebench_audit.statistics import aggregate_unlearning, compare_unlearning_to_published


def _has_sae_bench():
    return importlib.util.find_spec("sae_bench") is not None


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_unlearning", os.path.join(ROOT, "scripts", "run_unlearning.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- config
def test_config_defaults_are_shipped_grid():
    cfg = unl.UnlearningConfig()
    assert cfg.model_name == "gemma-2-2b-it" and "it" in cfg.model_name   # upstream requires instruct
    assert cfg.retain_thresholds == [0.001, 0.01]
    assert cfg.n_features_list == [10, 20]
    assert cfg.multipliers == [25, 50, 100, 200]
    assert cfg.random_seed == 42
    assert cfg.uses_shipped_grid() is True
    # 2 x 2 x 4 = the shipped 16-config sweep
    assert len(cfg.retain_thresholds) * len(cfg.n_features_list) * len(cfg.multipliers) == 16


def test_config_detects_grid_override():
    cfg = unl.UnlearningConfig(multipliers=[25, 50])
    assert cfg.uses_shipped_grid() is False


# --------------------------------------------------------------------------- output + aggregation
def test_flatten_output_extracts_score():
    out = {"eval_result_metrics": {"unlearning": {"unlearning_score": 0.037}}, "eval_result_details": []}
    flat = unl._flatten_output(out)
    assert flat == {"unlearning_score": 0.037}


def _row(name, arch, score, status="ok"):
    return {"sae_name": name, "arch": arch, "location": f"{arch}/{name}", "status": status,
            "unlearning_score": score}


def test_aggregate_unlearning_by_arch_and_status():
    rows = [
        _row("s_t0", "standard", 0.02),
        _row("s_t1", "standard", 0.06),
        _row("m_t0", "matryoshka", 0.10),
        _row("t_t0", "topk", None, status="error"),
    ]
    agg = aggregate_unlearning(rows)
    assert agg["n_saes"] == 4 and agg["n_ok"] == 3 and agg["n_failed"] == 1
    std = agg["by_arch"]["standard"]["unlearning_score"]
    assert std["n"] == 2 and abs(std["mean"] - 0.04) < 1e-9
    assert "topk" not in agg["by_arch"]        # failed SAEs excluded from score summaries
    assert len(agg["per_sae"]) == 4            # ...but still listed per-SAE for the record


def test_compare_unlearning_to_published():
    agg = aggregate_unlearning([_row("s0", "standard", 0.05)])
    pub = {"standard": {"unlearning_score": 0.04}}
    cmp = compare_unlearning_to_published(agg, pub)
    assert abs(cmp["standard"]["unlearning_score"]["abs_delta"] - 0.01) < 1e-9
    assert compare_unlearning_to_published(agg, None) == {}


# --------------------------------------------------------------------------- runner helpers + preflight
def test_runner_labels_arch_and_sae_name():
    r = _load_runner()
    assert r._arch_from_location("Standard_gemma-2-2b__0108/resid_post_layer_12/trainer_0") == "standard"
    assert r._arch_from_location("BatchTopK_x__0108/resid_post_layer_12/trainer_3") == "batchtopk"
    name = r._sae_name("adamkarvonen/saebench_gemma-2-2b_width-2pow12_date-0108",
                       "Standard_x__0108/resid_post_layer_12/trainer_0")
    assert name.endswith("Standard_x__0108_resid_post_layer_12_trainer_0") and "/" not in name


def test_require_forget_corpus_missing_raises_actionable():
    with tempfile.TemporaryDirectory() as d:
        try:
            unl.require_forget_corpus(cwd=d)
            assert False, "expected FileNotFoundError"
        except FileNotFoundError as e:
            assert "bio-forget-corpus" in str(e)


def test_require_forget_corpus_present_ok():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, unl.FORGET_CORPUS_RELPATH)
        os.makedirs(os.path.dirname(p))
        open(p, "w").write("{}\n")
        assert unl.require_forget_corpus(cwd=d) == p


# --------------------------------------------------------------------------- sae_bench-gated wiring
def test_build_eval_config_matches_fields():
    if not _has_sae_bench():
        print("  (skip build_eval_config: sae_bench not importable)"); return
    from sae_bench.evals.unlearning.eval_config import UnlearningEvalConfig
    cfg = unl.UnlearningConfig(llm_batch_size=7)
    ec = unl.build_eval_config(cfg)
    assert ec.model_name == cfg.model_name
    assert ec.llm_batch_size == 7
    assert ec.llm_dtype == cfg.llm_dtype
    assert list(ec.multipliers) == cfg.multipliers
    assert list(ec.retain_thresholds) == cfg.retain_thresholds
    assert ec.random_seed == cfg.random_seed
    assert isinstance(ec, UnlearningEvalConfig)


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
    print(f"\n{passed}/{len(tests)} unlearning unit tests passed")
    sys.exit(0 if passed == len(tests) else 1)
