"""
Quick, dependency-free reproduction of the SAEBench absorption_fraction VERSION-DRIFT finding on the
released Pythia-160M 4k Standard SAE (trainer_0). No model / SAE / GPU / sae_bench needed — runs anywhere.

Two parts:
  1. MECHANISM (synthetic): the 0.3.2 vs 0.6.0 fraction *formulas* on the same word population — the
     0.6.0 cos-gate + top-k cap + proportion floor collapse diffuse absorption the 0.3.2 form counts.
  2. EVIDENCE (committed fixtures): real per-SAE result JSONs show the published fraction reproduces
     under the release version (0.3.2) and collapses ~10x under current code (0.6.0), while the
     full-absorption score reproduces under both.

Full story + regeneration commands: docs/findings/absorption_version_drift.md
Run: python tests/test_absorption_version_drift.py   (or: pytest -q tests/test_absorption_version_drift.py)
"""
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from saebench_audit.metrics.absorption_fraction import (
    absorption_fraction_v032,
    absorption_fraction_v060,
)

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "absorption")
BAND = 0.05  # drift-aware "reproduced" band (upstream applies no seed); see docs/preregistration.md


def _mean(fixture):
    m = json.load(open(os.path.join(FIX, fixture)))["eval_result_metrics"]["mean"]
    return m["mean_absorption_fraction_score"], m["mean_full_absorption_score"]


# --------------------------------------------------------------------------- mechanism (synthetic)

def _diffuse_word():
    """Absorption spread over many WEAKLY probe-aligned non-main latents (cos < 0.1)."""
    return dict(act=1.0, main=0.1,
                cand_proj=[0.04] * 20,          # sum 0.8 of projection carried by non-main latents
                cand_cos=[0.05] * 20)           # ...but all below the 0.6.0 cos gate of 0.1


def _concentrated_word():
    """Absorption carried by a single STRONGLY aligned non-main latent (cos >= 0.1)."""
    return dict(act=1.0, main=0.1, cand_proj=[0.7, 0.03, 0.02], cand_cos=[0.5, 0.04, 0.03])


def test_mechanism_single_word_diffuse():
    w = _diffuse_word()
    all_proj = w["main"] + sum(w["cand_proj"])
    v032 = absorption_fraction_v032(w["act"], w["main"], all_proj)
    v060 = absorption_fraction_v060(w["act"], w["main"], w["cand_proj"], w["cand_cos"])
    assert v032 > 0.8, v032          # 0.3.2 counts all the diffuse leakage
    assert v060 == 0.0, v060         # 0.6.0 gates it all out (cos 0.05 < 0.1)


def test_mechanism_population_v032_dominates_v060():
    pop = [_diffuse_word()] * 4 + [_concentrated_word()]
    v032 = [absorption_fraction_v032(w["act"], w["main"], w["main"] + sum(w["cand_proj"])) for w in pop]
    v060 = [absorption_fraction_v060(w["act"], w["main"], w["cand_proj"], w["cand_cos"]) for w in pop]
    m032, m060 = sum(v032) / len(v032), sum(v060) / len(v060)
    assert m032 > 0.5, m032
    assert m060 < 0.3, m060
    assert m032 >= 3 * m060, (m032, m060)   # same inputs, ~>=3x from the formula change alone


def test_mechanism_concentrated_word_is_similar():
    """Sanity: when absorption IS concentrated + aligned, the two formulas roughly agree — so the
    population gap above is due to diffuse leakage, not a blanket offset."""
    w = _concentrated_word()
    v032 = absorption_fraction_v032(w["act"], w["main"], w["main"] + sum(w["cand_proj"]))
    v060 = absorption_fraction_v060(w["act"], w["main"], w["cand_proj"], w["cand_cos"])
    assert abs(v032 - v060) < 0.1, (v032, v060)


# --------------------------------------------------------------------------- evidence (real SAE fixtures)

def test_full_absorption_reproduces_under_both_versions():
    pub_frac, pub_full = _mean("published_0125_standard_4k_t0.json")
    _, v032_full = _mean("ours_v0.3.2_standard_4k_t0.json")
    _, v060_full = _mean("ours_v0.6.0_standard_4k_t0.json")
    assert abs(v032_full - pub_full) < BAND, (v032_full, pub_full)   # full stable @ 0.3.2
    assert abs(v060_full - pub_full) < BAND, (v060_full, pub_full)   # full stable @ 0.6.0 too


def test_fraction_reproduces_under_release_version_0_3_2():
    pub_frac, _ = _mean("published_0125_standard_4k_t0.json")
    v032_frac, _ = _mean("ours_v0.3.2_standard_4k_t0.json")
    assert abs(v032_frac - pub_frac) < BAND, (v032_frac, pub_frac)   # 0.164 vs 0.155


def test_fraction_collapses_under_current_version_0_6_0():
    pub_frac, _ = _mean("published_0125_standard_4k_t0.json")
    v060_frac, _ = _mean("ours_v0.6.0_standard_4k_t0.json")
    assert v060_frac < 0.5 * pub_frac, (v060_frac, pub_frac)         # ~10x low (0.016 vs 0.155)


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
    print(f"\n{passed}/{len(tests)} absorption version-drift tests passed")
    sys.exit(0 if passed == len(tests) else 1)
