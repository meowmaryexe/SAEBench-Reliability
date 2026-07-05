"""
Unit tests for the shared wrap-upstream scaffolding (saebench_audit.{provenance,suites,runner,run_record}
+ the generic statistics helpers). These are what Absorption uses today and what Unlearning/RAVEL will
reuse, so they're pinned here independently of any one metric.

Fully dependency-free — no torch, no sae_bench, no network — so a newcomer can `python
tests/test_scaffolding_units.py` (or pytest) under the plain repo interpreter and see them pass.
"""
import json
import os
import sys
import tempfile
from dataclasses import dataclass

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from saebench_audit import runner, suites
from saebench_audit.provenance import build_upstream_config, environment_provenance, installed_version
from saebench_audit.run_record import write_run_record
from saebench_audit.statistics import (aggregate_by_arch, compare_by_arch_to_reference, ranks, spearman,
                                       summary)


# --------------------------------------------------------------------------- provenance
@dataclass
class _Old:  # stands in for an older pinned upstream config missing newer fields
    a: int = 1
    b: int = 2


def test_build_upstream_config_drops_absent_fields_and_sets_extras():
    # `c` is not a field of _Old -> must be dropped (would raise if passed); `extra` set post-hoc.
    inst = build_upstream_config(_Old, {"a": 10, "c": 999}, {"extra": "hi"})
    assert inst.a == 10 and inst.b == 2
    assert not hasattr(inst, "c")
    assert inst.extra == "hi"


def test_installed_version_unknown_is_graceful():
    assert installed_version("definitely-not-a-real-package-xyz") == "unknown"


def test_environment_provenance_has_expected_keys():
    p = environment_provenance(device="cpu", packages=["torch"])
    for k in ("timestamp_utc", "python", "hostname", "device", "packages", "git_sha", "gpu"):
        assert k in p
    assert p["device"] == "cpu" and "torch" in p["packages"]


# --------------------------------------------------------------------------- statistics
def test_summary_mean_std_n():
    s = summary([1.0, 2.0, 3.0, None])
    assert s["n"] == 3 and abs(s["mean"] - 2.0) < 1e-9 and abs(s["std"] - 1.0) < 1e-9
    assert summary([None]) is None


def test_ranks_handles_ties():
    assert ranks([10, 20, 30]) == [1.0, 2.0, 3.0]
    assert ranks([5, 5, 9]) == [1.5, 1.5, 3.0]   # average-tie ranks


def test_spearman_perfect_and_inverse():
    assert abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-9
    assert abs(spearman([1, 2, 3, 4], [40, 30, 20, 10]) + 1.0) < 1e-9


def _row(name, arch, x, status="ok"):
    return {"sae_name": name, "arch": arch, "status": status, "x": x}


def test_aggregate_by_arch_groups_and_excludes():
    rows = [_row("a0", "std", 0.1), _row("a1", "std", 0.3),
            _row("b0", "top", 0.2), _row("c0", "top", None, status="error")]
    agg = aggregate_by_arch(rows, ["x"], ("sae_name", "arch", "status", "x"),
                            excluded_label="n_bad")
    assert agg["n_saes"] == 4 and agg["n_ok"] == 3 and agg["n_bad"] == 1
    std = agg["by_arch"]["std"]["x"]
    assert std["n"] == 2 and abs(std["mean"] - 0.2) < 1e-9
    assert "top" in agg["by_arch"] and len(agg["per_sae"]) == 4


def test_compare_by_arch_to_reference():
    agg = aggregate_by_arch([_row("a0", "std", 0.12)], ["x"], ("sae_name", "arch", "status", "x"))
    cmp = compare_by_arch_to_reference(agg, {"std": {"x": 0.10}}, ["x"])
    assert abs(cmp["std"]["x"]["abs_delta"] - 0.02) < 1e-9
    assert compare_by_arch_to_reference(agg, None, ["x"]) == {}


# --------------------------------------------------------------------------- suites (pure helpers)
def test_suite_label_helpers():
    assert suites.arch_from_location("BatchTopK_x__0108/resid_post_layer_8/trainer_3") == "batchtopk"
    name = suites.sae_name("org/repo_name", "Standard_x/resid_post_layer_8/trainer_0")
    assert name == "repo_name_Standard_x_resid_post_layer_8_trainer_0" and "/" not in name


# --------------------------------------------------------------------------- runner
def test_ledger_roundtrip_and_key():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "l.jsonl")
        runner._append_ledger(p, {"sae_name": "s0", "v": 1})
        runner._append_ledger(p, {"sae_name": "s1", "v": 2})
        done = runner._load_ledger(p, "sae_name")
        assert set(done) == {"s0", "s1"} and done["s1"]["v"] == 2


def test_run_sae_suite_resumes_and_isolates_errors():
    calls = []

    def process_one(location, name):
        calls.append(name)
        if name == "y":
            raise RuntimeError("boom")
        return {"status": "ok", "score": len(name)}

    with tempfile.TemporaryDirectory() as d:
        locs = ["a/x", "b/y", "c/z"]
        common = dict(workdir=d, process_one=process_one, name_fn=lambda l: l.split("/")[1],
                      row_extra_fn=lambda l, n: {"arch": l.split("/")[0]}, ledger_name="m.jsonl")
        done = runner.run_sae_suite(locs, **common)
        assert set(done) == {"x", "y", "z"}
        assert done["y"]["status"] == "error" and "boom" in done["y"]["error"]   # isolated, not fatal
        assert done["x"]["arch"] == "a" and done["z"]["score"] == 1
        # second invocation must skip everything already recorded (resumable)
        calls.clear()
        runner.run_sae_suite(locs, **common)
        assert calls == []


def test_announce_sentinel(capfd=None):
    done = {"a": {"status": "ok"}, "b": {"status": "ok"}, "c": {"status": "bad"}}
    runner.announce(done, total=3, extra_counts={"bad": "bad"})
    # not asserting stdout here (kept portable); just ensure it runs without error on complete + partial
    runner.announce({"a": {"status": "ok"}}, total=3)


# --------------------------------------------------------------------------- run_record
def test_write_run_record_generic():
    with tempfile.TemporaryDirectory() as d:
        wd = os.path.join(d, "wd")
        os.makedirs(wd)
        json.dump({"provenance": {"packages": {"sae-bench": "9.9"}, "hostname": "h", "git_sha": "abc123"},
                   "sae_bench_version": "9.9", "config": {"k": 1}, "n_saes": 2},
                  open(os.path.join(wd, "run_meta.json"), "w"))
        proc = os.path.join(d, "proc.json")
        json.dump({"result": {"n_ok": 2, "by_arch": {"std": {"x": {"mean": 0.5, "n": 2}}}}},
                  open(proc, "w"))
        pub = os.path.join(wd, "published_ref.json")
        json.dump({"std": {"x": 0.4, "n_trainers": 2}}, open(pub, "w"))
        rec = os.path.join(d, "record")
        jp, mp = write_run_record(rec, metric="demo", suite="s1",
                                  version_workdirs={"vA": wd}, version_processed={"vA": proc},
                                  version_labels={"vA": "vA (test)"}, published_path=pub,
                                  score_keys=[("x", "x")], report_text="hello", full_suite_n=2)
        rj = json.load(open(jp))
        assert rj["metric"] == "demo" and rj["runs"]["vA"]["n_ok"] == 2
        assert rj["published_ref"]["std"]["x"] == 0.4
        md = open(mp).read()
        assert "demo run record — s1" in md and "vA (test)" in md and "hello" in md
        assert "0.5000" in md and "0.4000" in md   # our value + published value rendered


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
    print(f"\n{passed}/{len(tests)} scaffolding unit tests passed")
    sys.exit(0 if passed == len(tests) else 1)
