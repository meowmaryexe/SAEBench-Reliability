"""
Oracle test: prove our pure-Python `absorption_fraction_v032` / `absorption_fraction_v060`
(src/saebench_audit/metrics/absorption_fraction.py) reproduce upstream SAEBench's actual per-word
arithmetic, byte-for-byte, over random inputs (incl. edge cases). This is what makes the version-drift
finding trustworthy — the ports, not just the recorded result JSONs.

Compares against tests/saebench_absorption_fraction_verbatim.py (verbatim upstream tensor code). Needs
torch + numpy, so skips gracefully under an interpreter that lacks them. Run under a torch venv:
  /Users/alor/saebench-absorption-env/.venv/bin/python tests/test_absorption_fraction_oracle.py
"""
import importlib.util
import os
import random
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from saebench_audit.metrics.absorption_fraction import (
    absorption_fraction_v032,
    absorption_fraction_v060,
)


def _has(mod):
    return importlib.util.find_spec(mod) is not None


def _draw(torch, L=32):
    """Random per-word scenario: per-latent projections (signed), cosines in [-1,1], scalar act, main ids."""
    proj = torch.randn(L)
    cos = torch.rand(L) * 2 - 1
    act = float(torch.randn(1).item())
    while act == 0.0:
        act = float(torch.randn(1).item())
    main = random.sample(range(L), random.randint(1, 4))
    return proj, cos, act, main


def test_v032_port_matches_verbatim():
    if not (_has("torch") and _has("numpy")):
        print("  (skip v032 oracle: torch/numpy not importable)"); return
    import torch
    import saebench_absorption_fraction_verbatim as vb
    torch.manual_seed(0); random.seed(0)
    for _ in range(1000):
        proj, cos, act, main = _draw(torch)
        mine = absorption_fraction_v032(act, proj[main].sum().item(), proj.sum().item())
        ref = vb.fraction_v032_verbatim(proj, act, main)
        assert abs(mine - ref) < 1e-9, (mine, ref, act, main)


def test_v060_port_matches_verbatim():
    if not (_has("torch") and _has("numpy")):
        print("  (skip v060 oracle: torch/numpy not importable)"); return
    import torch
    import saebench_absorption_fraction_verbatim as vb
    torch.manual_seed(1); random.seed(1)
    hits_zero = hits_one = hits_mid = 0
    for _ in range(2000):
        proj, cos, act, main = _draw(torch)
        non_main = [i for i in range(proj.size(0)) if i not in main]
        mine = absorption_fraction_v060(
            act, proj[main].sum().item(), proj[non_main].tolist(), cos[non_main].tolist())
        ref = vb.fraction_v060_verbatim(proj, cos, act, main)
        assert abs(mine - ref) < 1e-9, (mine, ref, act, main)
        # track coverage of all three branches so the test isn't trivially all-zeros
        if ref == 0.0: hits_zero += 1
        elif ref == 1.0: hits_one += 1
        else: hits_mid += 1
    assert hits_zero and hits_mid, (hits_zero, hits_one, hits_mid)   # exercised the interesting branches


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
    print(f"\n{passed}/{len(tests)} absorption fraction oracle tests passed")
    sys.exit(0 if passed == len(tests) else 1)
