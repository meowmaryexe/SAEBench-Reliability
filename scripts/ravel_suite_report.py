"""
Compare our RAVEL suite run against the published SAEBench numbers, per architecture: the headline
`disentanglement_score` (primary ranking) plus the cause/isolation components for each source + delta, and
the Spearman rank correlation on disentanglement (the reproduction check + architecture ranking-stability
answer). RAVEL has no known sae-bench version drift, so a single current-version run is compared to published.

  python scripts/ravel_suite_report.py \
    --ours results/processed/ravel/gemma-2-2b_4k.json \
    --published results/raw/ravel/gemma-2-2b_4k/published_ref.json

Reads the processed JSON written by `aggregate_results.py --metric ravel` (uses result.by_arch). Pure stdlib.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from saebench_audit.statistics import spearman

SCORE_KEYS = ("disentanglement_score", "cause_score", "isolation_score")
PRIMARY = "disentanglement_score"


def _ours_by_arch(processed_path, key):
    d = json.load(open(processed_path))["result"]["by_arch"]
    return {arch: (s.get(key)["mean"] if s.get(key) else None) for arch, s in d.items()}


def _pub_by_arch(published_path, key):
    return {a: v.get(key) for a, v in json.load(open(published_path)).items()}


def main():
    ap = argparse.ArgumentParser(description="RAVEL suite report: ours vs published")
    ap.add_argument("--ours", required=True, help="processed JSON from our run")
    ap.add_argument("--published", required=True, help="published_ref.json")
    ap.add_argument("--drift_band", type=float, default=0.05)
    args = ap.parse_args()

    n_ok = json.load(open(args.ours))["result"].get("n_ok", 0)
    if n_ok < 42:
        print(f"⚠  PARTIAL RUN (n_ok={n_ok}; full suite = 42). Per-arch deltas vs the published 6-trainer "
              f"means are not meaningful until all trainers per arch are run.")

    for key in SCORE_KEYS:
        ours, pub = _ours_by_arch(args.ours, key), _pub_by_arch(args.published, key)
        archs = sorted(a for a in pub if a in ours and None not in (pub[a], ours[a]))
        print(f"\n=== {key} ===")
        print(f"{'arch':>18} {'published':>10} {'ours':>10} {'Δ':>8}")
        for a in archs:
            print(f"{a:>18} {pub[a]:>10.4f} {ours[a]:>10.4f} {ours[a]-pub[a]:>+8.4f}")

    # Verdict on the primary (disentanglement) score.
    ours, pub = _ours_by_arch(args.ours, PRIMARY), _pub_by_arch(args.published, PRIMARY)
    archs = sorted(a for a in pub if a in ours and None not in (pub[a], ours[a]))
    if len(archs) >= 2:
        p = [pub[a] for a in archs]
        o = [ours[a] for a in archs]
        rho = spearman(p, o)
        max_d = max(abs(oo - pp) for oo, pp in zip(o, p))
        print(f"\n[verdict] ravel disentanglement: max |Δ vs published| = {max_d:.4f} "
              f"({'within' if max_d <= args.drift_band else 'OUTSIDE'} drift band {args.drift_band}) "
              f"→ {'reproduces' if max_d <= args.drift_band else 'does NOT reproduce'} published")
        print(f"[verdict] architecture ranking: published↔ours ρ={rho:+.3f} "
              f"→ {'ranking holds' if rho >= 0.9 else 'ranking differs'}")


if __name__ == "__main__":
    main()
