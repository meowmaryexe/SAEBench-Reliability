"""
Compare our unlearning suite run against the published SAEBench numbers, per architecture: the headline
`unlearning_score` for each source + delta, and the Spearman rank correlation (the reproduction check +
architecture ranking-stability answer). Unlearning has no known sae-bench version drift (the eval module is
logically identical across 0.3.2/0.6.0), so a single current-version run is compared to published.

  python scripts/unlearning_suite_report.py \
    --ours results/processed/unlearning/gemma-2-2b-it_4k.json \
    --published results/raw/unlearning/gemma-2-2b-it_4k/published_ref.json

Reads the processed JSON written by `aggregate_results.py --metric unlearning` (uses result.by_arch).
Pure stdlib (no scipy). Note: unlearning is pre-conceded degenerate (near-zero) on Gemma-2-2B — small
absolute scores, so read deltas with that in mind.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from saebench_audit.statistics import spearman

SCORE_KEY = "unlearning_score"


def _ours_by_arch(processed_path):
    d = json.load(open(processed_path))["result"]["by_arch"]
    return {arch: (s.get(SCORE_KEY)["mean"] if s.get(SCORE_KEY) else None) for arch, s in d.items()}


def _pub_by_arch(published_path):
    return {a: v.get(SCORE_KEY) for a, v in json.load(open(published_path)).items()}


def main():
    ap = argparse.ArgumentParser(description="Unlearning suite report: ours vs published")
    ap.add_argument("--ours", required=True, help="processed JSON from our run")
    ap.add_argument("--published", required=True, help="published_ref.json")
    ap.add_argument("--drift_band", type=float, default=0.05)
    args = ap.parse_args()

    ours, pub = _ours_by_arch(args.ours), _pub_by_arch(args.published)
    n_ok = json.load(open(args.ours))["result"].get("n_ok", 0)
    if n_ok < 42:
        print(f"⚠  PARTIAL RUN (n_ok={n_ok}; full suite = 42). Per-arch deltas vs the published 6-trainer "
              f"means are not meaningful until all trainers per arch are run.")

    archs = sorted(a for a in pub if a in ours and None not in (pub[a], ours[a]))
    print(f"\n=== unlearning_score ===")
    print(f"{'arch':>18} {'published':>10} {'ours':>10} {'Δ':>8}")
    for a in archs:
        print(f"{a:>18} {pub[a]:>10.4f} {ours[a]:>10.4f} {ours[a]-pub[a]:>+8.4f}")
    p = [pub[a] for a in archs]
    o = [ours[a] for a in archs]
    if len(archs) >= 2:
        print(f"  Spearman ρ  published↔ours = {spearman(p, o):+.3f}")
        max_d = max(abs(oo - pp) for oo, pp in zip(o, p))
        print(f"\n[verdict] unlearning: max |Δ vs published| = {max_d:.4f} "
              f"({'within' if max_d <= args.drift_band else 'OUTSIDE'} drift band {args.drift_band}) "
              f"→ {'reproduces' if max_d <= args.drift_band else 'does NOT reproduce'} published")
        rho = spearman(p, o)
        print(f"[verdict] architecture ranking: published↔ours ρ={rho:+.3f} "
              f"→ {'ranking holds' if rho >= 0.9 else 'ranking differs'} "
              f"(caveat: scores are near-zero/degenerate on Gemma-2-2B, so ranking is noise-sensitive)")


if __name__ == "__main__":
    main()
