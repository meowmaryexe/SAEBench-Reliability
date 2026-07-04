"""
Compare the absorption suite across sae-bench 0.3.2 (published-match) and 0.6.0 (current), against the
published SAEBench numbers. Prints, per architecture, both headline scores for each source + deltas vs
published, and Spearman rank correlations — i.e. the reproduction check AND the ranking-stability answer
(does the architecture ordering survive the metric redefinition?).

  python scripts/absorption_suite_report.py \
    --v032 results/processed/absorption/pythia_4k_v032.json \
    --v060 results/processed/absorption/pythia_4k_v060.json \
    --published results/raw/absorption/pythia_4k_v032/published_ref.json

Reads the processed JSONs written by `aggregate_results.py --metric absorption` (uses result.by_arch).
Pure stdlib (no scipy) so it runs anywhere.
"""
import argparse
import json


def _by_arch(processed_path):
    """arch -> (fraction_mean, full_mean) from a processed absorption JSON."""
    d = json.load(open(processed_path))["result"]["by_arch"]
    out = {}
    for arch, s in d.items():
        fr = s.get("mean_absorption_fraction_score")
        fu = s.get("mean_full_absorption_score")
        out[arch] = (fr["mean"] if fr else None, fu["mean"] if fu else None)
    return out


def _pub_by_arch(published_path):
    p = json.load(open(published_path))
    return {a: (v.get("mean_absorption_fraction_score"), v.get("mean_full_absorption_score"))
            for a, v in p.items()}


def _ranks(values):
    """Average-tie ranks for a list of numbers."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(a, b):
    """Spearman rho between two equal-length numeric lists (Pearson on ranks)."""
    if len(a) < 2:
        return float("nan")
    ra, rb = _ranks(a), _ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((x - mb) ** 2 for x in rb) ** 0.5
    return cov / (va * vb) if va and vb else float("nan")


def _aligned(d_pub, d_a, d_b, idx):
    """Architectures present in all three, with the metric at position idx (0=fraction, 1=full)."""
    archs = sorted(a for a in d_pub if a in d_a and a in d_b
                   and None not in (d_pub[a][idx], d_a[a][idx], d_b[a][idx]))
    return archs, [d_pub[a][idx] for a in archs], [d_a[a][idx] for a in archs], [d_b[a][idx] for a in archs]


def main():
    ap = argparse.ArgumentParser(description="Absorption suite report: 0.3.2 vs 0.6.0 vs published")
    ap.add_argument("--v032", required=True, help="processed JSON from the 0.3.2 run")
    ap.add_argument("--v060", required=True, help="processed JSON from the 0.6.0 run")
    ap.add_argument("--published", required=True, help="published_ref.json")
    ap.add_argument("--drift_band", type=float, default=0.05)
    args = ap.parse_args()

    pub, v032, v060 = _pub_by_arch(args.published), _by_arch(args.v032), _by_arch(args.v060)

    # Guard: published values are per-architecture means over all 6 trainers. Comparing a partial run
    # (e.g. a single-trainer smoke test) against them is apples-to-oranges — full_absorption especially
    # varies ~8x across the sparsity sweep. Warn if our runs look partial.
    n032 = json.load(open(args.v032))["result"].get("n_ok", 0)
    n060 = json.load(open(args.v060))["result"].get("n_ok", 0)
    if n032 < 42 or n060 < 42:
        print(f"⚠  PARTIAL RUN (0.3.2 n_ok={n032}, 0.6.0 n_ok={n060}; full suite = 42). Deltas vs the "
              f"published 6-trainer means are NOT meaningful until all trainers per arch are run — "
              f"full_absorption in particular varies ~8x across the sparsity sweep.")

    for label, idx in [("absorption_fraction", 0), ("full_absorption", 1)]:
        print(f"\n=== {label} ===")
        print(f"{'arch':>18} {'published':>10} {'v0.3.2':>10} {'Δ':>8} {'v0.6.0':>10} {'Δ':>8}")
        archs, p, a, b = _aligned(pub, v032, v060, idx)
        for arch, pv, av, bv in zip(archs, p, a, b):
            print(f"{arch:>18} {pv:>10.4f} {av:>10.4f} {av-pv:>+8.4f} {bv:>10.4f} {bv-pv:>+8.4f}")
        if len(archs) >= 2:
            print(f"  Spearman ρ  published↔0.3.2={_spearman(p, a):+.3f}  "
                  f"published↔0.6.0={_spearman(p, b):+.3f}  0.3.2↔0.6.0={_spearman(a, b):+.3f}")

    # Verdicts on the fraction (the version-sensitive score)
    archs, p, a, b = _aligned(pub, v032, v060, 0)
    if archs:
        max_d032 = max(abs(av - pv) for av, pv in zip(a, p))
        print(f"\n[verdict] fraction: 0.3.2 max |Δ vs published| = {max_d032:.4f} "
              f"({'within' if max_d032 <= args.drift_band else 'OUTSIDE'} drift band {args.drift_band}) "
              f"→ {'reproduces' if max_d032 <= args.drift_band else 'does NOT reproduce'} published")
        rho = _spearman(p, a)
        rho_b = _spearman(p, b)
        print(f"[verdict] ranking survives redefinition? published↔0.3.2 ρ={rho:+.3f}, "
              f"published↔0.6.0 ρ={rho_b:+.3f} "
              f"→ {'ranking holds' if rho_b >= 0.9 else 'ranking may shift under 0.6.0'}")


if __name__ == "__main__":
    main()
