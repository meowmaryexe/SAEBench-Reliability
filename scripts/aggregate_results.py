"""
Entry point: aggregate a raw per-batch checkpoint (results/raw/.../raw.jsonl + run_meta.json)
into a processed result JSON (results/processed/<metric>/<name>.json), with a comparison to the
SAE's bundled eval_results.json when available.

Example:
  python scripts/aggregate_results.py \
    --workdir results/raw/core_loss_recovered/standard_4k_t0_bundle_exact \
    --out results/processed/core_loss_recovered/standard_4k_t0_bundle_exact.json
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from saebench_audit.io import read_jsonl, write_json, _maybe_json
from saebench_audit.statistics import aggregate_core, compare_to_bundle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True, help="dir containing raw.jsonl + run_meta.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = read_jsonl(os.path.join(args.workdir, "raw.jsonl"))
    meta = _maybe_json(os.path.join(args.workdir, "run_meta.json")) or {}
    agg = aggregate_core(rows)

    bundled = None
    sae_dir = meta.get("local_sae_dir")
    if sae_dir:
        bundled = _maybe_json(os.path.join(sae_dir, "eval_results.json"))

    out = {
        "metric": meta.get("metric", "core_loss_recovered"),
        "variant": meta.get("variant"),
        "arch": meta.get("arch"),
        "config": meta.get("config"),
        "result": agg,
        "comparison_to_bundle": compare_to_bundle(agg, bundled),
        "bundled_eval_results": bundled,
        "per_batch": sorted(rows, key=lambda r: r["bi"]),
    }
    write_json(args.out, out)
    print(f"[agg] {args.out}")
    print(f"[agg] Loss Recovered={agg['frac_recovered']:.4f}  L0={agg['l0']:.1f}  "
          f"Horig={agg['loss_original']:.4f}  H0={agg['loss_zero']:.4f}  (n={agg['n_sequences']} seq)")
    if bundled:
        print(f"[ref] bundle frac={bundled.get('frac_recovered'):.4f}  l0={bundled.get('l0'):.1f}  "
              f"Horig={bundled.get('loss_original'):.4f}")


if __name__ == "__main__":
    main()
