"""
Entry point: aggregate a raw checkpoint (results/raw/.../ + run_meta.json) into a processed result
JSON (results/processed/<metric>/<name>.json).

  # Core (default): raw.jsonl per-batch -> aggregate, compared to the SAE's bundled eval_results.json
  python scripts/aggregate_results.py \
    --workdir results/raw/core_loss_recovered/standard_4k_t0_bundle_exact \
    --out results/processed/core_loss_recovered/standard_4k_t0_bundle_exact.json

  # Absorption: per-SAE absorption.jsonl -> collate both headline scores per architecture
  python scripts/aggregate_results.py --metric absorption \
    --workdir results/raw/absorption/standard_4k_t0 \
    --out results/processed/absorption/standard_4k_t0.json
"""
import argparse, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from saebench_audit.io import read_jsonl, write_json, _maybe_json
from saebench_audit.statistics import (
    aggregate_core, compare_to_bundle, aggregate_absorption, compare_absorption_to_published,
    aggregate_unlearning, compare_unlearning_to_published,
    aggregate_ravel, compare_ravel_to_published,
)


def _aggregate_absorption(workdir, out_path):
    meta = _maybe_json(os.path.join(workdir, "run_meta.json")) or {}
    rows = read_jsonl(os.path.join(workdir, "absorption.jsonl"))
    agg = aggregate_absorption(rows)
    published = _maybe_json(os.path.join(workdir, "published_ref.json"))  # optional results-repo values
    out = {
        "metric": "absorption",
        "sae_repo": meta.get("sae_repo"),
        "model_name": meta.get("model_name"),
        "config": meta.get("config"),
        "result": agg,
        "comparison_to_published": compare_absorption_to_published(agg, published),
        "published_ref": published,
    }
    write_json(out_path, out)
    print(f"[agg] {out_path}")
    print(f"[agg] absorption: {agg['n_ok']}/{agg['n_saes']} ok "
          f"({agg['n_insufficient_features']} insufficient_features)")
    for arch, s in agg["by_arch"].items():
        fr, fu = s["mean_absorption_fraction_score"], s["mean_full_absorption_score"]
        fr_s = f"{fr['mean']:.4f}±{fr['std']:.4f} (n={fr['n']})" if fr else "n/a"
        fu_s = f"{fu['mean']:.4f}±{fu['std']:.4f}" if fu else "n/a"
        print(f"[agg]   {arch:>12}: fraction={fr_s}  full={fu_s}")


def _aggregate_unlearning(workdir, out_path):
    meta = _maybe_json(os.path.join(workdir, "run_meta.json")) or {}
    rows = read_jsonl(os.path.join(workdir, "unlearning.jsonl"))
    agg = aggregate_unlearning(rows)
    published = _maybe_json(os.path.join(workdir, "published_ref.json"))  # optional results-repo values
    out = {
        "metric": "unlearning",
        "sae_repo": meta.get("sae_repo"),
        "model_name": meta.get("model_name"),
        "config": meta.get("config"),
        "result": agg,
        "comparison_to_published": compare_unlearning_to_published(agg, published),
        "published_ref": published,
    }
    write_json(out_path, out)
    print(f"[agg] {out_path}")
    print(f"[agg] unlearning: {agg['n_ok']}/{agg['n_saes']} ok ({agg['n_failed']} failed)")
    for arch, s in agg["by_arch"].items():
        u = s["unlearning_score"]
        u_s = f"{u['mean']:.4f}±{u['std']:.4f} (n={u['n']})" if u else "n/a"
        print(f"[agg]   {arch:>18}: unlearning_score={u_s}")


def _aggregate_ravel(workdir, out_path):
    meta = _maybe_json(os.path.join(workdir, "run_meta.json")) or {}
    rows = read_jsonl(os.path.join(workdir, "ravel.jsonl"))
    agg = aggregate_ravel(rows)
    published = _maybe_json(os.path.join(workdir, "published_ref.json"))  # optional results-repo values
    out = {
        "metric": "ravel",
        "sae_repo": meta.get("sae_repo"),
        "model_name": meta.get("model_name"),
        "config": meta.get("config"),
        "result": agg,
        "comparison_to_published": compare_ravel_to_published(agg, published),
        "published_ref": published,
    }
    write_json(out_path, out)
    print(f"[agg] {out_path}")
    print(f"[agg] ravel: {agg['n_ok']}/{agg['n_saes']} ok ({agg['n_failed']} failed)")
    for arch, s in agg["by_arch"].items():
        d, c, i = s["disentanglement_score"], s["cause_score"], s["isolation_score"]
        d_s = f"{d['mean']:.4f}±{d['std']:.4f} (n={d['n']})" if d else "n/a"
        c_s = f"{c['mean']:.4f}" if c else "n/a"
        i_s = f"{i['mean']:.4f}" if i else "n/a"
        print(f"[agg]   {arch:>18}: disentangle={d_s}  cause={c_s}  iso={i_s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", choices=["core", "absorption", "unlearning", "ravel"], default="core")
    ap.add_argument("--workdir", required=True, help="raw checkpoint dir + run_meta.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.metric == "absorption":
        _aggregate_absorption(args.workdir, args.out)
        return
    if args.metric == "unlearning":
        _aggregate_unlearning(args.workdir, args.out)
        return
    if args.metric == "ravel":
        _aggregate_ravel(args.workdir, args.out)
        return

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
