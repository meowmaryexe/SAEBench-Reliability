"""
Download SAEBench's published absorption results for a suite (from adamkarvonen/sae_bench_results_0125)
and write a `published_ref.json` into a workdir, for `aggregate_results.py --metric absorption` and
`absorption_suite_report.py` to compare against.

  python scripts/fetch_published_absorption.py --suite pythia-160m_4k \
    --workdir results/raw/absorption/pythia_4k_v032

Output `published_ref.json` maps arch -> {mean_absorption_fraction_score, mean_full_absorption_score,
n_trainers, per_sae:{sae_name: {...}}}. Per-arch means are what `compare_absorption_to_published` reads.
"""
import argparse
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))


def _resolve_repo(suite: str) -> str:
    import yaml

    reg = yaml.safe_load(open(os.path.join(ROOT, "configs", "registry.yaml")))
    return reg["sae_suites"][suite]["hf_repo"]


def _arch_from_stem(stem: str, repo_base: str) -> str:
    """Published filename stem is `{repo_base}_{Arch}_..._resid_post_layer_L_trainer_i` — parse the arch
    the same way the runner labels SAEs (leading token of the SAE folder, lowercased)."""
    rest = stem[len(repo_base) + 1:] if stem.startswith(repo_base + "_") else stem
    return rest.split("_")[0].lower()


def main():
    ap = argparse.ArgumentParser(description="Fetch published absorption results -> published_ref.json")
    ap.add_argument("--suite", help="registry suite name (e.g. pythia-160m_4k)")
    ap.add_argument("--sae_repo", help="SAE hf repo (alternative to --suite)")
    ap.add_argument("--results_repo", default="adamkarvonen/sae_bench_results_0125")
    ap.add_argument("--workdir", required=True)
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download, list_repo_files

    sae_repo = args.sae_repo or _resolve_repo(args.suite)
    repo_base = sae_repo.split("/")[-1]
    prefix = f"absorption/{repo_base}/"

    files = [f for f in list_repo_files(args.results_repo, repo_type="dataset")
             if f.startswith(prefix) and f.endswith("_eval_results.json")]
    if not files:
        raise SystemExit(f"No published absorption files under {prefix} in {args.results_repo}")

    by_arch: dict[str, dict] = {}
    for f in sorted(files):
        stem = os.path.basename(f)[: -len("_eval_results.json")]
        arch = _arch_from_stem(stem, repo_base)
        m = json.load(open(hf_hub_download(args.results_repo, f, repo_type="dataset")))["eval_result_metrics"]["mean"]
        d = by_arch.setdefault(arch, {"_frac": [], "_full": [], "per_sae": {}})
        d["_frac"].append(m["mean_absorption_fraction_score"])
        d["_full"].append(m["mean_full_absorption_score"])
        d["per_sae"][stem] = {"mean_absorption_fraction_score": m["mean_absorption_fraction_score"],
                              "mean_full_absorption_score": m["mean_full_absorption_score"]}

    published = {}
    for arch, d in sorted(by_arch.items()):
        fr, fu = d["_frac"], d["_full"]
        published[arch] = {
            "mean_absorption_fraction_score": sum(fr) / len(fr),
            "mean_full_absorption_score": sum(fu) / len(fu),
            "n_trainers": len(fr),
            "per_sae": d["per_sae"],
        }

    os.makedirs(args.workdir, exist_ok=True)
    out = os.path.join(args.workdir, "published_ref.json")
    json.dump(published, open(out, "w"), indent=2)
    print(f"[published] {out}  ({len(files)} SAEs, {len(published)} architectures)")
    for arch, v in published.items():
        print(f"[published]   {arch:>18}: fraction={v['mean_absorption_fraction_score']:.4f}  "
              f"full={v['mean_full_absorption_score']:.4f}  (n={v['n_trainers']})")


if __name__ == "__main__":
    main()
