"""
Download SAEBench's published RAVEL results for a suite (from adamkarvonen/sae_bench_results_0125) and
write a `published_ref.json` into a workdir, for `aggregate_results.py --metric ravel` and
`ravel_suite_report.py` to compare against.

  python scripts/fetch_published_ravel.py --suite gemma-2-2b_4k \
    --workdir results/raw/ravel/gemma-2-2b_4k

Output `published_ref.json` maps arch -> {disentanglement_score, cause_score, isolation_score, n_trainers,
per_sae:{sae_name:{...}}}.

VERIFY the published layout before trusting it (it differs per metric): the RAVEL results dir is expected at
`ravel/{sae_repo_basename}/` with each file's score under
`eval_result_metrics.ravel.{disentanglement,cause,isolation}_score`. If the published RAVEL dirs use a
different prefix or width naming (the way unlearning's 16k/65k use adamkarvonen date-0108 while the SAE
weights are canrager date-0107), pass `--results_prefix` explicitly. If RAVEL results are NOT in this
repo at all, record that and fall back to Neuronpedia / the paper's Table 7 as the comparison source.
"""
import argparse
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from saebench_audit.suites import resolve_suite

SCORE_KEYS = ("disentanglement_score", "cause_score", "isolation_score")


def _arch_from_stem(stem: str, repo_base: str) -> str:
    """Published filename stem is `{repo_base}_{Arch}_..._resid_post_layer_L_trainer_i` — parse the arch
    the same way the runner labels SAEs (leading token of the SAE folder, lowercased)."""
    rest = stem[len(repo_base) + 1:] if stem.startswith(repo_base + "_") else stem
    return rest.split("_")[0].lower()


def main():
    ap = argparse.ArgumentParser(description="Fetch published RAVEL results -> published_ref.json")
    ap.add_argument("--suite", help="registry suite name (e.g. gemma-2-2b_4k)")
    ap.add_argument("--sae_repo", help="SAE hf repo (alternative to --suite)")
    ap.add_argument("--results_prefix", default=None,
                    help="explicit results dir under the repo (e.g. ravel/saebench_gemma-2-2b_width-"
                         "2pow16_date-0108); overrides the SAE-repo-derived prefix for 16k/65k gemma")
    ap.add_argument("--results_repo", default="adamkarvonen/sae_bench_results_0125")
    ap.add_argument("--workdir", required=True)
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download, list_repo_files

    if args.results_prefix:
        prefix = args.results_prefix.rstrip("/") + "/"
    else:
        sae_repo = args.sae_repo or resolve_suite(args.suite)["sae_repo"]
        prefix = f"ravel/{sae_repo.split('/')[-1]}/"
    repo_base = prefix.rstrip("/").split("/")[-1]

    files = [f for f in list_repo_files(args.results_repo, repo_type="dataset")
             if f.startswith(prefix) and f.endswith("_eval_results.json")]
    if not files:
        raise SystemExit(f"No published RAVEL files under {prefix} in {args.results_repo}. "
                         f"Pass --results_prefix explicitly, or fall back to Neuronpedia / paper Table 7 "
                         f"if RAVEL results are not published in this repo.")

    by_arch: dict[str, dict] = {}
    for f in sorted(files):
        stem = os.path.basename(f)[: -len("_eval_results.json")]
        arch = _arch_from_stem(stem, repo_base)
        m = json.load(open(hf_hub_download(args.results_repo, f, repo_type="dataset")))[
            "eval_result_metrics"]["ravel"]
        d = by_arch.setdefault(arch, {"_scores": {k: [] for k in SCORE_KEYS}, "per_sae": {}})
        for k in SCORE_KEYS:
            d["_scores"][k].append(m[k])
        d["per_sae"][stem] = {k: m[k] for k in SCORE_KEYS}

    published = {}
    for arch, d in sorted(by_arch.items()):
        s = d["_scores"]
        n = len(s["disentanglement_score"])
        published[arch] = {**{k: sum(s[k]) / len(s[k]) for k in SCORE_KEYS},
                           "n_trainers": n, "per_sae": d["per_sae"]}

    os.makedirs(args.workdir, exist_ok=True)
    out = os.path.join(args.workdir, "published_ref.json")
    json.dump(published, open(out, "w"), indent=2)
    print(f"[published] {out}  ({len(files)} SAEs, {len(published)} architectures)")
    for arch, v in published.items():
        print(f"[published]   {arch:>18}: disentangle={v['disentanglement_score']:.4f}  "
              f"cause={v['cause_score']:.4f}  iso={v['isolation_score']:.4f}  (n={v['n_trainers']})")


if __name__ == "__main__":
    main()
