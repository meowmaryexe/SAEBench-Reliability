"""
Download SAEBench's published unlearning results for a suite (from adamkarvonen/sae_bench_results_0125)
and write a `published_ref.json` into a workdir, for `aggregate_results.py --metric unlearning` and
`unlearning_suite_report.py` to compare against.

  python scripts/fetch_published_unlearning.py --suite gemma-2-2b-it_4k \
    --workdir results/raw/unlearning/gemma-2-2b-it_4k

Output `published_ref.json` maps arch -> {unlearning_score, n_trainers, per_sae:{sae_name:{...}}}.

NOTE — results↔repo naming: the published dirs are `unlearning/saebench_gemma-2-2b_width-2pow{12,14,16}
_date-0108/`. For 4k this equals the SAE repo base; for 16k/65k the registry SAE-weight repos are
`canrager/…date-0107`, so the results prefix can't be derived from the SAE repo — pass `--results_prefix`
explicitly for those widths. Verify the per-width mapping before trusting the comparison.
"""
import argparse
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from saebench_audit.suites import resolve_suite

SCORE_KEY = "unlearning_score"


def _arch_from_stem(stem: str, repo_base: str) -> str:
    """Published filename stem is `{repo_base}_{Arch}_..._resid_post_layer_L_trainer_i` — parse the arch
    the same way the runner labels SAEs (leading token of the SAE folder, lowercased)."""
    rest = stem[len(repo_base) + 1:] if stem.startswith(repo_base + "_") else stem
    return rest.split("_")[0].lower()


def main():
    ap = argparse.ArgumentParser(description="Fetch published unlearning results -> published_ref.json")
    ap.add_argument("--suite", help="registry suite name (e.g. gemma-2-2b-it_4k)")
    ap.add_argument("--sae_repo", help="SAE hf repo (alternative to --suite)")
    ap.add_argument("--results_prefix", default=None,
                    help="explicit results dir under the repo (e.g. unlearning/saebench_gemma-2-2b_width-"
                         "2pow16_date-0108); overrides the SAE-repo-derived prefix for 16k/65k gemma")
    ap.add_argument("--results_repo", default="adamkarvonen/sae_bench_results_0125")
    ap.add_argument("--workdir", required=True)
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download, list_repo_files

    if args.results_prefix:
        prefix = args.results_prefix.rstrip("/") + "/"
    else:
        sae_repo = args.sae_repo or resolve_suite(args.suite)["sae_repo"]
        prefix = f"unlearning/{sae_repo.split('/')[-1]}/"
    repo_base = prefix.rstrip("/").split("/")[-1]

    files = [f for f in list_repo_files(args.results_repo, repo_type="dataset")
             if f.startswith(prefix) and f.endswith("_eval_results.json")]
    if not files:
        raise SystemExit(f"No published unlearning files under {prefix} in {args.results_repo}. "
                         f"For 16k/65k gemma pass --results_prefix explicitly (date-0108 naming).")

    by_arch: dict[str, dict] = {}
    for f in sorted(files):
        stem = os.path.basename(f)[: -len("_eval_results.json")]
        arch = _arch_from_stem(stem, repo_base)
        score = json.load(open(hf_hub_download(args.results_repo, f, repo_type="dataset")))[
            "eval_result_metrics"]["unlearning"][SCORE_KEY]
        d = by_arch.setdefault(arch, {"_score": [], "per_sae": {}})
        d["_score"].append(score)
        d["per_sae"][stem] = {SCORE_KEY: score}

    published = {}
    for arch, d in sorted(by_arch.items()):
        s = d["_score"]
        published[arch] = {SCORE_KEY: sum(s) / len(s), "n_trainers": len(s), "per_sae": d["per_sae"]}

    os.makedirs(args.workdir, exist_ok=True)
    out = os.path.join(args.workdir, "published_ref.json")
    json.dump(published, open(out, "w"), indent=2)
    print(f"[published] {out}  ({len(files)} SAEs, {len(published)} architectures)")
    for arch, v in published.items():
        print(f"[published]   {arch:>18}: unlearning_score={v[SCORE_KEY]:.4f}  (n={v['n_trainers']})")


if __name__ == "__main__":
    main()
