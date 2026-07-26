"""
Entry point: regenerate figures/ from processed results. Dependency-free (SVG via
saebench_audit.plotting). Default target = the Core / Loss Recovered exact bundle reproduction.

  python scripts/make_figures.py
"""
import glob, json, os, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from saebench_audit import plotting

PROC = os.path.join(ROOT, "results", "processed", "core_loss_recovered")
FIG = os.path.join(ROOT, "figures")


def _pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    return cov / (vx * vy) if vx and vy else float("nan")


def _summary(rows):
    """Summary block fig_reproduction_scatter needs, over SAEs that ship a bundled value.

    42 of the 252 released SAEs have no eval_results.json, so bundle_frac can be None.
    """
    pairs = [(r["bundle_frac"], r["loss_recovered"]) for r in rows if r.get("bundle_frac") is not None]
    if len(pairs) < 2:
        return None
    b, m = [p[0] for p in pairs], [p[1] for p in pairs]
    d = [abs(x - y) for x, y in pairs]
    return {"n_saes": len(pairs), "max_abs_dLR": max(d), "mean_abs_dLR": sum(d) / len(d),
            "pearson_LR": _pearson(b, m)}


def main():
    os.makedirs(FIG, exist_ok=True)
    wrote = []

    # --- CPU-validation artifact (not produced by the GPU sweep; results/processed is gitignored) ---
    exact_path = os.path.join(PROC, "standard_4k_t0_bundle_EXACT_perdoc_ctx1024_pile.json")
    if os.path.exists(exact_path):
        exact = json.load(open(exact_path))
        r, b = exact["result"], exact["bundled_eval_results"]
        sub = "Standard 4k SAE · Pythia-160M L8 · exact per-document path (128 docs) vs bundled eval_results.json"

        svg1 = plotting.fig_mine_vs_bundle(r, b, title_sub=sub)
        open(os.path.join(FIG, "core_lr_mine_vs_bundle.svg"), "w").write(svg1)
        wrote.append("core_lr_mine_vs_bundle.svg")

        svg2 = plotting.fig_convergence(exact["per_batch"], b["loss_original"], b["frac_recovered"],
                                        title_sub="Standard 4k SAE · Pythia-160M L8 · running means vs document count")
        open(os.path.join(FIG, "core_lr_convergence.svg"), "w").write(svg2)
        wrote.append("core_lr_convergence.svg")
    else:
        print(f"skip: {os.path.basename(exact_path)} not present (CPU-validation artifact)")

    # --- per-suite figures straight from the GPU sweep output ---
    for path in sorted(glob.glob(os.path.join(PROC, "*_saebench_core.json"))):
        d = json.load(open(path))
        rows = d.get("per_sae") or []
        if not rows:
            print(f"skip: {os.path.basename(path)} has no per_sae rows"); continue
        name = d.get("suite") or os.path.basename(path).replace("_saebench_core.json", "")
        sub = (f"{d.get('model', '?')} L{d.get('layer', '?')} · {name} · "
               f"{len(rows)} SAEs · packed ctx{d.get('eval', {}).get('context_size', '?')}")
        open(os.path.join(FIG, f"core_lr_frontier_{name}.svg"), "w").write(
            plotting.fig_frontier(rows, subtitle=sub))
        wrote.append(f"core_lr_frontier_{name}.svg")

        summ = _summary(rows)
        if summ is None:
            print(f"skip: core_lr_reproduction_{name}.svg — fewer than 2 SAEs ship a bundled value")
        else:
            open(os.path.join(FIG, f"core_lr_reproduction_{name}.svg"), "w").write(
                plotting.fig_reproduction_scatter(rows, summ, subtitle=sub))
            wrote.append(f"core_lr_reproduction_{name}.svg")

    # --- full-suite figures (7 architectures x 6 sparsities, 4k Pythia-160M) ---
    suite_path = os.path.join(PROC, "suite_4k_pythia160m.json")
    if os.path.exists(suite_path):
        suite = json.load(open(suite_path))
        rows, summ = suite["per_sae"], suite["summary"]
        sub = "Pythia-160M L8 · 4k width · 7 architectures × 6 sparsities · per-document ctx1024"
        svg3 = plotting.fig_frontier(rows, subtitle=sub)
        open(os.path.join(FIG, "core_lr_frontier_4k.svg"), "w").write(svg3)
        wrote.append("core_lr_frontier_4k.svg")
        svg4 = plotting.fig_reproduction_scatter(rows, summ, subtitle=sub)
        open(os.path.join(FIG, "core_lr_reproduction_4k.svg"), "w").write(svg4)
        wrote.append("core_lr_reproduction_4k.svg")

    # --- full Core metric set vs published Neuronpedia values ---
    full_path = os.path.join(PROC, "full_metrics_vs_neuronpedia.json")
    if os.path.exists(full_path):
        fd = json.load(open(full_path))
        svg5 = plotting.fig_metric_agreement(
            fd["per_metric_summary"],
            subtitle="42 SAEs (7 arch × 6 sparsity), 4k Pythia-160M L8 — mean rel error vs published, Pearson r")
        open(os.path.join(FIG, "core_full_metrics_vs_neuronpedia.svg"), "w").write(svg5)
        wrote.append("core_full_metrics_vs_neuronpedia.svg")

    # --- AutoInterp figures ---
    aproc = os.path.join(ROOT, "results", "processed", "autointerp")
    conv = os.path.join(aproc, "autointerp_vs_published.json")
    if os.path.exists(conv):
        c = json.load(open(conv))
        pts = [(p["n_tokens"], p["autointerp_score"]) for p in c["convergence_standard_t0"] if p["n_tokens"] < 2_000_000]
        open(os.path.join(FIG, "autointerp_convergence.svg"), "w").write(
            plotting.fig_autointerp_convergence(pts, 0.7803, c["null_baseline"],
                subtitle="Standard 4k SAE, Pythia-160M L8, judge = gpt-4o-mini (the paper's judge)"))
        wrote.append("autointerp_convergence.svg")
    r96 = os.path.join(aproc, "standard_4k_t0_autointerp_96k.json")
    if os.path.exists(r96):
        d = json.load(open(r96))
        rows = [r for r in d["per_latent"] if r.get("score") is not None]
        scores = [r["score"] for r in rows]
        open(os.path.join(FIG, "autointerp_score_histogram.svg"), "w").write(
            plotting.fig_autointerp_score_hist(scores, d["autointerp_score"], 0.7803, 0.7143,
                subtitle=f"Standard 4k SAE, Pythia-160M L8, gpt-4o-mini, n={len(scores)} latents @ 96k tokens"))
        rs = sorted(rows, key=lambda r: -r["score"])
        sample = rs[:4] + rs[len(rs)//2-2:len(rs)//2+2] + rs[-4:]
        open(os.path.join(FIG, "autointerp_explanations_showcase.svg"), "w").write(
            plotting.fig_autointerp_showcase(sample,
                subtitle="Standard 4k SAE, Pythia-160M L8 — sample latents (high / mid / low detection score)"))
        wrote += ["autointerp_score_histogram.svg", "autointerp_explanations_showcase.svg"]

    for f in wrote:
        print(f"wrote figures/{f}")
    print(f"\n{len(wrote)} figure(s) written to figures/")
    if not wrote:
        print("nothing to plot — run the sweep first (results/processed/ is empty)")


if __name__ == "__main__":
    main()
