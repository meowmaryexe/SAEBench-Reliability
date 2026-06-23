"""
Entry point: regenerate figures/ from processed results. Dependency-free (SVG via
saebench_audit.plotting). Default target = the Core / Loss Recovered exact bundle reproduction.

  python scripts/make_figures.py
"""
import json, os, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from saebench_audit import plotting

PROC = os.path.join(ROOT, "results", "processed", "core_loss_recovered")
FIG = os.path.join(ROOT, "figures")


def main():
    os.makedirs(FIG, exist_ok=True)
    exact = json.load(open(os.path.join(PROC, "standard_4k_t0_bundle_EXACT_perdoc_ctx1024_pile.json")))
    r, b = exact["result"], exact["bundled_eval_results"]
    sub = "Standard 4k SAE · Pythia-160M L8 · exact per-document path (128 docs) vs bundled eval_results.json"

    svg1 = plotting.fig_mine_vs_bundle(r, b, title_sub=sub)
    open(os.path.join(FIG, "core_lr_mine_vs_bundle.svg"), "w").write(svg1)
    print("wrote figures/core_lr_mine_vs_bundle.svg")

    svg2 = plotting.fig_convergence(exact["per_batch"], b["loss_original"], b["frac_recovered"],
                                    title_sub="Standard 4k SAE · Pythia-160M L8 · running means vs document count")
    open(os.path.join(FIG, "core_lr_convergence.svg"), "w").write(svg2)
    print("wrote figures/core_lr_convergence.svg")

    # --- full-suite figures (7 architectures x 6 sparsities, 4k Pythia-160M) ---
    suite_path = os.path.join(PROC, "suite_4k_pythia160m.json")
    if os.path.exists(suite_path):
        suite = json.load(open(suite_path))
        rows, summ = suite["per_sae"], suite["summary"]
        sub = "Pythia-160M L8 · 4k width · 7 architectures × 6 sparsities · per-document ctx1024"
        svg3 = plotting.fig_frontier(rows, subtitle=sub)
        open(os.path.join(FIG, "core_lr_frontier_4k.svg"), "w").write(svg3)
        print("wrote figures/core_lr_frontier_4k.svg")
        svg4 = plotting.fig_reproduction_scatter(rows, summ, subtitle=sub)
        open(os.path.join(FIG, "core_lr_reproduction_4k.svg"), "w").write(svg4)
        print("wrote figures/core_lr_reproduction_4k.svg")

    # --- full Core metric set vs published Neuronpedia values ---
    full_path = os.path.join(PROC, "full_metrics_vs_neuronpedia.json")
    if os.path.exists(full_path):
        fd = json.load(open(full_path))
        svg5 = plotting.fig_metric_agreement(
            fd["per_metric_summary"],
            subtitle="42 SAEs (7 arch × 6 sparsity), 4k Pythia-160M L8 — mean rel error vs published, Pearson r")
        open(os.path.join(FIG, "core_full_metrics_vs_neuronpedia.svg"), "w").write(svg5)
        print("wrote figures/core_full_metrics_vs_neuronpedia.svg")


if __name__ == "__main__":
    main()
