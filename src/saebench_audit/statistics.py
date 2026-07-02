"""
Aggregation / summary statistics over per-batch records.
"""
from __future__ import annotations
from .schema import CORE_AGG_KEYS


def aggregate_core(rows):
    """Aggregate per-batch Core records into summary statistics.

    Loss Recovered and L0 are means over per-batch values (matching dictionary_learning's
    evaluate(), which averages frac_recovered across batches). A pooled frac (ratio of mean
    losses) is reported as a diagnostic.
    """
    if not rows:
        raise ValueError("no rows to aggregate")
    rows = sorted(rows, key=lambda r: r["bi"])
    n = len(rows)
    agg = {k: sum(r[k] for r in rows) / n for k in CORE_AGG_KEYS}
    pooled = ((agg["loss_reconstructed"] - agg["loss_zero"]) /
              (agg["loss_original"] - agg["loss_zero"]))
    return {
        **agg,
        "loss_recovered": agg["frac_recovered"],
        "frac_recovered_pooled": pooled,
        "n_batches": n,
        "n_sequences": sum(r.get("n_seqs", 0) for r in rows),
    }


def compare_to_bundle(agg, bundled):
    """Per-quantity absolute + relative deltas vs a bundled eval_results.json reference."""
    if not bundled:
        return {}
    out = {}
    for k in ("loss_original", "loss_reconstructed", "loss_zero", "frac_recovered", "l0"):
        if k in bundled and k in agg:
            mine, ref = agg[k], bundled[k]
            out[k] = {"mine": mine, "bundle": ref, "abs_delta": abs(mine - ref),
                      "rel_delta_pct": 100 * abs(mine - ref) / ref if ref else float("nan")}
    return out


# ---------------------------------------------------------------------------
# Absorption. Each SAE is its own data point (upstream already aggregates over the 26 letters),
# so we collate per-SAE rows and summarize the two headline scores per architecture.
# ---------------------------------------------------------------------------
def _summary(values):
    """mean / sample-std / n for a list of floats (std uses n-1; 0.0 when n<2)."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 if len(vals) > 1 else 0.0
    return {"mean": m, "std": sd, "n": len(vals), "values": vals}


def aggregate_absorption(rows):
    """Collate per-SAE absorption rows (from absorption.jsonl) into a processed result.

    Reports both headline scores per SAE and their mean/std per architecture. Rows that tripped the
    min-features guard (status != "ok") are counted but excluded from the score summaries.
    """
    if not rows:
        raise ValueError("no absorption rows to aggregate")
    ok = [r for r in rows if r.get("status") == "ok"]
    insufficient = [r for r in rows if r.get("status") != "ok"]

    by_arch = {}
    for r in ok:
        by_arch.setdefault(r.get("arch", "?"), []).append(r)

    return {
        "n_saes": len(rows),
        "n_ok": len(ok),
        "n_insufficient_features": len(insufficient),
        "by_arch": {
            arch: {
                "mean_absorption_fraction_score": _summary(
                    [r.get("mean_absorption_fraction_score") for r in rs]),
                "mean_full_absorption_score": _summary(
                    [r.get("mean_full_absorption_score") for r in rs]),
            }
            for arch, rs in sorted(by_arch.items())
        },
        "per_sae": sorted(
            [{k: r.get(k) for k in (
                "sae_name", "arch", "location", "status",
                "mean_absorption_fraction_score", "mean_full_absorption_score",
                "mean_num_split_features", "std_dev_absorption_fraction_score",
                "std_dev_full_absorption_score", "std_dev_num_split_features")}
             for r in rows],
            key=lambda r: (r.get("arch") or "", r.get("sae_name") or ""),
        ),
    }


def compare_absorption_to_published(agg, published):
    """Per-architecture absolute deltas of the two headline scores vs published SAEBench values.

    `published` maps arch -> {"mean_absorption_fraction_score": x, "mean_full_absorption_score": y}
    (source: the results repo adamkarvonen/sae_bench_results_0125). Returns {} if not provided.
    Note: upstream applies no seed, so exact matching is not expected — see docs/preregistration.md.
    """
    if not published:
        return {}
    out = {}
    for arch, ref in published.items():
        mine = agg.get("by_arch", {}).get(arch)
        if not mine:
            continue
        row = {}
        for k in ("mean_absorption_fraction_score", "mean_full_absorption_score"):
            if mine.get(k) and k in ref:
                m, r = mine[k]["mean"], ref[k]
                row[k] = {"mine": m, "published": r, "abs_delta": abs(m - r)}
        out[arch] = row
    return out
