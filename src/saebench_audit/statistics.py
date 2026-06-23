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
