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
# Generic helpers shared by the per-SAE / by-architecture metrics (Absorption now, Unlearning/RAVEL next).
# ---------------------------------------------------------------------------
def summary(values):
    """mean / sample-std / n for a list of floats (std uses n-1; 0.0 when n<2). None -> None."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 if len(vals) > 1 else 0.0
    return {"mean": m, "std": sd, "n": len(vals), "values": vals}


_summary = summary  # backward-compat alias (internal callers)


def ranks(values):
    """Average-tie ranks for a list of numbers (1-based)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def spearman(a, b):
    """Spearman rho between two equal-length numeric lists (Pearson on ranks). No scipy dependency."""
    if len(a) < 2:
        return float("nan")
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((x - mb) ** 2 for x in rb) ** 0.5
    return cov / (va * vb) if va and vb else float("nan")


def aggregate_by_arch(rows, score_keys, per_sae_keys, ok_status="ok", excluded_label="n_excluded"):
    """Collate per-SAE rows into a by-architecture result: mean/std/n of each score column per arch,
    plus a full per-SAE table. Rows with status != ok_status are counted (under `excluded_label`) but
    excluded from the score summaries. This is the shared shape behind the wrap-upstream aggregations.
    """
    if not rows:
        raise ValueError("no rows to aggregate")
    ok = [r for r in rows if r.get("status") == ok_status]
    excluded = [r for r in rows if r.get("status") != ok_status]
    by = {}
    for r in ok:
        by.setdefault(r.get("arch", "?"), []).append(r)
    return {
        "n_saes": len(rows),
        "n_ok": len(ok),
        excluded_label: len(excluded),
        "by_arch": {arch: {k: summary([r.get(k) for r in rs]) for k in score_keys}
                    for arch, rs in sorted(by.items())},
        "per_sae": sorted([{k: r.get(k) for k in per_sae_keys} for r in rows],
                          key=lambda r: (r.get("arch") or "", r.get("sae_name") or "")),
    }


def compare_by_arch_to_reference(agg, reference, score_keys, ref_label="published"):
    """Per-architecture absolute deltas of each score column vs a reference {arch: {score_key: value}}.
    Returns {} if no reference. Exact matching is not expected for unseeded upstreams (see preregistration).
    """
    if not reference:
        return {}
    out = {}
    for arch, ref in reference.items():
        mine = agg.get("by_arch", {}).get(arch)
        if not mine:
            continue
        row = {}
        for k in score_keys:
            if mine.get(k) and k in ref:
                m, r = mine[k]["mean"], ref[k]
                row[k] = {"mine": m, ref_label: r, "abs_delta": abs(m - r)}
        out[arch] = row
    return out


# ---------------------------------------------------------------------------
# Absorption. Each SAE is its own data point (upstream already aggregates over the 26 letters),
# so we collate per-SAE rows and summarize the two headline scores per architecture.
# ---------------------------------------------------------------------------
_ABS_SCORE_KEYS = ["mean_absorption_fraction_score", "mean_full_absorption_score"]
_ABS_PER_SAE_KEYS = (
    "sae_name", "arch", "location", "status",
    "mean_absorption_fraction_score", "mean_full_absorption_score",
    "mean_num_split_features", "std_dev_absorption_fraction_score",
    "std_dev_full_absorption_score", "std_dev_num_split_features",
)


def aggregate_absorption(rows):
    """Collate per-SAE absorption rows (from absorption.jsonl) into a processed result.

    Reports both headline scores per SAE and their mean/std per architecture. Rows that tripped the
    min-features guard (status != "ok") are counted but excluded from the score summaries.
    """
    return aggregate_by_arch(rows, _ABS_SCORE_KEYS, _ABS_PER_SAE_KEYS,
                             excluded_label="n_insufficient_features")


def compare_absorption_to_published(agg, published):
    """Per-architecture absolute deltas of the two headline scores vs published SAEBench values.

    `published` maps arch -> {"mean_absorption_fraction_score": x, "mean_full_absorption_score": y}
    (source: the results repo adamkarvonen/sae_bench_results_0125). Returns {} if not provided.
    """
    return compare_by_arch_to_reference(agg, published, _ABS_SCORE_KEYS)


# ---------------------------------------------------------------------------
# Unlearning. One scalar `unlearning_score` per SAE (upstream already reduces the 16-config sweep); collate
# per-SAE rows and summarize the headline score per architecture.
# ---------------------------------------------------------------------------
_UNLEARN_SCORE_KEYS = ["unlearning_score"]
_UNLEARN_PER_SAE_KEYS = ("sae_name", "arch", "location", "status", "unlearning_score")


def aggregate_unlearning(rows):
    """Collate per-SAE unlearning rows (from unlearning.jsonl) into a processed result: the headline
    `unlearning_score` mean/std per architecture. Non-ok rows are counted but excluded from summaries."""
    return aggregate_by_arch(rows, _UNLEARN_SCORE_KEYS, _UNLEARN_PER_SAE_KEYS, excluded_label="n_failed")


def compare_unlearning_to_published(agg, published):
    """Per-architecture absolute deltas of `unlearning_score` vs published SAEBench values.
    `published` maps arch -> {"unlearning_score": x}. Returns {} if not provided."""
    return compare_by_arch_to_reference(agg, published, _UNLEARN_SCORE_KEYS)


# ---------------------------------------------------------------------------
# RAVEL. Upstream already reduces to per-SAE scores (disentanglement = mean of cause + isolation, averaged
# over attributes/entity types); collate per-SAE rows and summarize the three scores per architecture.
# disentanglement_score is the primary ranking column.
# ---------------------------------------------------------------------------
_RAVEL_SCORE_KEYS = ["disentanglement_score", "cause_score", "isolation_score"]
_RAVEL_PER_SAE_KEYS = (
    "sae_name", "arch", "location", "status",
    "disentanglement_score", "cause_score", "isolation_score",
)


def aggregate_ravel(rows):
    """Collate per-SAE RAVEL rows (from ravel.jsonl) into a processed result: mean/std per architecture of
    the disentanglement/cause/isolation scores. Non-ok rows are counted but excluded from the summaries."""
    return aggregate_by_arch(rows, _RAVEL_SCORE_KEYS, _RAVEL_PER_SAE_KEYS, excluded_label="n_failed")


def compare_ravel_to_published(agg, published):
    """Per-architecture absolute deltas of the RAVEL scores vs published SAEBench values.
    `published` maps arch -> {"disentanglement_score": x, ...}. Returns {} if not provided."""
    return compare_by_arch_to_reference(agg, published, _RAVEL_SCORE_KEYS)
