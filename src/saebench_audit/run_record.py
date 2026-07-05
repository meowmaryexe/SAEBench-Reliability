"""
Consolidate one metric suite run (across pinned upstream versions) into a durable, auditable **run
record**: `run_record.json` (machine) + `run_record.md` (human). Bundles provenance (versions, git SHA,
host, GPU, timestamps from each run's run_meta.json), the per-architecture results for each version, the
published reference, and the report text — so a run stays reproducible/attributable after the box is gone.

Generic over the metric's headline score columns (`score_keys`), so Absorption/Unlearning/RAVEL all reuse
it. Results live under gitignored results/; the committed run record embeds everything and stands alone.
Pure stdlib.
"""
from __future__ import annotations

import datetime
import json
import os


def _load(path):
    return json.load(open(path)) if path and os.path.exists(path) else None


def _by_arch(processed, score_keys):
    """arch -> {score_key: mean, ..., 'n': n} from a processed metric JSON."""
    out = {}
    for arch, s in (processed or {}).get("result", {}).get("by_arch", {}).items():
        row = {k: (s.get(k) or {}).get("mean") for k, _ in score_keys}
        row["n"] = next(((s.get(k) or {}).get("n") for k, _ in score_keys if s.get(k)), None)
        out[arch] = row
    return out


def write_run_record(
    record_dir: str,
    *,
    metric: str,
    suite: str,
    version_workdirs: dict,
    version_processed: dict,
    version_labels: dict,
    published_path: str | None,
    score_keys,
    report_text: str | None = None,
    full_suite_n: int | None = None,
) -> tuple[str, str]:
    """Write run_record.{json,md} into record_dir. `version_*` dicts are keyed by version tag (insertion
    order preserved for column order); `score_keys` is a list of (result_key, short_label)."""
    tags = list(version_processed.keys())
    metas = {t: (_load(os.path.join(version_workdirs[t], "run_meta.json")) or {}) for t in tags}
    procs = {t: _load(version_processed[t]) for t in tags}
    published = _load(published_path) or {}
    by_arch = {t: _by_arch(procs[t], score_keys) for t in tags}
    n_ok = {t: (procs[t] or {}).get("result", {}).get("n_ok", 0) for t in tags}

    record = {
        "metric": metric,
        "suite": suite,
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "runs": {
            t: {"provenance": metas[t].get("provenance"),
                "sae_bench_version": metas[t].get("sae_bench_version"),
                "config": metas[t].get("config"), "n_saes": metas[t].get("n_saes"),
                "result": (procs[t] or {}).get("result", {}).get("by_arch"),
                "n_ok": n_ok[t]}
            for t in tags
        },
        "published_ref": {a: {**{k: v.get(k) for k, _ in score_keys}, "n_trainers": v.get("n_trainers")}
                          for a, v in published.items()},
        "report_text": report_text,
        "artifacts": {**{f"{t}_workdir": version_workdirs[t] for t in tags},
                      **{f"{t}_processed": version_processed[t] for t in tags}},
    }
    os.makedirs(record_dir, exist_ok=True)
    json_path = os.path.join(record_dir, "run_record.json")
    json.dump(record, open(json_path, "w"), indent=2)

    # ---- human-readable markdown ----
    def _f(x):
        return f"{x:.4f}" if isinstance(x, (int, float)) else "—"

    L = [f"# {metric} run record — {suite}", f"\n_generated {record['generated_utc']}_\n"]
    for t in tags:
        p = metas[t].get("provenance") or {}
        pk = p.get("packages") or {}
        L.append(f"## {version_labels.get(t, t)} — {n_ok[t]}/{metas[t].get('n_saes', '?')} SAEs ok")
        L.append(f"- sae-bench `{pk.get('sae-bench')}` · sae_lens `{pk.get('sae_lens')}` · "
                 f"transformer_lens `{pk.get('transformer_lens')}` · transformers `{pk.get('transformers')}` · "
                 f"torch `{pk.get('torch')}`")
        L.append(f"- host `{p.get('hostname')}` · device `{p.get('device')}` · gpu `{p.get('gpu')}` · "
                 f"git `{(p.get('git_sha') or '')[:10]}`{'(dirty)' if p.get('git_dirty') else ''} · "
                 f"finished `{p.get('timestamp_utc')}`\n")

    # per-arch table: for each score, a published column then one column per version
    header = ["arch"]
    for _, lab in score_keys:
        header.append(f"published {lab}")
        header += [f"{t} {lab}" for t in tags]
    L.append("## Per-architecture")
    L.append("\n| " + " | ".join(header) + " |")
    L.append("|" + "---|" * len(header))
    for arch in sorted(set().union(*(by_arch[t] for t in tags), published)):
        cells = [arch]
        for k, _ in score_keys:
            cells.append(_f(published.get(arch, {}).get(k)))
            cells += [_f(by_arch[t].get(arch, {}).get(k)) for t in tags]
        L.append("| " + " | ".join(cells) + " |")

    if full_suite_n is not None and any(n_ok[t] < full_suite_n for t in tags):
        counts = ", ".join(f"{t} n_ok={n_ok[t]}" for t in tags)
        L.append(f"\n⚠ **Partial run** ({counts}; full suite = {full_suite_n}). Per-arch deltas vs the "
                 f"published means are not meaningful until all trainers per arch run.")
    if report_text:
        L.append("\n## Report\n```\n" + report_text.rstrip() + "\n```")
    md_path = os.path.join(record_dir, "run_record.md")
    open(md_path, "w").write("\n".join(L) + "\n")
    return json_path, md_path
