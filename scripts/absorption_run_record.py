"""
Consolidate one absorption suite run (both sae-bench versions) into a durable, auditable **run record**:
`run_record.json` (machine) + `run_record.md` (human). Bundles provenance (versions, git SHA, host, GPU,
timestamps from each run's run_meta.json), the per-architecture results for each version, the published
reference, and the report text — so a run is fully reproducible/attributable after the box is gone.

  python scripts/absorption_run_record.py --suite pythia-160m_4k --record_dir <dir> \
    --v032_workdir results/raw/absorption/pythia-160m_4k_v0.3.2 \
    --v060_workdir results/raw/absorption/pythia-160m_4k_v0.6.0 \
    --v032_processed results/processed/absorption/pythia-160m_4k_v0.3.2.json \
    --v060_processed results/processed/absorption/pythia-160m_4k_v0.6.0.json \
    --report <record_dir>/report.txt

Called by scripts/run_absorption_suite.sh; pure stdlib.
"""
import argparse
import datetime
import json
import os


def _load(path):
    return json.load(open(path)) if path and os.path.exists(path) else None


def _by_arch(processed):
    out = {}
    for arch, s in (processed or {}).get("result", {}).get("by_arch", {}).items():
        fr, fu = s.get("mean_absorption_fraction_score"), s.get("mean_full_absorption_score")
        out[arch] = {"fraction": fr["mean"] if fr else None, "full": fu["mean"] if fu else None,
                     "n": (fr or fu or {}).get("n")}
    return out


def main():
    ap = argparse.ArgumentParser(description="Write a durable absorption suite run record (json + md)")
    ap.add_argument("--suite", required=True)
    ap.add_argument("--record_dir", required=True)
    ap.add_argument("--v032_workdir", required=True)
    ap.add_argument("--v060_workdir", required=True)
    ap.add_argument("--v032_processed", required=True)
    ap.add_argument("--v060_processed", required=True)
    ap.add_argument("--report", help="path to the captured report text")
    args = ap.parse_args()

    meta032 = _load(os.path.join(args.v032_workdir, "run_meta.json")) or {}
    meta060 = _load(os.path.join(args.v060_workdir, "run_meta.json")) or {}
    proc032, proc060 = _load(args.v032_processed), _load(args.v060_processed)
    published = _load(os.path.join(args.v032_workdir, "published_ref.json")) or {}
    report_txt = open(args.report).read() if args.report and os.path.exists(args.report) else None

    record = {
        "suite": args.suite,
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "runs": {
            "v0.3.2": {"provenance": meta032.get("provenance"), "sae_bench_version": meta032.get("sae_bench_version"),
                       "config": meta032.get("config"), "n_saes": meta032.get("n_saes"),
                       "result": (proc032 or {}).get("result", {}).get("by_arch"),
                       "n_ok": (proc032 or {}).get("result", {}).get("n_ok")},
            "v0.6.0": {"provenance": meta060.get("provenance"), "sae_bench_version": meta060.get("sae_bench_version"),
                       "config": meta060.get("config"), "n_saes": meta060.get("n_saes"),
                       "result": (proc060 or {}).get("result", {}).get("by_arch"),
                       "n_ok": (proc060 or {}).get("result", {}).get("n_ok")},
        },
        "published_ref": {a: {"fraction": v.get("mean_absorption_fraction_score"),
                              "full": v.get("mean_full_absorption_score"), "n_trainers": v.get("n_trainers")}
                          for a, v in published.items()},
        "report_text": report_txt,
        "artifacts": {"v032_workdir": args.v032_workdir, "v060_workdir": args.v060_workdir,
                      "v032_processed": args.v032_processed, "v060_processed": args.v060_processed},
    }
    os.makedirs(args.record_dir, exist_ok=True)
    json.dump(record, open(os.path.join(args.record_dir, "run_record.json"), "w"), indent=2)

    # ---- human-readable markdown ----
    a032, a060 = _by_arch(proc032), _by_arch(proc060)
    n032 = (proc032 or {}).get("result", {}).get("n_ok", 0)
    n060 = (proc060 or {}).get("result", {}).get("n_ok", 0)
    L = []
    L.append(f"# Absorption run record — {args.suite}")
    L.append(f"\n_generated {record['generated_utc']}_\n")
    for tag, meta, n in [("v0.3.2 (published-match)", meta032, n032), ("v0.6.0 (current)", meta060, n060)]:
        p = meta.get("provenance") or {}
        pk = p.get("packages") or {}
        L.append(f"## {tag} — {n}/{meta.get('n_saes','?')} SAEs ok")
        L.append(f"- sae-bench `{pk.get('sae-bench')}` · sae_lens `{pk.get('sae_lens')}` · "
                 f"transformer_lens `{pk.get('transformer_lens')}` · transformers `{pk.get('transformers')}` · "
                 f"torch `{pk.get('torch')}`")
        L.append(f"- host `{p.get('hostname')}` · device `{p.get('device')}` · gpu `{p.get('gpu')}` · "
                 f"git `{(p.get('git_sha') or '')[:10]}`{'(dirty)' if p.get('git_dirty') else ''} · "
                 f"finished `{p.get('timestamp_utc')}`\n")
    L.append("## Per-architecture (fraction | full)")
    L.append(f"\n| arch | published frac | 0.3.2 | 0.6.0 | published full | 0.3.2 | 0.6.0 |")
    L.append("|---|---|---|---|---|---|---|")
    for arch in sorted(set(a032) | set(a060) | set(published)):
        pf = published.get(arch, {}).get("mean_absorption_fraction_score")
        pu = published.get(arch, {}).get("mean_full_absorption_score")
        def _f(x): return f"{x:.4f}" if isinstance(x, (int, float)) else "—"
        L.append(f"| {arch} | {_f(pf)} | {_f(a032.get(arch,{}).get('fraction'))} | "
                 f"{_f(a060.get(arch,{}).get('fraction'))} | {_f(pu)} | "
                 f"{_f(a032.get(arch,{}).get('full'))} | {_f(a060.get(arch,{}).get('full'))} |")
    if n032 < 42 or n060 < 42:
        L.append(f"\n⚠ **Partial run** (0.3.2 n_ok={n032}, 0.6.0 n_ok={n060}; full suite = 42). Per-arch "
                 f"deltas vs the published 6-trainer means are not meaningful until all trainers per arch run.")
    if report_txt:
        L.append("\n## Report\n```\n" + report_txt.rstrip() + "\n```")
    open(os.path.join(args.record_dir, "run_record.md"), "w").write("\n".join(L) + "\n")

    print(f"[record] {os.path.join(args.record_dir, 'run_record.json')}")
    print(f"[record] {os.path.join(args.record_dir, 'run_record.md')}")


if __name__ == "__main__":
    main()
