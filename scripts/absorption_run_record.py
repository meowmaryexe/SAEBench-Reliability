"""
Thin absorption adapter over saebench_audit.run_record.write_run_record: consolidate one absorption suite
run (both sae-bench versions) into a durable `run_record.{json,md}` for recordkeeping. All the generic
assembly (provenance, per-arch table, published-ref, embedded report, partial-run guard) lives in the
shared module; this file only supplies the absorption score columns + version labels.

  python scripts/absorption_run_record.py --suite pythia-160m_4k --record_dir <dir> \
    --v032_workdir results/raw/absorption/pythia-160m_4k_v0.3.2 \
    --v060_workdir results/raw/absorption/pythia-160m_4k_v0.6.0 \
    --v032_processed results/processed/absorption/pythia-160m_4k_v0.3.2.json \
    --v060_processed results/processed/absorption/pythia-160m_4k_v0.6.0.json \
    --report <record_dir>/report.txt

Called by scripts/run_absorption_suite.sh.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from saebench_audit.run_record import write_run_record

SCORE_KEYS = [("mean_absorption_fraction_score", "frac"), ("mean_full_absorption_score", "full")]


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

    report_txt = open(args.report).read() if args.report and os.path.exists(args.report) else None
    json_path, md_path = write_run_record(
        args.record_dir, metric="absorption", suite=args.suite,
        version_workdirs={"v0.3.2": args.v032_workdir, "v0.6.0": args.v060_workdir},
        version_processed={"v0.3.2": args.v032_processed, "v0.6.0": args.v060_processed},
        version_labels={"v0.3.2": "v0.3.2 (published-match)", "v0.6.0": "v0.6.0 (current)"},
        published_path=os.path.join(args.v032_workdir, "published_ref.json"),
        score_keys=SCORE_KEYS, report_text=report_txt, full_suite_n=42)
    print(f"[record] {json_path}")
    print(f"[record] {md_path}")


if __name__ == "__main__":
    main()
