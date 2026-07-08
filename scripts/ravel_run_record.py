"""
Thin RAVEL adapter over saebench_audit.run_record.write_run_record: consolidate one RAVEL suite run into a
durable `run_record.{json,md}` for recordkeeping. Single-version (no known sae-bench version drift for
RAVEL). The generic assembly (provenance, per-arch table, published-ref, embedded report, partial-run
guard) lives in the shared module; this file supplies the RAVEL score columns.

  python scripts/ravel_run_record.py --suite gemma-2-2b_4k --record_dir <dir> \
    --workdir results/raw/ravel/gemma-2-2b_4k \
    --processed results/processed/ravel/gemma-2-2b_4k.json \
    --report <record_dir>/report.txt

Called by scripts/run_ravel_suite.sh.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from saebench_audit.run_record import write_run_record

SCORE_KEYS = [("disentanglement_score", "disent"), ("cause_score", "cause"), ("isolation_score", "iso")]


def main():
    ap = argparse.ArgumentParser(description="Write a durable RAVEL suite run record (json + md)")
    ap.add_argument("--suite", required=True)
    ap.add_argument("--record_dir", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--processed", required=True)
    ap.add_argument("--report", help="path to the captured report text")
    args = ap.parse_args()

    report_txt = open(args.report).read() if args.report and os.path.exists(args.report) else None
    json_path, md_path = write_run_record(
        args.record_dir, metric="ravel", suite=args.suite,
        version_workdirs={"v0.6.0": args.workdir},
        version_processed={"v0.6.0": args.processed},
        version_labels={"v0.6.0": "v0.6.0 (current)"},
        published_path=os.path.join(args.workdir, "published_ref.json"),
        score_keys=SCORE_KEYS, report_text=report_txt, full_suite_n=42)
    print(f"[record] {json_path}")
    print(f"[record] {md_path}")


if __name__ == "__main__":
    main()
