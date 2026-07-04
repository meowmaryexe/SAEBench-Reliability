# Run records

Durable, auto-generated audit trail of suite runs — committed for recordkeeping.

Each run of `scripts/run_absorption_suite.sh` writes `absorption/<suite>_<UTC>/`:
- `run_record.md` — human summary: per-version provenance (package versions, host, device, GPU, git
  SHA+dirty, UTC timestamps), the per-architecture results table (published vs 0.3.2 vs 0.6.0), and the
  embedded report.
- `run_record.json` — machine-readable, self-contained (embeds the by-arch results, so it stands alone
  even though the bulky raw outputs under `results/` are gitignored).
- `report.txt` — the raw suite report.

See `docs/aws_absorption_runbook.md`.
