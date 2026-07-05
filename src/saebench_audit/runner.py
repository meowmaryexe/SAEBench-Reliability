"""
Generic resumable per-SAE suite driver shared by the wrap-upstream metrics.

`run_sae_suite` walks a list of SAE locations, skips ones already in the JSONL ledger, optionally resumes
from an upstream output already on disk (without reloading the model), time-boxes the wall clock, isolates
per-SAE errors so one bad SAE can't halt the suite, and appends one row per SAE. `write_run_meta` writes
the sidecar; `announce` prints the `ALL_SAES_DONE` / `PROGRESS n/N` sentinel the shell suite driver greps.

A per-metric runner supplies three small callables (name_fn, process_one, and optionally resume_row_fn /
row_extra_fn) and gets the whole resumable loop for free — this is what makes Unlearning/RAVEL thin.
"""
from __future__ import annotations

import json
import os
import time


def _load_ledger(path: str, key: str) -> dict:
    """Load a JSONL ledger into {record[key]: record}. Torch-free (kept out of io.py so the runner and
    its fast unit tests import without torch); `key` is the per-SAE dedup field (default "sae_name")."""
    done = {}
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line:
                r = json.loads(line)
                done[r[key]] = r
    return done


def _append_ledger(path: str, record: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def run_sae_suite(
    locations,
    *,
    workdir: str,
    process_one,
    name_fn,
    row_extra_fn=None,
    resume_row_fn=None,
    ledger_name: str,
    key: str = "sae_name",
    max_seconds: float = 1e9,
    force_rerun: bool = False,
    tag: str | None = None,
    verbose: bool = True,
) -> dict:
    """Run `process_one(location, name)` over every location, resumably. Returns the `done` ledger dict.

    - name_fn(location) -> str: the ledger dedup key (unique per SAE).
    - process_one(location, name) -> dict: does the load + eval, returns a flat result row (may raise;
      an exception is caught and recorded as a {"status": "error", ...} row so the suite keeps going).
    - resume_row_fn(location, name) -> dict | None: if the upstream output already exists on disk, return
      a result row to record it without reloading the model; return None to fall through to process_one.
      Skipped when force_rerun is set.
    - row_extra_fn(location, name) -> dict: extra fields merged into every recorded row (e.g. arch/location).
    """
    tag = tag or ledger_name.rsplit(".", 1)[0]
    os.makedirs(workdir, exist_ok=True)
    summary_path = os.path.join(workdir, ledger_name)
    done = _load_ledger(summary_path, key)

    def _record(row):
        _append_ledger(summary_path, row)
        done[row[key]] = row

    t0, processed, n_total = time.time(), 0, len(locations)
    for location in locations:
        name = name_fn(location)
        if name in done:
            continue
        extra = row_extra_fn(location, name) if row_extra_fn else {}
        # Resume without reloading the model if the upstream output already exists.
        if resume_row_fn is not None and not force_rerun:
            res = resume_row_fn(location, name)
            if res is not None:
                _record({key: name, **extra, **res})
                continue
        if processed > 0 and time.time() - t0 > max_seconds:
            break

        if verbose:
            print(f"[{tag}] {name}", flush=True)
        try:
            res = process_one(location, name)
        except Exception as e:  # one bad SAE must not halt (or infinite-loop) the whole suite
            print(f"[{tag}] ERROR on {name}: {type(e).__name__}: {e}", flush=True)
            res = {"status": "error", key: name, "error": f"{type(e).__name__}: {e}"}
        _record({key: name, **extra, **res})
        processed += 1

    return done


def write_run_meta(workdir: str, **fields) -> str:
    """Write the run_meta.json sidecar next to the ledger (config + provenance + suite metadata)."""
    path = os.path.join(workdir, "run_meta.json")
    json.dump(fields, open(path, "w"), indent=2)
    return path


def announce(done: dict, total: int, extra_counts: dict | None = None,
             sentinel: str = "ALL_SAES_DONE") -> None:
    """Print the completion sentinel (`ALL_SAES_DONE (...)`) or `PROGRESS n/total` for the shell driver.

    extra_counts maps a display label -> the status string to count, e.g.
    {"insufficient_features": "insufficient_features"} -> "40 ok, 2 insufficient_features of 42".
    """
    import sys

    n_ok = len([r for r in done.values() if r.get("status") == "ok"])
    parts = [f"{n_ok} ok"]
    for label, status_val in (extra_counts or {}).items():
        parts.append(f"{len([r for r in done.values() if r.get('status') == status_val])} {label}")
    if len(done) >= total:
        print(f"{sentinel} ({', '.join(parts)} of {total})", flush=True)
    else:
        print(f"PROGRESS {len(done)}/{total}", flush=True)
    sys.stdout.flush()
