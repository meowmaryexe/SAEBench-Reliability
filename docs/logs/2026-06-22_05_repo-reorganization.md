# Log 2026-06-22 #05 — Repository reorganization

Reorganized the Core / Loss Recovered work from a single `core_loss_recovered/` folder into the agreed
top-level research-repo layout (`configs/ docs/ figures/ results/ scripts/ src/`). No results changed —
this is purely a move + import-path update + gap-fill.

## Moves

| From | To |
|---|---|
| `core_loss_recovered/src/sae_models.py`, `loss_recovered.py` | `src/saebench_repro/` (now an importable package) |
| `core_loss_recovered/src/run_eval.py`, `eval_resumable.py`, `eval_bundle_exact.py` | `scripts/` |
| `core_loss_recovered/PREREGISTRATION.md` | `docs/` |
| `core_loss_recovered/README.md` | `docs/core_loss_recovered.md` |
| `core_loss_recovered/logs/*.md` | `docs/logs/` |
| `core_loss_recovered/results/*` | `results/core_loss_recovered/` |

## Gaps filled

- **`src/saebench_repro/__init__.py`** — turns the library into a package; scripts now do
  `sys.path.insert(.., "../src")` + `from saebench_repro... import ...` (updated in all 3 scripts; imports
  re-tested OK).
- **`configs/core_loss_recovered/`** — extracted the three run configs to YAML
  (`paper_owt_ctx128`, `bundle_match_pile_ctx1024`, `bundle_exact_perdoc_ctx1024`) plus `registry.yaml`
  (base models + released SAE suites + what's validated). Configs carry both CPU-validation counts and the
  full Table-4 GPU counts.
- **`scripts/make_figures.py`** — dependency-free SVG figure generation from `results/`. Produced
  `figures/core_lr_mine_vs_bundle.svg` and `figures/core_lr_convergence.svg`.
- **`README.md`** (repo root) — enriched with structure table, status, quickstart (preserving the
  existing intro + references).
- **`docs/repo_structure.md`** — file-by-file map + conventions (src=library, scripts=entry points,
  per-metric subfolders, generated figures, gitignored caches).
- **`.gitignore`** (repo root) — ignores weights/caches/`work_*` scratch.

## Note on cleanup

This mount blocks file deletion (`rm` → "Operation not permitted"); creation/move/rename work. Files were
**moved** out of `core_loss_recovered/`, leaving empty old directories that I could not `rmdir`. Requested
delete permission to remove the empty `core_loss_recovered/` tree (and a stray `_perm_test/`); if any
empty shells remain they contain no files and can be deleted from the host.

## Verification

- `import saebench_repro` and all 3 scripts import cleanly from the new paths.
- `make_figures.py` runs and regenerates both SVGs from the result JSON/JSONL.
- Aggregate-vs-per-batch consistency was re-checked earlier (log #02) and is unaffected by the move.
