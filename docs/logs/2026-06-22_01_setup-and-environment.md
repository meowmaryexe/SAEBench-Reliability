# Log 2026-06-22 #01 — Setup, environment, and reference gathering

**Owner:** Ari · **Milestone:** Core / Loss Recovered reproduction (Stage 1, faithful reproduction)
**Goal for the day:** Stand up an independent Loss Recovered harness and confirm a single eval runs
end-to-end on one released SAE (Deliverable-checklist item: "[Ari] confirm a single eval script runs
end-to-end on one SAE").

## What Loss Recovered is (locked in PREREGISTRATION.md)

SAEBench Eq. 4: `Loss Recovered = (H* - H0) / (H_orig - H0)`, where H_orig is the model's next-token CE,
H* is CE with the layer activation replaced by the SAE reconstruction, and H0 is CE with the activation
zero-ablated. The released code calls this `frac_recovered` and computes it identically
(`dictionary_learning/evaluation.py::loss_recovered`).

## Reference sources pulled (so we implement against ground truth, not memory)

- **Paper:** `Documents/2503.09532v4.pdf` — §3.2.1 (Eq. 4) and Table 4 (Core hyperparameters:
  OpenWebText, ctx 128, 3,200 loss-recovered seqs, 32,000 sparsity seqs).
- **Released Core config:** `sae_bench/evals/core/eval_config.py` — confirms dataset
  `Skylion007/openwebtext`, `context_size=128`, `batch_size_prompts=16`. Table 4's 3,200 / 32,000 counts
  = overriding `n_eval_reconstruction_batches=200` and `n_eval_sparsity_variance_batches=2000`.
- **Reference SAE forward + frac_recovered:** `dictionary_learning/dictionary.py` (AutoEncoder, Gated,
  JumpReLU) and `evaluation.py`. Reimplemented independently in `src/sae_models.py` and
  `src/loss_recovered.py` (we use only the released *weights*, not the library).
- **Validation SAE:** `adamkarvonen/saebench_pythia-160m-deduped_width-2pow12_date-0108`,
  `Standard_.../resid_post_layer_8/trainer_0` (4k width, ReLU). Each released SAE ships
  `config.json` + `eval_results.json` — the authors' own measured numbers for that exact SAE, which we
  use as the ground-truth validation target.

## Compute environment

This reproduction box has **no GPU** (4 CPUs, ~3.8 GB RAM, ~3 GB free disk). Gemma-2-2B @ 65k width is
therefore out of scope to *run* here (project plan budgets ~400–600 A100-hr). Decision (confirmed with
Ari): build the harness as an **independent reimplementation** and validate end-to-end on a CPU-feasible
released SAE (Pythia-160M, 4k), deferring the Gemma run to GPU via the same code path.

### Environment notes / gotchas (for reproducibility)

- Installed CPU-only stack: `torch==2.12.1+cpu`, `transformers==4.57.6`, `datasets==5.0.0`,
  `safetensors`, `huggingface_hub`, `zstandard` (Pile shards are zstd-compressed), `numpy`.
- **Large downloads exceed the per-process wall-clock limit on this box.** torch (150 MB), the model
  (358 MB safetensors) and SAE were fetched with **resumable `curl -C -`** across calls, then installed /
  loaded from local paths. `huggingface_hub` snapshot resume was unreliable here; direct resumable curl of
  the `resolve/main/...` URLs worked.
- **Background processes do not survive across calls** (reaped on call return). The evaluator is therefore
  written to be **checkpoint-resumable** (`src/eval_resumable.py`): each invocation processes as many
  batches as fit in a time budget, appends per-batch metrics to a JSONL, and re-invocation continues.
  Results are independent of how the work is split (fixed pre-tokenized window pool).
- torch can segfault during interpreter finalization on CPU (`PyGILState_Release`) *after* results are
  flushed — handled with a hard `os._exit(0)`.

## Files created

```
core_loss_recovered/
  PREREGISTRATION.md            # tolerances + ranking criterion, locked before running
  src/sae_models.py             # independent SAE forwards (Standard/Gated/JumpReLU/TopK/BatchTopK)
  src/loss_recovered.py         # Eq.4 harness: CE, resid_post hook intervention, L0, tokenize modes
  src/run_eval.py               # single-SAE runner (HF or local paths)
  src/eval_resumable.py         # checkpoint-resumable runner used on this CPU box
  results/                      # per-SAE result JSON + per-batch JSONL
  logs/                         # these logs
```

Next: validation runs + results (see `2026-06-22_02_validation-and-results.md`).
