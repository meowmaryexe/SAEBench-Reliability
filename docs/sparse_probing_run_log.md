# Sparse Probing Run Log

## Goal

Establish a working Sparse Probing execution path before scaling to faithful reproduction and reliability audit runs.

As with SCR and TPP, the initial objective is to verify that the official Sparse Probing benchmark runs successfully on CUDA, archive the outputs, and establish a reproducible workflow.

## SAEBench Commit

`8042bb3828c6340da8d12062324e92b2077c571c`

## Files Investigated

Sparse Probing:

- `sae_bench/evals/sparse_probing/main.py`
- `sae_bench/evals/sparse_probing/eval_config.py`
- `sae_bench/evals/sparse_probing/probe_training.py`
- `tests/acceptance/test_sparse_probing.py`

## Initial Observations

Sparse Probing trains probes using SAE features and compares performance against probes trained on model activations.

Key configurable parameters:

- `random_seed`
- `k_values`
- `llm_batch_size`
- `llm_dtype`
- `lower_vram_usage`

Potential sources of variance:

- Probe training
- Dataset sampling
- Feature selection
- Random initialization

## Acceptance Test Configuration

- Dataset: `LabHC/bias_in_bios_class_set1`
- Model: `pythia-70m-deduped`
- SAE release: `sae_bench_pythia70m_sweep_topk_ctx128_0730`
- SAE block: `blocks.4.hook_resid_post__trainer_10`
- Random seed: `44`
- `k_values = [1, 2, 5, 10, 20, 50, 100]`

## Sparse Probing Acceptance Test on Colab CUDA

### Command

```bash
python -m pytest -s tests/acceptance/test_sparse_probing.py::test_end_to_end_different_seed
```

### Environment

- Google Colab
- CUDA GPU runtime
- Python 3.12

### Result

PASS

### Artifact

- `results/raw/smoke_tests/sparse_probing/sae_bench_pythia70m_sweep_topk_ctx128_0730_blocks.4.hook_resid_post__trainer_10_eval_results.json`

### Notes

- Acceptance test reproduced successfully on CUDA.
- Colab workflow established for future Sparse Probing experiments.
- Sparse Probing now joins SCR and TPP as a verified benchmark execution path.

## Open Questions

1. How sensitive are Sparse Probing scores to random seed?
2. Does metric variance exceed architecture-level score differences?
3. Are some `k` values more stable than others?
4. How correlated are Sparse Probing rankings with SCR and TPP rankings?

## Current Status

Completed:

- Sparse Probing acceptance test reproduced successfully on CUDA.
- Raw benchmark output archived.
- CUDA execution path verified.

Next immediate objectives:

1. Build a reusable Sparse Probing reproduction script.
2. Investigate the Pythia-160M evaluation path.
3. Run faithful reproduction experiments on target benchmark SAEs.