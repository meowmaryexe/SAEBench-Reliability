# Sparse Probing Run Log

## Goal

Establish a working Sparse Probing execution path before scaling to faithful reproduction and reliability audit runs.

The immediate objective is to reproduce the official acceptance test on CUDA, archive the outputs, and then build a reusable smoke-test/reproduction script.

## SAEBench Commit

`8042bb3828c6340da8d12062324e92b2077c571c`

## Files Investigated

Sparse Probing:

- `sae_bench/evals/sparse_probing/main.py`
- `sae_bench/evals/sparse_probing/eval_config.py`
- `sae_bench/evals/sparse_probing/probe_training.py`
- `tests/acceptance/test_sparse_probing.py`

## Acceptance Test Configuration

From `tests/acceptance/test_sparse_probing.py`:

- Dataset: `LabHC/bias_in_bios_class_set1`
- Model: `pythia-70m-deduped`
- SAE release: `sae_bench_pythia70m_sweep_topk_ctx128_0730`
- SAE block: `blocks.4.hook_resid_post__trainer_10`
- Random seed: `44`
- LLM batch size: `512`
- LLM dtype: `float32`
- `lower_vram_usage = True`
- `k_values = [1, 2, 5, 10, 20, 50, 100]`

## Metrics Compared in Acceptance Test

LLM metrics:

- `llm_test_accuracy`
- `llm_top_1_test_accuracy`
- `llm_top_2_test_accuracy`
- `llm_top_5_test_accuracy`
- `llm_top_10_test_accuracy`
- `llm_top_20_test_accuracy`
- `llm_top_50_test_accuracy`
- `llm_top_100_test_accuracy`

SAE metrics:

- `sae_top_1_test_accuracy`
- `sae_top_2_test_accuracy`
- `sae_top_5_test_accuracy`
- `sae_top_10_test_accuracy`
- `sae_top_20_test_accuracy`
- `sae_top_50_test_accuracy`
- `sae_top_100_test_accuracy`

Tolerance:

- `0.04`

## Planned CUDA Smoke Test

Command:

```bash
python -m pytest -s tests/acceptance/test_sparse_probing.py::test_end_to_end_different_seed