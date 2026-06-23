# SCR / TPP Run Log

## Goal

Establish a working SCR/TPP execution path before scaling to the full SAEBench reproduction and reliability audit.

The initial target was Pythia-160M layer 8, but the current SAEBench SCR/TPP code path uses `get_saes_from_regex()`, and the Pythia-160M SAEBench releases are not registered in the local SAE Lens pretrained SAE directory. For initial smoke tests, we used the registered Pythia-70M SAE Lens release.

## SAEBench Commit

`8042bb3828c6340da8d12062324e92b2077c571c`

## Files Investigated

SCR/TPP:

- `sae_bench/evals/scr_and_tpp/main.py`
- `sae_bench/evals/scr_and_tpp/eval_config.py`
- `sae_bench/evals/scr_and_tpp/dataset_creation.py`

Sparse Probing:

- `sae_bench/evals/sparse_probing/main.py`
- `sae_bench/evals/sparse_probing/probe_training.py`

## Initial Observations

SCR and TPP share the same implementation directory.

Key configurable parameters:

- `random_seed`
- `probe_epochs`
- `probe_lr`
- `probe_l1_penalty`
- `train_set_size`
- `test_set_size`
- `n_values = [2, 5, 10, 20, 50, 100, 500]`

Potential sources of variance:

- Probe training
- Dataset sampling
- Random negative-class selection
- Feature selection / top-k operations

## Notes from Code Inspection

`prepare_probe_data()`:

- Uses `torch.randperm()`
- Randomly samples negative examples
- Randomly shuffles combined dataset

`train_probe_gpu()`:

- Uses random minibatch ordering each epoch
- Uses AdamW optimizer
- Uses early stopping

Implication:

SCR and TPP are not deterministic unless all relevant random seeds are properly controlled.

## Open Questions

1. Which components are actually controlled by `random_seed`?
2. Does seed variation change architecture rankings?
3. Is variance larger at some L0 values than others?
4. Are SCR and TPP measuring distinct properties or largely the same signal?

## Pythia-160M SAE Selection Blocker

Pythia-160M was the intended first target, but `get_saes_from_regex()` returned zero SAEs for the Pythia-160M SAEBench release names listed in the README.

Current hypothesis: Pythia-160M SAEs may require a custom loading path rather than the SAE Lens registry path used by `scr_and_tpp/main.py`.

For now, Pythia-70M is being used only as a smoke test to verify the SCR/TPP execution path.

## Local Mac Attempt: Full TPP Smoke Test

### Command

```bash
python sae_bench/evals/scr_and_tpp/main.py \
  --sae_regex_pattern "sae_bench_pythia70m_sweep_topk_ctx128_0730" \
  --sae_block_pattern "blocks.4.hook_resid_post__trainer_10" \
  --model_name pythia-70m-deduped \
  --perform_scr false \
  --output_folder eval_results/smoke_tpp_pythia70m
```

### Result

Stopped manually.

### Reason

The process stayed alive but produced no output JSON and appeared to be crawling/stalled on local Apple Silicon MPS.

### Notes

Use a real CUDA GPU or a reduced config for future smoke tests.

## Failed Local Attempt: TPP Acceptance Test on MPS

### Command

```bash
python -m pytest -s tests/acceptance/test_scr_and_tpp.py::test_tpp_end_to_end_different_seed
```

### Result

Failed due to local Apple Silicon MPS out-of-memory.

### Error Summary

```text
RuntimeError: MPS backend out of memory
MPS allocated: 9.91 GiB
Other allocations: 7.84 GiB
Max allowed: 18.13 GiB
Tried to allocate 1.95 GiB
```

### Interpretation

The reduced official TPP acceptance test reached the SAE encoding path but exceeded available local MPS memory. This appears to be a hardware/backend limitation rather than a benchmark implementation issue.

### Next Step

Move SCR/TPP smoke tests to a CUDA GPU environment, starting with Google Colab.

## TPP Acceptance Test on Colab CUDA

Date: 2026-06-22

### Environment

- Google Colab
- CUDA GPU runtime
- Python 3.12

### Command

```bash
python -m pytest -s tests/acceptance/test_scr_and_tpp.py::test_tpp_end_to_end_different_seed
```

### Configuration

From the official acceptance test:

- Dataset: `LabHC/bias_in_bios_class_set1`
- Model: `pythia-70m-deduped`
- SAE release: `sae_bench_pythia70m_sweep_topk_ctx128_0730`
- SAE block: `blocks.4.hook_resid_post__trainer_10`
- Random seed: `44`
- `n_values = [10]`
- `perform_scr = False`
- SAE batch size: `250`
- LLM batch size: `500`

### Result

PASS

### Runtime

Approximately 199 seconds, or 3.3 minutes.

### Observed Output

```text
Global mean difference: 0.008533358573913566
Global max difference: 0.012800037860870361

1 passed in 199.37s
```

### Notes

- Colab CUDA execution completed successfully without modifying benchmark code.
- Acceptance test validates the SCR/TPP execution path, dataset loading, SAE loading, probe training, activation collection, and metric computation.
- This is the first successful end-to-end SCR/TPP benchmark run.

## SCR Acceptance Test on Colab CUDA

Date: 2026-06-22

### Environment

- Google Colab
- CUDA GPU runtime
- Python 3.12

### Command

```bash
python -m pytest -s tests/acceptance/test_scr_and_tpp.py::test_scr_end_to_end_different_seed
```

### Configuration

From the official acceptance test:

- Dataset: `LabHC/bias_in_bios_class_set1`
- Model: `pythia-70m-deduped`
- SAE release: `sae_bench_pythia70m_sweep_topk_ctx128_0730`
- SAE block: `blocks.4.hook_resid_post__trainer_10`
- Random seed: `48`
- `n_values = [10]`
- `perform_scr = True`
- SAE batch size: `250`
- LLM batch size: `500`
- `lower_vram_usage = True`

### Result

PASS

### Runtime

Approximately 190 seconds, or 3.2 minutes.

### Observed Output

```text
scr_score dir 1: 0.8357665734093227
scr_score dir 2: 0.5411764623384583
Global mean difference: 0.07869108679148029
Global max difference: 0.07869108679148029

1 passed in 189.76s
```

### Notes

- SCR acceptance test passes on CUDA without modifying benchmark code.
- This confirms that the local MacBook failure was an MPS/hardware limitation, not a benchmark-code issue.
- SCR and TPP smoke tests now both pass on Colab CUDA.