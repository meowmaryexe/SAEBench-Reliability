# SCR / TPP Run Log

## Goal

Run SCR and TPP smoke tests to establish the execution path before scaling to the full SAEBench reproduction.

Initial target was Pythia-160M layer 8, but the current SAEBench code path uses `get_saes_from_regex()`, and the Pythia-160M SAEBench releases are not registered in the local SAE Lens pretrained SAE directory. For the first smoke test, we are using the registered Pythia-70M SAE Lens release.

## SAEBench commit

`8042bb3828c6340da8d12062324e92b2077c571c`

### Files Investigated

SCR/TPP:

- sae_bench/evals/scr_and_tpp/main.py

- sae_bench/evals/scr_and_tpp/eval_config.py

- sae_bench/evals/scr_and_tpp/dataset_creation.py

Sparse Probing:

- sae_bench/evals/sparse_probing/main.py

- sae_bench/evals/sparse_probing/probe_training.py

### Initial Observations

SCR and TPP share the same implementation directory.

Key configurable parameters:

- random_seed

- probe_epochs

- probe_lr

- probe_l1_penalty

- train_set_size

- test_set_size

- n_values = [2, 5, 10, 20, 50, 100, 500]

Potential sources of variance:

- Probe training

- Dataset sampling

- Random negative-class selection

- Feature selection / top-k operations

### Notes from Code Inspection

prepare_probe_data():

- Uses torch.randperm()

- Randomly samples negative examples

- Randomly shuffles combined dataset

train_probe_gpu():

- Uses random minibatch ordering each epoch

- Uses AdamW optimizer

- Early stopping enabled

Implication:

SCR and TPP are not deterministic unless all random seeds are properly controlled.

### Questions

1. Which components are actually controlled by random_seed?

2. Does seed variation change architecture rankings?

3. Is variance larger at some L0 values than others?

4. Are SCR and TPP measuring distinct properties or largely the same signal?

### Current Status

- Successfully installed SAEBench dependencies.

- Confirmed sae_lens import.

- Running initial TPP smoke test on Pythia benchmark SAE.

- Monitoring runtime and output behavior.

## Current Smoke Test

Metric: TPP  
Model: `pythia-70m-deduped`  
SAE release: `sae_bench_pythia70m_sweep_topk_ctx128_0730`  
SAE block: `blocks.4.hook_resid_post__trainer_10`  
Device: Apple Silicon MPS  

Command:

```bash
python sae_bench/evals/scr_and_tpp/main.py \
  --sae_regex_pattern "sae_bench_pythia70m_sweep_topk_ctx128_0730" \
  --sae_block_pattern "blocks.4.hook_resid_post__trainer_10" \
  --model_name pythia-70m-deduped \
  --perform_scr false \
  --output_folder eval_results/smoke_tpp_pythia70m
```

## Errors / Blockers

### Pythia-160M SAE Selection

Pythia-160M was the intended first target, but `get_saes_from_regex()` returned zero SAEs for the Pythia-160M SAEBench release names listed in the README.

Current hypothesis: Pythia-160M SAEs may require a custom loading path rather than the SAE Lens registry path used by `scr_and_tpp/main.py`.

For now, Pythia-70M is being used only as a smoke test to verify the SCR/TPP execution path.