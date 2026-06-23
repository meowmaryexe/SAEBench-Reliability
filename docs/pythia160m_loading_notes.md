# Pythia-160M SAE Loading Notes

## Finding

Pythia-160M SAEBench SAEs are not available through the SAE Lens pretrained SAE registry used by `get_saes_from_regex()`.

The intended path appears to be the dictionary-learning custom SAE runner:

`sae_bench/custom_saes/run_all_evals_dictionary_learning_saes.py`

## Relevant Code

- `get_all_hf_repo_autoencoders(repo_id)`
- `load_dictionary_learning_sae(repo_id, location, model_name, device, dtype)`
- `run_evals(...)`

## Pythia-160M Repo

`adamkarvonen/saebench_pythia-160m-deduped_width-2pow14_date-0108`

## Model Config

From `MODEL_CONFIGS`:

- model: `pythia-160m-deduped`
- layer: 8
- d_model: 768
- llm_batch_size: 256
- dtype: `float32`

## Implication

For faithful reproduction on Pythia-160M, we should not rely on the acceptance-test SAE Lens registry path. We need a project-owned runner that loads dictionary-learning SAEs directly from HuggingFace and then calls the SAEBench eval functions for Mary-owned metrics:

- TPP
- SCR
- Sparse Probing

## Next Step

Build a minimal runner that:

1. Downloads / enumerates Pythia-160M dictionary-learning SAE locations.
2. Filters to one SAE location for smoke testing.
3. Runs only `tpp`, `scr`, and `sparse_probing`.
4. Writes outputs under `results/raw/reproduction_smoke/`.
