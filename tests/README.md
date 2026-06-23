# Tests

Methodology-equivalence and unit tests for the Core / Loss Recovered metric. The point of these tests is
to prove there is **no methodology difference** between our implementation and
[adamkarvonen/SAEBench](https://github.com/adamkarvonen/SAEBench) `sae_bench/evals/core/main.py`.

## Files

- **`saebench_verbatim.py`** — verbatim transcription of SAEBench's `get_recons_loss` +
  reduction + L0 path (each block annotated with its source lines). The ground-truth oracle.
- **`test_core_oracle.py`** — loads Pythia-160M as a real `transformer_lens.HookedTransformer`
  (`from_pretrained_no_processing`, exactly as SAEBench), runs the verbatim SAEBench code, and asserts our
  `saebench_audit.metrics.core_loss_recovered.saebench_core_eval` matches on the same tokens + same SAE.
  Passing means our methodology is identical (Δ ≤ 2.4e-5 across all 7 architectures).
- **`test_core_units.py`** — fast, no-LLM unit tests pinning each primitive (formula, per-token CE,
  special-token masking, L0 definition, zero-ablation, SAE forwards, loaders).

## Running

```bash
python tests/test_core_units.py                 # fast, no model
python tests/test_core_oracle.py \
  --model_dir <PYTHIA_DIR> --sae_dir <SAE_DIR> --arch standard   # needs transformer_lens
```

## Test environment

The oracle needs `transformer_lens` (the actual model class SAEBench uses). Install it **without**
disturbing the torch CPU build:

```bash
pip install --no-deps transformer_lens==2.15.4 einops jaxtyping fancy_einsum typeguard beartype rich better_abc sentencepiece
# transformer_lens imports wandb at load (training only) — provide a stub module named `wandb` on the path.
```

`HookedTransformer.from_pretrained_no_processing('pythia-160m-deduped', hf_model=<local HF model>)` loads
in seconds and produces `hook_resid_post` activations identical to the raw HF residual stream.
