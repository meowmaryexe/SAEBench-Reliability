# Log 2026-06-22 #09 — GPU config setup (full Core reproduction)

Set up (did **not** run) the configs to reproduce Core / Loss Recovered at paper scale on a GPU, using
the `saebench_core` methodology (proven identical to `core/main.py`, log #08).

## Added
- `configs/registry.yaml` (rewritten) — all 6 SAE suites with verified HF repos + per-suite folder
  layouts. Two naming conventions confirmed on HuggingFace 2026-06-22:
  - **adamkarvonen** (Pythia 4k/16k/65k `date-0108`, Gemma-2-2B 4k `date-0108`):
    `Standard_<modeltag>__<date>`, `TopK_…`, `BatchTopK_…`, `JumpRelu_…`, `GatedSAE_…`,
    `MatryoshkaBatchTopK_…`, `PAnneal_…`.
  - **canrager** (Gemma-2-2B 16k `2pow14` / 65k `2pow16`, `date-0107`):
    `gemma-2-2b_standard_new_width-2pow<W>_date-0107`, `…_top_k_…`, `…_batch_top_k_…`, `…_gated_…`,
    `…_jump_relu_…`, `…_matryoshka_batch_top_k_…`, `…_p_anneal_…`.
  - subpath = `<folder>/resid_post_layer_<L>/trainer_<i>` (L=8 Pythia, L=12 Gemma; i in 0..5).
  - reference values: HF dataset `adamkarvonen/sae_bench_results_0125` + Neuronpedia.
- `configs/gpu/core_gpu.yaml` — canonical eval config (Table 4: OWT, ctx128, 3200 recon / 32000 sparsity,
  batch 16; CE & L0 exclude {bos,eos,pad}; reconstruct all positions; device cuda; dtype float32).
- `configs/gpu/jobs.yaml` — full 252-SAE job matrix in compute tiers (Tier 0 = 65k Gemma headline;
  Tier 1 = scaling + cross-model; full = 6 suites). Per-suite minute estimates + GPU-hour totals
  (~11 GPU-h Tier 0, ~40 GPU-h full on 1×A100).
- `configs/gpu/README.md` — launch commands + environment (note: Gemma is gated → HF login) + the
  post-run cross-check against `sae_bench_results_0125`.
- `scripts/run_core_gpu.py` — GPU runner that consumes the configs and calls the **tested**
  `saebench_core_eval` (+ a dedicated L0 pass over the 32k sparsity set). Resolves repo/folder from the
  registry, downloads each `ae.pt` on demand and deletes after eval.

## Validation (config-level, no eval run)
- All YAMLs parse; runner parses + imports.
- Folder resolution checked against the actual HF folder names for 7 (suite, arch) spot-cases — all match.
- Job matrix resolves to exactly **252 SAEs** (6 suites × 7 arch × 6 sparsity).

## Caveats to confirm at runtime
- Gemma-2-2B is a gated model (accept license / `huggingface-cli login`).
- Trainer indices assumed `0..5` (6 sparsities) for all suites per the paper; the runner fail-softs if a
  trainer is absent. The canrager inner path `resid_post_layer_12/trainer_<i>` was confirmed for the
  TopK/65k folder; other canrager arch folders assumed identical.
- `dtype: float32` matches CoreEvalConfig's default; use `bfloat16` for Gemma on <40GB GPUs (perturbs
  low-order digits).

## Not done (by request)
No GPU execution. Running any suite (start with `--suite gemma-2-2b_65k`) is the next step once a GPU is
available; then aggregate + compare to `sae_bench_results_0125`.
