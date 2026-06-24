# Log 2026-06-23 #12 — Methodology audit follow-up (Core + AutoInterp)

Re-audited six specific methodology points against the code and the published `sae_bench_results_0125`
eval_configs. Result: **5/6 already faithful; 1 Core flag was wrong (now fixed)**.

## Verified faithful (no change)

**Core**
- **Zero-ablation** Loss Recovered: `torch.zeros_like(resid_post)`; `(H*−H0)/(Horig−H0)`. Not mean-ablation.
- **High-frequency cutoffs**: `freq_over_1_percent = (density > 0.01).mean()`,
  `freq_over_10_percent = (density > 0.1).mean()` — oracle-confirmed vs SAEBench verbatim.

**AutoInterp**
- **Raw (unbalanced) accuracy**: `score = Σ(pred==actual)/14` over all 14 (4 positive + 10 negative).
  Not balanced accuracy / F1.
- **Importance sampling ∝ activation²**: `get_iw_sample_indices(use_squared_values=True): x = x.pow(2)`.
- **Dead-latent threshold = 15**: `dead_latent_threshold = 15`; `counts > 15` selects alive latents.

## Fixed: Core `exclude_special_tokens_from_reconstruction`

The published 0125 Core run used **`exclude_special_tokens_from_reconstruction = True`** (verified directly
in the released `eval_config` of every core result file). Our setup had it **False**:
- `configs/gpu/core_gpu.yaml` had `false`;
- the CPU 42-SAE validation (`run_core_full_suite.py`) reconstructs all positions (= False);
- the oracle test exercised the function-default (False) on both sides.

**Measured impact** (Standard 4k t0, 48 OWT seqs): Loss Recovered = 0.9834 (True) vs 0.9835 (False) — a
**~1e-4** difference. The flag only changes whether the SAE reconstructs the BOS/EOS/PAD positions (whose
own loss is already excluded from the mean); it touches downstream CE only weakly via attention. This is
why our False run still matched published to <1% (Pearson 0.998).

**Changes**
- `configs/gpu/core_gpu.yaml`: `exclude_special_tokens_from_reconstruction: true` (the canonical run
  setting). `run_core_gpu.py` reads it → the paper-scale run is now exactly faithful.
- `run_core_full_suite.py`: `eval_sae` now reconstructs only non-special positions (keeps the original
  activation at BOS/EOS/PAD), default `exclude_special=True`.
- **Re-ran the full 42-SAE CPU suite with `exclude_special=True`** (reusing the cached orig/zero/sparsity
  passes — only the SAE-reconstruction forward changes). Results:
  - LR shift False→True: **max 0.0010, mean 0.00015** (confirms the ~1e-4 estimate).
  - LR vs published (True): **max\|Δ\| 0.0051, mean 0.0013, Pearson 0.9998** — slightly *better* than the
    False run (max 0.0059), as expected since the published data also used True.
  - All other metrics (explained var, MSE, cosine, L0/L1, density, weight) **unchanged** (exclude_special
    does not touch them).
  - Updated `results/raw/.../full_metrics_4k_pythia160m_vs_neuronpedia.jsonl`,
    `results/processed/.../full_metrics_vs_neuronpedia.json` (provenance now records `exclude_special=True`),
    and regenerated `figures/core_full_metrics_vs_neuronpedia.svg`.
- The `compute_core_full` / oracle keep the function-level default **False** — matching SAEBench's own
  function default (`get_recons_loss(..., exclude_special_tokens_from_reconstruction=False)`); the
  published *run* simply passed the flag, which both `core_gpu.yaml` and the CPU runner now do too.

Net: committed CPU results + figures are now at the faithful `exclude_special=True` setting (numbers within
sampling noise of before), and the GPU config matches the published run exactly — good to go for GPU.
