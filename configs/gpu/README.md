# GPU configs — paper-scale reproduction (Core + AutoInterp)

Everything needed to run the metrics at **paper scale** on a GPU. Nothing here is meant to run on the CPU
sandbox. The same `../registry.yaml` (models + SAE suites) is shared by both metrics.

## Files
- **`core_gpu.yaml`** — Core eval config: dataset/ctx/counts (Table 4), special-token handling, device,
  dtype. Methodology = `saebench_core` (proven identical to `core/main.py`; `tests/test_core_oracle.py`).
- **`autointerp_gpu.yaml`** — AutoInterp eval config: Table 5 (2M tokens, 1000 latents, example counts,
  gpt-4o-mini judge). Deterministic pipeline proven identical to SAEBench (`tests/test_autointerp_oracle.py`).
- **`jobs.yaml`** — the full Core job matrix (6 suites × 42 SAEs = 252), grouped into compute tiers.
- **`../registry.yaml`** — base models + released SAE suite repos + per-suite folder layouts (two naming
  conventions: adamkarvonen for Pythia + Gemma-4k, canrager for Gemma 16k/65k). Shared by Core + AutoInterp.

## Runners
- Core:       `scripts/run_core_gpu.py --config configs/gpu/core_gpu.yaml --suite <name> --out ...`
- AutoInterp: `scripts/run_autointerp_gpu.py --config configs/gpu/autointerp_gpu.yaml --suite <name>
  --device cuda --judge_workers 10 --out ...`  (needs `openai_api_key.txt` at repo root, gitignored;
  set `AUTOINTERP_REF_DIR` to a local mirror of `sae_bench_results_0125/autointerp/...` for the Δ column).
  Both iterate a suite (7 arch × 6 sparsity), resolve repos/folders from the registry, build the
  SAE-independent activation cache once, and write per-SAE results vs published.

## Suites (resolved in `registry.yaml`)
| suite | model · layer | width | HF repo |
|---|---|---|---|
| pythia-160m_4k  | Pythia-160M · L8  | 4k  | adamkarvonen/…2pow12_date-0108 |
| pythia-160m_16k | Pythia-160M · L8  | 16k | adamkarvonen/…2pow14_date-0108 |
| pythia-160m_65k | Pythia-160M · L8  | 65k | adamkarvonen/…2pow16_date-0108 |
| gemma-2-2b_4k   | Gemma-2-2B · L12  | 4k  | adamkarvonen/…2pow12_date-0108 |
| gemma-2-2b_16k  | Gemma-2-2B · L12  | 16k | canrager/…2pow14_date-0107 |
| gemma-2-2b_65k  | Gemma-2-2B · L12  | 65k | canrager/…2pow16_date-0107 |

## Launch

```bash
# headline: 65k Gemma-2-2B (paper Figure 2)
python scripts/run_core_gpu.py --config configs/gpu/core_gpu.yaml --suite gemma-2-2b_65k \
  --out results/processed/core_loss_recovered/gemma-2-2b_65k_saebench_core.json

# one suite at a time; loop over jobs.yaml -> tiers.full.suites for everything
for s in pythia-160m_4k pythia-160m_16k pythia-160m_65k gemma-2-2b_4k gemma-2-2b_16k gemma-2-2b_65k; do
  python scripts/run_core_gpu.py --config configs/gpu/core_gpu.yaml --suite $s \
    --out results/processed/core_loss_recovered/${s}_saebench_core.json
done

python scripts/make_figures.py            # frontier + reproduction figures
```

## Environment (GPU)
```bash
pip install torch transformers datasets safetensors huggingface_hub pyyaml
# gemma-2-2b is gated: huggingface-cli login (accept the license)
```
Set `runtime.dtype: bfloat16` in `core_gpu.yaml` for Gemma on <40GB GPUs (slightly perturbs low-order
digits vs the float32 default).

## Compute (planning, ±50%; Appendix A anchor: 16k Gemma ≈ 9 min/SAE)
- Tier 0 (65k Gemma, 42 SAEs): **~11 GPU-h**
- Full (252 SAEs, 1×A100): **~40 GPU-h** (~1 day; parallelize across GPUs to compress)

## After running
Aggregate (`scripts/aggregate_results.py`) then compare to the **published reference values**
`adamkarvonen/sae_bench_results_0125` (HF dataset; `registry.yaml → reference_results`) against the
pre-registered tolerances (`docs/preregistration.md`: Loss Recovered ±0.01, L0 ±5%).
