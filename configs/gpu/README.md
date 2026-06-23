# GPU configs — full Core / Loss Recovered reproduction

Everything needed to run the Core metric at **paper scale** on a GPU, using the methodology proven
identical to SAEBench `core/main.py` (`saebench_core`; see `docs/logs/2026-06-22_08` and
`tests/test_core_oracle.py`). Nothing here is meant to run on the CPU sandbox.

## Files
- **`core_gpu.yaml`** — the canonical eval config: dataset/ctx/counts (Table 4), special-token handling,
  device, dtype. Methodology = `saebench_core`.
- **`jobs.yaml`** — the full job matrix (6 suites × 42 SAEs = 252), grouped into compute tiers.
- **`../registry.yaml`** — base models + released SAE suite repos + per-suite folder layouts (two naming
  conventions: adamkarvonen for Pythia + Gemma-4k, canrager for Gemma 16k/65k).

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
