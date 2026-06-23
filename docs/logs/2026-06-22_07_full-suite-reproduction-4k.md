# Log 2026-06-22 #07 — Full Core / Loss Recovered reproduction (4k suite, all 7 architectures)

**Outcome: PASS.** All 42 SAEs of the 4k Pythia-160M suite (7 architectures × 6 sparsities) reproduce
their released Loss Recovered and L0 within the pre-registered tolerances. This completes the Core /
Loss Recovered reproduction on the anchor model at the CPU-feasible width.

## Scope

- **Model:** Pythia-160M-deduped, layer 8 (resid_post).
- **SAEs:** `adamkarvonen/saebench_pythia-160m-deduped_width-2pow12_date-0108` — all 7 architectures
  (Standard, TopK, BatchTopK, JumpRelu, GatedSAE, MatryoshkaBatchTopK, PAnneal) × 6 sparsities (trainer
  0–5) = **42 SAEs**.
- **Config:** exact per-document path (ctx 1024, the Pile), 12 documents, matching the bundled
  `eval_results.json` semantics. Each SAE's Loss Recovered + L0 compared directly to its released values.
- 16k / 65k widths and Gemma-2-2B are GPU-deferred (same code path; see below).

## Architecture loaders (all verified against released values)

Confirmed each architecture's `ae.pt` layout and forward (`src/saebench_audit/sae_models.py`):
Standard/PAnneal = `AutoEncoder` (ReLU); Gated = gating+magnitude; JumpReLU = per-feature threshold;
TopK = top-k; BatchTopK = learned-threshold (jump-style at inference); **Matryoshka = distinct `W_enc`/
`W_dec` + threshold layout** (a new loader was added — it does NOT share BatchTopK's Linear layout).
TopK-family L0 reproduces the exact k (20/40/80/160/320/640), a strong correctness check.

## Efficiency: shared baseline

H_orig (model CE) and H0 (zero-ablation CE) are SAE-independent, so `scripts/run_suite.py` computes them
**once** for a shared document pool (caching the layer activations for L0 reuse), then evaluates each SAE
with a single forward pass per batch — ~3× fewer forwards than the naive approach. Resumable (one line
per SAE), auto-downloads + deletes each `ae.pt` to bound disk.

## Results (per architecture)

| arch | # | L0 range | Loss Recovered range | max \|ΔLR\| vs released |
|---|---|---|---|---|
| Standard   | 6 | 43–454 | 0.9105–0.9876 | 0.0010 |
| TopK       | 6 | 20–640 | 0.9375–0.9991 | 0.0038 |
| BatchTopK  | 6 | 20–630 | 0.9439–0.9995 | 0.0059 |
| JumpRelu   | 6 | 20–606 | 0.9362–0.9991 | 0.0038 |
| GatedSAE   | 6 | 29–456 | 0.9530–0.9987 | 0.0019 |
| Matryoshka | 6 | 20–630 | 0.9342–0.9993 | 0.0042 |
| PAnneal    | 6 | 30–442 | 0.9368–0.9962 | 0.0036 |

**Overall (42 SAEs):**
- max \|ΔLoss Recovered\| = **0.0059**, mean = **0.0013** → all within the pre-registered **0.01** band ✅
- L0 relative error: max **3.8%**, mean **1.8%** → all within **5%** ✅
- mine vs released Loss Recovered: **Pearson r = 0.9984, Spearman ρ = 0.9966** → the sparsity–fidelity
  frontier and architecture ranking reproduce.

Residuals are a small, consistent positive bias (~+0.001–0.006) from the 12-document sample + data-order
offset (the authors averaged 1000 shuffled sequences; we take the first 12 docs of
`monology/pile-uncopyrighted`). It cancels out of the ratio at scale — the Standard 4k trainer_0 point at
128 docs (log #04) matches to 0.0006.

## Pre-registered ranking criterion (met)

§5 of the pre-registration required Spearman ρ ≥ 0.9 of architecture Loss Recovered vs released, with no
inversion of pairs whose released gap > 0.01. Achieved ρ = 0.997; the frontier shape (Loss Recovered rising
with L0; clean per-architecture curves) matches the released suite.

## Artifacts

- Raw: `results/raw/core_loss_recovered/suite_4k_pythia160m.jsonl` (42 lines).
- Processed: `results/processed/core_loss_recovered/suite_4k_pythia160m.json` (per-SAE + summary).
- Figures: `figures/core_lr_frontier_4k.svg` (sparsity–fidelity frontier, 7 arch),
  `figures/core_lr_reproduction_4k.svg` (mine vs released scatter, r=0.998).
- Runner: `scripts/run_suite.py`; loaders: `src/saebench_audit/sae_models.py`.

## Remaining for a complete Core reproduction (GPU)

1. **Widths 16k & 65k** (Pythia-160M) — same `run_suite.py`, larger SAEs; 65k needs more RAM than this CPU
   box. The headline paper figure (Fig. 2) is 65k Gemma.
2. **Gemma-2-2B, layer 12** — `--repo` the Gemma SAE suites; needs GPU (2B model). Same code path.
3. **Paper-config cross-check** — run a subset under the `paper` variant (OpenWebText ctx128, packed) and
   compare to Neuronpedia Core values, to validate the packed path against published numbers too.
