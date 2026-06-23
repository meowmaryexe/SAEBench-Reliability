# Log 2026-06-22 #08 — Methodology audit vs adamkarvonen/SAEBench + oracle equivalence test

**Goal (per Ari):** this is a *reproduction* — there must be **no methodology difference** between
adamkarvonen/SAEBench and our implementation. Triple-check, write tests.

**Outcome:** our Core / Loss Recovered methodology is now proven **identical** to SAEBench's
`sae_bench/evals/core/main.py`, by an oracle test that runs SAEBench's **verbatim** code on the real
`transformer_lens` model and matches our implementation to **Δ ≤ 2.4e-5 Loss Recovered** across all 7
architectures.

## Two SAEBench eval code paths (important clarification)

There are two evaluations in the SAEBench ecosystem, and they differ in methodology:

1. **`dictionary_learning/evaluation.py::loss_recovered`** → produced the bundled `eval_results.json`
   (per-document, ctx 1024, the Pile). Our 42-SAE suite (log #07) reproduced these to ≤0.6% — a faithful
   match to *that* path.
2. **`sae_bench/evals/core/main.py`** → produced the **paper Figure 2 / Neuronpedia** numbers. This is the
   canonical "SAEBench Core" eval.

The two are only comparable in Loss Recovered, not absolute CE (tokenization differs). To satisfy "no
methodology difference with the SAEBench repo," the authoritative target is **#2**.

## Exact `core/main.py` methodology (audited line-by-line)

| Aspect | SAEBench core/main.py | Our implementation |
|---|---|---|
| Model load | `HookedTransformer.from_pretrained_no_processing` → raw activations = HF residual stream (main.py:1033) | HF model, raw resid_post — **identical activations** |
| Hook site | `blocks.L.hook_resid_post` | output of decoder block L |
| SAE forward | `sae.decode(sae.encode(x))` | same |
| Zero ablation | `torch.zeros_like(resid_post)`, all positions | same |
| Reconstruction sites | all positions (default `exclude_special_tokens_from_reconstruction=False`) | same (configurable) |
| Per-token CE | transformer_lens `loss_per_token` (log_softmax+gather) | `per_token_ce` (verified == it) |
| CE reduction | exclude `ignore_tokens={bos,eos,pad}` via `mask[:, :-1]`, concat batches, `.mean()` (main.py:1101) | same |
| Loss Recovered | `(ce_abl − ce_sae)/(ce_abl − ce_orig)` | same |
| L0 | `(acts != 0).sum(-1)`, exclude special tokens, `.mean()` | same |

**Discrepancies found in our earlier suite (now fixed in the exact path):** our `bundle_exact`/suite path
excluded only PAD from the CE mean and L0; `core/main.py` excludes **{bos, eos, pad}**. Implemented in
`saebench_audit.metrics.core_loss_recovered.saebench_core_eval`.

## Oracle equivalence test (the proof)

`tests/test_core_oracle.py` + `tests/saebench_verbatim.py`:
- `saebench_verbatim.py` is a **verbatim transcription** of SAEBench's `get_recons_loss`,
  `get_downstream_reconstruction_metrics` reduction, and the L0 path (each block annotated with its
  source lines).
- The test loads Pythia-160M as a real `transformer_lens.HookedTransformer`
  (`from_pretrained_no_processing`, `hf_model=` the local HF model → loads in ~3s), runs the verbatim
  SAEBench code, and compares to our `saebench_core_eval` (HF) on the **same** OpenWebText token batches
  and the **same** released SAE.

**Results (all 7 architectures, 4k Pythia-160M L8, OWT ctx128):**

| arch | oracle LR | mine LR | Δ Loss Recovered |
|---|---|---|---|
| Standard   | 0.982584 | 0.982584 | 4.1e-07 |
| TopK       | 0.972111 | 0.972117 | 6.4e-06 |
| BatchTopK  | 0.973764 | 0.973788 | 2.4e-05 |
| JumpRelu   | 0.974104 | 0.974100 | 4.4e-06 |
| GatedSAE   | 0.988475 | 0.988474 | 1.3e-06 |
| Matryoshka | 0.968074 | 0.968058 | 1.6e-05 |
| PAnneal    | 0.978901 | 0.978899 | 1.8e-06 |

CE components match to ~1e-5; L0 to <0.004% (relative). Framework note: max\|HF_logit − TL_logit\| ≈ 0.14
(transformer_lens vs HF float-path differences) but the **CE losses match to ~1e-5** because the
activations are effectively identical under `no_processing` — confirming the SAE sees the same input.

## Unit tests

`tests/test_core_units.py` (8 tests, all pass) pin each primitive: loss-recovered formula; per-token CE ==
`cross_entropy(reduction='none')`; special-token masking of CE and L0; L0 = `(acts!=0).sum(-1)`;
zero-ablation; Standard SAE forward == `ReLU(W_enc(x−b_dec))` then `decode`; TopK exactly-k active; all 7
arch loaders load and run.

## Environment for the tests

`transformer_lens==2.15.4` installed `--no-deps` (avoids pulling a conflicting torch), plus
`einops jaxtyping fancy_einsum typeguard beartype rich better_abc sentencepiece`; `wandb` stubbed (TL
imports it at load for training only). The real model class is used — not a reimplementation.

## Standing on "no methodology difference"

- **core/main.py methodology**: proven identical (oracle, Δ ≤ 2.4e-5, 7/7 architectures).
- **dictionary_learning methodology** (eval_results.json): reproduced to ≤0.6% (log #07).
Both SAEBench eval paths are therefore reproduced; the canonical Core path is methodologically exact.

## Follow-up

Re-run the full 4k frontier under `saebench_core_eval` (OWT ctx128, {bos,eos,pad}-excluded) to publish
the suite numbers in the canonical methodology and cross-check against Neuronpedia Core values — code is
ready (`saebench_core_eval`); only a scaled run (ideally GPU) remains.
