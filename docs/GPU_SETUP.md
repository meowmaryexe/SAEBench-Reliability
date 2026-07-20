# GPU Setup & Run Guide — Core/Loss Recovered + AutoInterp

Step-by-step instructions to run both reproduced metrics at **paper scale** on a GPU, for the two anchor
models (**Pythia-160M** layer 8, **Gemma-2-2B** layer 12) across all three SAE widths (4k / 16k / 65k).

Everything was developed and validated on CPU; this guide is for the full-size runs that need a GPU
(a 24 GB card — A10G/L4 — is enough for everything here; see §9b for instance choice).

---

## TL;DR — if you just want to run it

On a GPU box (see §9b to stand one up on AWS), this is the whole thing:

```bash
git clone https://github.com/meowmaryexe/SAEBench-Reliability.git
cd SAEBench-Reliability

pip install torch --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
pip install -r requirements.txt

huggingface-cli login                      # needed for Gemma (gated) — accept the license first
echo "sk-...yourkey..." > openai_api_key.txt   # needed for AutoInterp (gitignored)

bash scripts/smoke_test.sh    # ~3 min: checks GPU, HF auth, SAE download, OpenAI key, runs the tests
bash scripts/run_all_gpu.sh   # the full sweep (both metrics, all 6 suites) -> results/ + figures/
```

**`run_all_gpu.sh` is safe to re-run** — every runner checkpoints per-SAE and per-latent, so if SSH
drops, the box reboots, or a Spot instance is reclaimed, just run it again and it resumes. Run it under
`tmux` so an SSH drop doesn't kill it.

Useful variants:
```bash
METRICS=core bash scripts/run_all_gpu.sh                 # Core only (no OpenAI cost)
SUITES="gemma-2-2b_65k" bash scripts/run_all_gpu.sh      # just the paper headline
```

If `smoke_test.sh` reports ❌ on anything, fix that first (§10 troubleshooting) — it's much cheaper than
discovering it 40 minutes into a GPU run. The rest of this document is the detail behind those commands.

---

## 0. The suite matrix

6 SAE suites = 2 models × 3 widths. Each suite = 7 architectures × 6 sparsities = **42 SAEs**.
Repos/folder layouts live in `configs/registry.yaml` (already verified against HuggingFace).

| suite name | model · layer | width | HF repo |
|---|---|---|---|
| `pythia-160m_4k`  | Pythia-160M · L8  | 4k  | adamkarvonen/saebench_pythia-160m-deduped_width-2pow12_date-0108 |
| `pythia-160m_16k` | Pythia-160M · L8  | 16k | adamkarvonen/saebench_pythia-160m-deduped_width-2pow14_date-0108 |
| `pythia-160m_65k` | Pythia-160M · L8  | 65k | adamkarvonen/saebench_pythia-160m-deduped_width-2pow16_date-0108 |
| `gemma-2-2b_4k`   | Gemma-2-2B · L12  | 4k  | adamkarvonen/saebench_gemma-2-2b_width-2pow12_date-0108 |
| `gemma-2-2b_16k`  | Gemma-2-2B · L12  | 16k | canrager/saebench_gemma-2-2b_width-2pow14_date-0107 |
| `gemma-2-2b_65k`  | Gemma-2-2B · L12  | 65k | canrager/saebench_gemma-2-2b_width-2pow16_date-0107 |

---

## 1. Prerequisites

- A CUDA GPU + recent NVIDIA driver.
- **HuggingFace account with Gemma access**: `google/gemma-2-2b` is gated — accept the license at
  https://huggingface.co/google/gemma-2-2b, then `huggingface-cli login` (or set `HF_TOKEN`). Pythia is
  ungated.
- **OpenAI API key** (AutoInterp only): the judge is `gpt-4o-mini`.

---

## 2. Environment

```bash
git clone https://github.com/meowmaryexe/SAEBench-Reliability.git
cd SAEBench-Reliability

python -m venv .venv && source .venv/bin/activate

# runtime deps for the RUNS (the runners use HuggingFace transformers, not transformer_lens)
pip install "torch" --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
pip install -r requirements.txt

# (optional) to also run the methodology ORACLE tests on the GPU box, add:
pip install --no-deps transformer_lens==2.15.4 jaxtyping fancy_einsum typeguard beartype rich better_abc sentencepiece
#   and provide a stub `wandb` module on PYTHONPATH (transformer_lens imports it at load; inference never uses it)

huggingface-cli login          # for Gemma
```

Sanity check (no GPU work):

```bash
python tests/test_core_units.py
python tests/test_autointerp_units.py
python tests/test_autointerp_prompts.py
python tests/test_autointerp_oracle.py     # deterministic AutoInterp oracle (CPU)
# Full transformer_lens oracles (need transformer_lens installed):
python tests/test_core_oracle.py --model_dir <pythia-dir> --sae_dir <sae-dir>
python tests/test_core_full_oracle.py
```

---

## 3. Secrets (AutoInterp)

Put the OpenAI key in a gitignored file at the repo root (the runner's default `--keyfile`):

```bash
printf 'sk-...yourkey...' > openai_api_key.txt    # .gitignore already blocks *api_key*, *.key, .openai_key
```

Never commit it. The key never appears in any result/log file.

---

## 4. Configs (already set; edit only if needed)

- `configs/gpu/core_gpu.yaml` — Core (Table 4): OpenWebText, ctx 128, 3200 recon / 32000 sparsity,
  batch 16, `runtime.device: cuda`, `dtype: float32`. **For 65k Gemma on a smaller GPU, set
  `dtype: bfloat16`** (perturbs only low-order digits).
- `configs/gpu/autointerp_gpu.yaml` — AutoInterp (Table 5): the Pile, 2M tokens, 1000 latents, the
  example counts, `gpt-4o-mini` judge, `judge_workers: 10`.
- `configs/registry.yaml` — models + suite repos + folder layouts (shared by both metrics).

---

## 5. Run CORE / Loss Recovered

One command per suite; writes a per-SAE results JSON (each SAE compared to its bundled `eval_results.json`).
The runner builds the token pools once, then loops all 42 SAEs (downloads each `ae.pt`, evaluates, deletes).

```bash
mkdir -p results/processed/core_loss_recovered

for SUITE in pythia-160m_4k pythia-160m_16k pythia-160m_65k \
             gemma-2-2b_4k gemma-2-2b_16k gemma-2-2b_65k; do
  python scripts/run_core_gpu.py \
    --config configs/gpu/core_gpu.yaml \
    --suite $SUITE \
    --out results/processed/core_loss_recovered/${SUITE}_saebench_core.json
done
```

Notes:
- Device/dtype come from `core_gpu.yaml` (`cuda` / `float32`). To override dtype for Gemma-65k, edit the
  config (`runtime.dtype: bfloat16`).
- Run a subset with `--archs standard topk` and/or `--trainers 0 1 2`.
- This produces **Loss Recovered, L0, explained variance, MSE, cosine, L2 ratio, relative reconstruction
  bias, L1, feature density, max-cosine-sim** (the full Core set; methodology proven identical to
  `core/main.py` — `tests/test_core_oracle.py`).

---

## 6. Run AUTOINTERP

One command per suite. Builds the SAE-independent residual cache **once per (model, n_tokens)** and reuses
it across all 42 SAEs; runs the gpt-4o-mini judge concurrently.

```bash
mkdir -p results/processed/autointerp

for SUITE in pythia-160m_4k pythia-160m_16k pythia-160m_65k \
             gemma-2-2b_4k gemma-2-2b_16k gemma-2-2b_65k; do
  python scripts/run_autointerp_gpu.py \
    --config configs/gpu/autointerp_gpu.yaml \
    --registry configs/registry.yaml \
    --suite $SUITE \
    --device cuda \
    --judge_workers 10 \
    --keyfile openai_api_key.txt \
    --out results/processed/autointerp/${SUITE}_autointerp.json
done
```

Notes:
- The residual cache defaults to `_ai_sae_tmp/resid_<suite>_<n_tokens>/`. Reuse it across reruns; delete to
  reclaim disk (2M tokens ≈ a few GB of bf16 activations).
- Increase `--judge_workers` if your OpenAI rate limit allows (more concurrency = faster).
- gpt-4o-mini cost ≈ a few cents per SAE (~1000 latents × 2 calls); the full 252-SAE sweep is ~tens of $.

---

## 7. Reference values (compare to the published SAEBench / Neuronpedia numbers)

The published results live in the HF dataset `adamkarvonen/sae_bench_results_0125`. Mirror the folders you
want, then point the comparisons at them.

```bash
huggingface-cli download adamkarvonen/sae_bench_results_0125 --repo-type dataset \
  --local-dir sae_bench_results_0125

# Core: the runner already records each SAE's bundled eval_results.json. For the published-Core diff,
#   use the per-SAE files under sae_bench_results_0125/core/<suite>/ .
# AutoInterp: set AUTOINTERP_REF_DIR so the runner fills the Δ-vs-published column:
export AUTOINTERP_REF_DIR=$PWD/sae_bench_results_0125/autointerp/saebench_pythia-160m-deduped_width-2pow12_date-0108
#   (point this at the matching <suite> folder before each AutoInterp run)
```

---

## 8. Figures

```bash
python scripts/make_figures.py
# -> figures/core_lr_frontier_4k.svg, core_full_metrics_vs_neuronpedia.svg,
#    autointerp_convergence.svg, autointerp_score_histogram.svg, autointerp_explanations_showcase.svg
```

The frontier / vs-width figures become most meaningful once 16k & 65k are in (they're the paper's
scaling claims).

---

## 9. Compute & cost ($, planning estimates — treat ±50% as the real band)

**Scope priced here = the full reproduction:** both metrics × both anchor models × 4k/16k/65k =
6 suites × 42 SAEs = **252 SAEs** (AutoInterp at 1000 latents/SAE).

### Bottom line

**≈ $75–115 total** (≈ **$40 GPU** + ≈ **$55 OpenAI**), or **≈ $65–90 using OpenAI's Batch API**.

### GPU compute (~$25–60)

Core dominates GPU time; AutoInterp only needs one activation pass per model (reused across widths).
Per-SAE minutes are the paper's Appendix-A timings (RTX-3090 class):

| | per-SAE | × 42 SAEs |
|---|---|---|
| Pythia 4k / 16k / 65k | 1 / 2 / 4 min | 0.7 / 1.4 / 2.8 h |
| Gemma 4k / 16k / 65k | 5 / 9 / 16 min | 3.5 / 6.3 / 11.2 h |

→ Core ≈ **26 GPU-h on a 3090**, ~**15–20 GPU-h on an A100** (faster card). AutoInterp adds ~**2–4 GPU-h**
(the residual-cache passes). Total ≈ **18–30 A100-hours**.
At current **A100-80GB ~$1.20–2.00/hr** (RunPod ~$1.19–1.39, Lambda ~$1.99) → **~$25–60** (point ~$40).

### OpenAI API — AutoInterp only (~$50–55)

`gpt-4o-mini` = **$0.15 / 1M input, $0.60 / 1M output**. AutoInterp makes **2 calls/latent**
(write explanation, then predict), **1000 latents/SAE**:

- input ≈ 1,240 tok/latent (system prompts + the 15/14 example windows) → 252,000 × 1,240 ≈ **313M input** → ~$47
- output ≈ 45 tok/latent → ~**11M output** → ~$7
- **≈ $54 total** → **~$27 with the Batch API** (50% off, async — fine for this).

### Cheaper scopes

| scope | GPU | OpenAI | total |
|---|---|---|---|
| Headline only (65k Gemma, 42 SAEs, both metrics) | ~$8–12 | ~$9 | **~$20** |
| Core only (all 252 SAEs) | ~$25–45 | $0 | **~$25–45** |
| AutoInterp only (all 252 SAEs) | ~$3–6 | ~$27–54 | **~$30–60** |

### Caveats
- ±50% real band (card choice, spot vs on-demand, retries, OpenAI rate-limit stalls).
- The CPU validation runs already done cost **~$0.05** on the key.
- AutoInterp is judge-stochastic — budget a 2nd pass to bound variance (≈ doubles the OpenAI figure).

*Prices verified 2026-06 — [OpenAI pricing](https://openai.com/api/pricing/),
[RunPod](https://www.runpod.io/pricing), [Lambda](https://lambda.ai/pricing),
[Spheron GPU pricing 2026](https://www.spheron.network/blog/gpu-cloud-pricing-comparison-2026/). Re-check before budgeting.*

---

## 9b. Standing up the GPU on AWS (with Activate credits)

### Step 0 — credits
[AWS Activate](https://aws.amazon.com/startups/credits/): **Founders tier = $1,000**, self-serve (no
accelerator needed); Portfolio tier up to $100K via an accelerator/VC. Eligibility: pre-Series B, founded
within 10 years, a working company website, AWS account on a paid support plan, new to Activate credits.
Approval **7–10 business days**; credits expire **12–24 months** after activation. Apply with a Builder ID
at the link above, then confirm the credits show under *Billing → Credits*.

> If this is a purely academic project rather than a startup, Activate may not fit — look at **AWS Cloud
> Credit for Research** or your university's AWS allocation instead.

### Step 1 — request GPU quota FIRST (the actual blocker)
New accounts default to **0 vCPUs** for GPU families, so a launch will fail with
"instance limit exceeded" even with credits.
*Console → Service Quotas → Amazon EC2 → **"Running On-Demand G and VT instances"*** → request **8–16
vCPUs** (enough for one `g5.2xlarge`/`g6.2xlarge`). Do this in the region you'll actually use
(`us-east-1` or `us-west-2` have the best G5/G6 availability). Approval ranges from minutes to a couple of
days — start it before you need it. (Request the *G and VT* quota, not P — you don't need P instances.)

### Step 2 — pick the instance (don't over-buy)
| instance | GPU | ~$/hr | verdict |
|---|---|---|---|
| **g5.2xlarge** | A10G 24 GB, 8 vCPU | ~$1.21 | **recommended** — fits everything, cheap |
| g6.2xlarge | L4 24 GB, 8 vCPU | ~$1.0–1.3 | newer, similar; fine |
| g6e.xlarge | L40S 48 GB | ~$2 | faster if you want shorter wall-clock |
| p4d / p5 | A100 / H100 | $32+ | **overkill** — you'd waste 7 of 8 GPUs, harder quota |

Sizing: Gemma-2-2B bf16 ≈ 5 GB + 65k SAE ≈ 1 GB + ctx-128 activations → comfortably under 24 GB.
The 8 vCPUs matter for tokenization / dataset streaming.

### Step 3 — launch
- **AMI:** *Deep Learning OSS Nvidia Driver AMI GPU PyTorch (Ubuntu 22.04)* — NVIDIA driver, CUDA and
  PyTorch preinstalled (supports G5/G6). Saves an hour of driver pain.
- **Storage:** **200 GB gp3** (HF model cache + SAE downloads + the 2M-token residual cache ≈ a few GB).
- Key pair + security group allowing SSH from your IP.

```bash
ssh -i key.pem ubuntu@<public-ip>
nvidia-smi                      # confirm the GPU is visible
```

### Step 4 — set up and run
Follow sections 2–8 above (env, `huggingface-cli login` for Gemma, `openai_api_key.txt`, then the Core and
AutoInterp loops).

### Step 5 — cost control
- **Timing on this hardware:** A10G/L4 is ~2–3× slower than the A100 baseline in §9, so budget
  **~40–70 GPU-h → ~$50–85 on-demand** for the full 252-SAE sweep. Comfortably inside $1,000 of credits.
- **Use Spot for ~70% off (~$15–25).** Safe here: every runner in this repo is **checkpoint-resumable**
  (per-SAE and per-latent), so an interruption just means re-invoking the same command.
- **`stop` the instance whenever you're not running** — GPU instances bill per second while *running*.
  Stopped instances still bill for EBS (~$16/mo for 200 GB), so delete the volume when done.
- Set an **AWS Budget alert** (e.g. $100) so a forgotten instance can't silently burn the credits.

---

## 10. Troubleshooting

- **"Instance limit exceeded" / "insufficient quota" on launch** → your G-family vCPU quota is 0. See §9b
  Step 1 (Service Quotas → EC2 → *Running On-Demand G and VT instances*). Credits do **not** raise quotas.
- **Gemma 401/403 on download** → accept the license + `huggingface-cli login`.
- **OOM on 65k Gemma** → set `core_gpu.yaml runtime.dtype: bfloat16`, and/or lower `batch_size_prompts`.
- **AutoInterp `openai_api_key.txt` not found** → create it at the repo root or pass `--keyfile`.
- **canrager Gemma folder mismatch** → the 16k/65k Gemma suites use a different folder convention
  (snake_case); it's already encoded in `registry.yaml`. If a `trainer_i` is missing, the runner skips it.
- **Resuming** → both runners skip already-cached SAEs/latents; safe to re-invoke.

---

## What "reproduced" means here

- **Core**: methodology proven *identical* to SAEBench `core/main.py` (oracle, ~1e-7), full metric set
  reproduced on Pythia-160M 4k to <1% vs published (`docs/logs/2026-06-22_08`, `_10`). This guide scales it
  to the remaining widths + Gemma.
- **AutoInterp**: deterministic pipeline proven *identical* to SAEBench (oracle + verbatim prompts), real
  gpt-4o-mini judge; score converges to published with token budget (`docs/logs/2026-06-22_11`). This guide
  runs it at the full 2M-token / 1000-latent scale.
