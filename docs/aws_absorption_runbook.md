# AWS runbook — absorption suite (Pythia-160M now; Gemma later)

Run the full 42-SAE absorption suite under **both** sae-bench versions (0.3.2 = matches published,
0.6.0 = current code), on an AWS GPU box, then pull results and **terminate**. Pythia-160M is tiny and
ungated, so this is a **~$2–4, ~1-hour** job — it's the dress rehearsal for the (later) Gemma run. The
only real cost risk is leaving the instance running; the guardrails below prevent that.

> Code is cloud-agnostic (no AWS-specific code). Any CUDA box (RunPod/Lambda/Colab) works — skip to
> **Setup**. Pythia doesn't actually need a GPU; a `c6i.2xlarge` CPU box is cheaper, but a GPU box
> rehearses the exact Gemma path.

## 0. Prereqs (once)
- AWS account with credits (✓). Check the **g5** vCPU quota (Service Quotas → EC2 → "All G and VT Spot
  Instance Requests" ≥ 4, or the on-demand equivalent); request an increase if it's 0 (can take hours).
- **Set a budget alarm first** (cheap insurance): Billing → **Budgets** → monthly cost budget, e.g. **$20**,
  alert at 80% and 100% to your email. Optionally add a *budget action* that stops EC2 instances at 100%.
- No Hugging Face token needed — Pythia-160M is ungated. (Gemma later **is** gated → `huggingface-cli login`.)

## 1. Launch (spot, auto-priced)
- Region **us-east-1**, instance **g5.xlarge** (1× A10G 24 GB, ~$1.006/hr on-demand, ~$0.35–0.45 spot).
- **Request as Spot** (our runner is resumable, so an interruption just resumes) with **"Stop"**, not
  "Terminate", as the interruption behavior.
- AMI: **Deep Learning OSS Nvidia Driver AMI (Ubuntu 22.04)** — CUDA + PyTorch preinstalled.
- Storage: **50 GB gp3** (Pythia model ~0.3 GB + tiny SAEs + two venvs ~3 GB).
- Key pair for SSH.

## 2. Setup (on the box)
```bash
sudo apt-get update -y && sudo apt-get install -y git python3.11-venv
git clone https://github.com/meowmaryexe/SAEBench-Reliability.git && cd SAEBench-Reliability
git checkout alor/ravel-abs-unlearning

# 0.6.0 (current) venv
python3.11 -m venv ~/venv-060 && ~/venv-060/bin/pip -q install "sae-bench" "transformers>=4.51,<5"
# 0.3.2 (published) venv
python3.11 -m venv ~/venv-032 && ~/venv-032/bin/pip -q install \
  "sae-bench @ git+https://github.com/adamkarvonen/SAEBench.git@141aff72928f7588c1451bed47c401e1d565d471" \
  "sae_lens==5.3.1" "transformers>=4.40,<5"
```
Note: each venv keeps its own upstream cache (`RESULTS_DIR`/`PROBES_DIR` are package-relative), so the two
versions never cross-contaminate even in the same working dir.

## 3. Run (one command, resumable, auto-shutdown)
```bash
VENV_032=~/venv-032/bin/python VENV_060=~/venv-060/bin/python \
  DEVICE=cuda AUTO_SHUTDOWN=1 \
  bash scripts/run_absorption_suite.sh 2>&1 | tee suite.log
```
This runs all 42 SAEs under both versions (resumable), fetches the published numbers, aggregates, prints
the per-arch + Spearman report, and — because `AUTO_SHUTDOWN=1` — **shuts the box down on success** (60 s
grace). Expect ~1.5 GPU-h total. To run **Pythia only** (default) no edit is needed; to add Gemma later,
uncomment the `SUITES=(...)` line in `scripts/run_absorption_suite.sh` (and `huggingface-cli login` first).

## 4. Get results back
Before the box shuts down (or from a re-launched stopped box), either commit from the box:
```bash
git add results/processed/absorption/*_v0.*.json && git commit -m "Pythia absorption suite (both versions)" && git push
```
or copy locally: `scp -r ubuntu@<ip>:~/SAEBench-Reliability/results/processed/absorption ./`.
(Raw per-SAE outputs under `results/raw/` are gitignored; the processed JSONs + report are the deliverable.)

## 5. Teardown (verify!)
- If `AUTO_SHUTDOWN=1` the box is stopped/terminated on success. **Confirm in EC2 → Instances that nothing
  is "running."** A stopped spot box still bills ~$0.40/mo for the 50 GB EBS — **terminate** it once results
  are saved (or keep it stopped for a day to reuse for Gemma).

## Cost & guardrails (why this can't run away)
| lever | setting |
|---|---|
| estimate | 42 SAEs × ~1 min × 2 versions ≈ ~1.5 GPU-h + setup ≈ **~$2 spot / ~$3 on-demand** |
| auto-shutdown | `AUTO_SHUTDOWN=1` in the driver → shuts down on completion (the #1 safeguard) |
| budget alarm | AWS Budgets $20, alert 80/100% (+ optional stop-instances action) |
| spot | ~60% cheaper; resumable runner tolerates interruptions |
| right-size | Pythia needs no GPU — `c6i.2xlarge` CPU is cheaper if you're not rehearsing Gemma |
| EBS | 50 GB gp3; terminate when done so it stops billing |

## Verify it worked
`scripts/absorption_suite_report.py` should show, for `absorption_fraction`: **0.3.2 within ~0.05 of
published** across archs (reproduced) and **0.6.0 well below** (the redefinition), plus Spearman ρ for the
ranking-stability question. `full_absorption` should reproduce under both. Then delete the instance.

## Later: Gemma
Same box, bigger model. Changes: `huggingface-cli login` (accept the Gemma license); uncomment the Gemma
suites in `run_absorption_suite.sh`; `DEVICE=cuda LLM_DTYPE=bfloat16`. Cost ~$4–24/width/version (see
`configs/gpu/absorption_gpu.yaml`). Everything else is identical.
