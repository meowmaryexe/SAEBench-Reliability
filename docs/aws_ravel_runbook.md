# AWS runbook — SAEBench RAVEL (disentanglement) on base Gemma-2-2b

Reproduce the RAVEL suite (42 SAEs/width, base `gemma-2-2b`, layer 12) on an AWS GPU box. This mirrors
`docs/aws_absorption_runbook.md` for the generic AWS bits (launch, budget alarm, auto-shutdown, S3/git
auto-save) — read that for the cloud mechanics. This file covers the **RAVEL-specific** setup, which is
notably **simpler** than unlearning (no gated corpus) but **more expensive** (it's the priciest metric).

## Use an ON-DEMAND box
RAVEL is the suite's most expensive metric (~45 min/SAE per the plan doc's Appendix A timing table, ~41% of
the all-8 per-SAE cost). A single 42-SAE width is a long, uninterrupted run — use an **on-demand** g5.xlarge
(A10G, 24 GB), not spot, to avoid mid-run reclaims (spot thrashed the unlearning run). Cost per width ≈ ~32
GPU-h ≈ **~$33–38** on-demand (@ ~$1.006/hr; treat ±50% as the real band). 65k is OOM-prone on 24 GB —
consider `--llm_batch_size 4` or a g5.2xlarge.

## What RAVEL needs (vs unlearning)
- **GPU required** (gemma-2-2b MDBM forward+backward passes; no CPU path).
- **Gemma license** — `google/gemma-2-2b` (the **BASE** model, not `-it`) is HF-gated.
- **NO gated corpus.** RAVEL's prompts (`adamkarvonen/ravel_prompts`) are a **public** HF dataset that
  auto-downloads to `artifacts/ravel/base/` on first use. Nothing to request or place manually.
- **One-time per-model dataset generation** on the first SAE: the model generates completions over all
  (template × entity) prompts to find the correctly-answered subset, then caches the filtered dataset under
  `artifact_dir` (`artifacts/ravel`), reused across every SAE, resume, and width. Budget ~1 GPU-h for it.

## Launch the on-demand box (copy-paste, safe to re-run)

On your **Mac** (AWS CLI configured):

```bash
REGION=us-east-1

# 1. key pair — reuse if present, else create (this is the fresh key; retire the leaked absorption-key)
if [ ! -f ~/ravel-key.pem ]; then
  aws ec2 create-key-pair --region $REGION --key-name ravel-key \
    --query 'KeyMaterial' --output text > ~/ravel-key.pem && chmod 400 ~/ravel-key.pem
fi

# 2. security group — look up or create; allow SSH from your current IP
SG=$(aws ec2 describe-security-groups --region $REGION --group-names ravel-sg \
      --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null)
if [ -z "$SG" ] || [ "$SG" = "None" ]; then
  VPC=$(aws ec2 describe-vpcs --region $REGION --filters Name=isDefault,Values=true \
        --query 'Vpcs[0].VpcId' --output text)
  SG=$(aws ec2 create-security-group --region $REGION --group-name ravel-sg \
        --description "SSH for RAVEL run" --vpc-id "$VPC" --query 'GroupId' --output text)
fi
MYIP=$(curl -s https://checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress --region $REGION --group-id "$SG" \
  --protocol tcp --port 22 --cidr "${MYIP}/32" 2>/dev/null || echo "(SSH rule already present)"

# 3. current Deep Learning AMI — NOTE the hyphenated path (ubuntu-22.04) + a guard.
#    AWS rotates/renames these; if it 404s, list valid ones with:
#      aws ssm get-parameters-by-path --region $REGION --recursive \
#        --path /aws/service/deeplearning/ami/x86_64/ --query 'Parameters[].Name' --output text | tr '\t' '\n' | grep gpu-pytorch
AMI=$(aws ssm get-parameter --region $REGION \
  --name /aws/service/deeplearning/ami/x86_64/oss-nvidia-driver-gpu-pytorch-2.7-ubuntu-22.04/latest/ami-id \
  --query 'Parameter.Value' --output text)
[ -n "$AMI" ] && [ "$AMI" != "None" ] && echo "AMI=$AMI" || { echo "!! AMI did not resolve — fix the param name before launching"; }

# 4. launch ON-DEMAND g5.xlarge, STOP-on-shutdown (so you can reconnect + results survive), 100 GB disk
ID=$(aws ec2 run-instances --region $REGION \
  --image-id "$AMI" --instance-type g5.xlarge \
  --key-name ravel-key --security-group-ids "$SG" \
  --instance-initiated-shutdown-behavior stop \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=ravel-gpu}]' \
  --count 1 --query 'Instances[0].InstanceId' --output text)

# 5. wait, fetch the (new) public IP, SSH in  (⚠ ssh, never sh — sh prints your key)
aws ec2 wait instance-running --region $REGION --instance-ids "$ID"
IP=$(aws ec2 describe-instances --region $REGION --instance-ids "$ID" \
  --query 'Reservations[].Instances[].PublicIpAddress' --output text)
ssh -i ~/ravel-key.pem ubuntu@"$IP"
```

> On-demand G quota: if `run-instances` errors `VcpuLimitExceeded`, request "Running On-Demand G and VT
> Instances" (code `L-DB2E81BA`) ≥ 4 vCPUs in Service Quotas (can take hours). This is a *different* quota
> from the Spot one the absorption runbook mentions.

## On-box setup

This DLAMI is **Ubuntu 22.04** (system python 3.10) — use the base `python3` venv. Do **not** `apt install
python3.11-venv` (not in 22.04's default repos → `ensurepip is not available`); `python3 -m venv` on 3.10 is
fine, sae-bench supports it.

```bash
sudo shutdown -h +2400 &      # dead-man's switch (~40h; sized above a ~32 GPU-h 4k run)

sudo apt-get update -y && sudo apt-get install -y git python3-venv
git clone https://github.com/meowmaryexe/SAEBench-Reliability.git && cd SAEBench-Reliability
git checkout alor/ravel-abs-unlearning

# ONE venv — RAVEL is single-version (sae-bench 0.6.0, transformers<5)
python3 -m venv ~/venv-060 && ~/venv-060/bin/pip -q install "sae-bench" "transformers>=4.51,<5"

# gemma-2-2b (BASE) is gated → log in + accept the license. RAVEL prompt data is public (no corpus).
~/venv-060/bin/huggingface-cli login          # paste a token, then accept at hf.co/google/gemma-2-2b
~/venv-060/bin/huggingface-cli download google/gemma-2-2b config.json   # verify access
```

Then run from the **repo root** (so the `artifacts/ravel` dataset cache is created there and persists across
SAEs/resumes — it's gitignored under `artifacts/`).

## Run

All from the **repo root**, on the on-demand GPU box.

**1. Single-SAE smoke first** (strongly recommended before the full 42 — it also triggers the one-time
dataset gen, so you can measure real per-SAE wall-clock and catch an OOM before committing 32 GPU-h):
```bash
~/venv-060/bin/python scripts/run_ravel.py --suite gemma-2-2b_4k --device cuda \
  --sae_location Standard_gemma-2-2b__0108/resid_post_layer_12/trainer_0 \
  --workdir results/raw/ravel/gemma-2-2b_4k
```

**2. Full 42-SAE suite** — in tmux so it survives an SSH drop:
```bash
tmux new -s ravel
VENV=~/venv-060/bin/python \
DEVICE=cuda LLM_DTYPE=bfloat16 LLM_BATCH=8 AUTO_SHUTDOWN=1 GIT_PUSH=1 \
bash scripts/run_ravel_suite.sh 2>&1 | tee suite.log
# detach: Ctrl-b then d    |    reconnect: ssh back in (re-fetch IP), then `tmux attach -t ravel`
```

The driver is resumable (`until … grep ALL_SAES_DONE`), fetches published values, aggregates, writes a
report + durable run record under `docs/run_records/ravel/`, and (with `AUTO_SHUTDOWN=1`) stops the box on
success. Set `S3_DEST=s3://…` and/or `GIT_PUSH=1` to auto-save results off the box (recommended). To add
16k/65k, uncomment them in the `SUITES=(...)` line — no code change (they're just registry suite keys); for
65k drop `LLM_BATCH=4` if it OOMs.

## Published comparison — confirmed
`fetch_published_ravel.py` derives the results prefix `ravel/<sae_repo_basename>/` and reads each score
under `eval_result_metrics.ravel.{disentanglement,cause,isolation}_score`. **Confirmed present** (2026-07-07)
in `adamkarvonen/sae_bench_results_0125`: `ravel/saebench_gemma-2-2b_width-2pow12_date-0108/` has 42 files
(e.g. BatchTopK t0 disentanglement = 0.4795). **4k** needs no `--results_prefix`; **16k/65k** use `date-0108`
naming while the SAE weights are `canrager/…date-0107`, so pass `--results_prefix ravel/saebench_gemma-2-2b_
width-2pow{14,16}_date-0108` for those (the driver logs a warning if the plain fetch misses).

## After completion
- Grab the run record from `docs/run_records/ravel/<suite>_<TS>/` (scp or S3/git).
- Keep `artifacts/ravel` if you plan to run 16k/65k next — the filtered dataset cache is model-keyed and
  reused across widths.
- Terminate (or stop to reuse for the next width).

## Cost + guardrails
On-demand g5.xlarge, `--instance-initiated-shutdown-behavior stop`, AWS Budgets alarm, `AUTO_SHUTDOWN=1`,
dead-man's-switch `sudo shutdown`. One width ≈ ~32 GPU-h (~$33–38 on-demand); the one-time dataset gen (~1
GPU-h) is amortized across the 42 SAEs. All three widths ≈ ~95 GPU-h (~$100–130).

## Stage-2 audit (deferred — not part of this Stage-1 reproduction)
RAVEL's undocumented choices need source edits + reruns (no compute-free recompute): the dropped
reconstruction-error term (`mdbm.py` `add_error=False`, un-toggleable), the dead `top_n_templates=90` filter
(defined but never applied), and entity ranking by raw correct count vs accuracy rate. See
`configs/gpu/ravel_gpu.yaml` and the RAVEL preregistration section.

## Reconnect / teardown
- **Reconnect:** `describe-instances` (tag `ravel-gpu`) → `start-instances` if it self-stopped → **re-fetch
  the public IP** (it changes on stop/start without an Elastic IP) → `ssh -i ~/ravel-key.pem ubuntu@$IP` →
  `tmux attach -t ravel`.
- **Teardown (to $0):** `ID=$(aws ec2 describe-instances --region us-east-1 --filters
  Name=tag:Name,Values=ravel-gpu --query 'Reservations[].Instances[].InstanceId' --output text); aws ec2
  terminate-instances --region us-east-1 --instance-ids $ID`. `AUTO_SHUTDOWN` only *stops* the box (compute
  billing ends; ~$0.10/hr EBS remains until you terminate).