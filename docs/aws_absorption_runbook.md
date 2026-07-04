# AWS runbook — absorption suite (Pythia-160M now; Gemma later)

Run the full 42-SAE absorption suite under **both** sae-bench versions (0.3.2 = matches published,
0.6.0 = current code) on an AWS GPU box, produce a **durable run record**, pull it back, and **terminate**.
Pythia-160M is tiny and ungated → **~$2–4, ~1 hour**. This is the dress rehearsal for the (later) Gemma
run. The only real cost risk is a forgotten instance; the guardrails below make that near-impossible.

> **Cloud-agnostic:** the code has no AWS-specific bits. Any CUDA box (RunPod/Lambda/Colab) works — jump
> to **Step 3 (Setup)**. Pythia needs no GPU at all (a `c6i.2xlarge` CPU box is cheaper), but a GPU box
> rehearses the exact Gemma path. Commands below use the AWS CLI; every step has a Console equivalent.

---

## Step 0 — one-time safety: budget alarm (5 min, do this first)
Console: **Billing → Budgets → Create budget → Cost budget → $20/month → alert at 80% and 100% → your email.**
CLI:
```bash
cat > /tmp/budget.json <<'JSON'
{"BudgetName":"absorption-cap","BudgetLimit":{"Amount":"20","Unit":"USD"},"TimeUnit":"MONTHLY","BudgetType":"COST"}
JSON
cat > /tmp/notify.json <<'JSON'
[{"Notification":{"NotificationType":"ACTUAL","ComparisonOperator":"GREATER_THAN","Threshold":80},
  "Subscribers":[{"SubscriptionType":"EMAIL","Address":"YOU@example.com"}]}]
JSON
aws budgets create-budget --account-id "$(aws sts get-caller-identity --query Account --output text)" \
  --budget file:///tmp/budget.json --notifications-with-subscribers file:///tmp/notify.json
```
Pythia won't come close to $20; this just guarantees you get pinged if anything is ever left running.

## Step 1 — check GPU quota (once)
Console: **Service Quotas → EC2 →** "All G and VT **Spot** Instance Requests" must be ≥ 4 vCPUs (g5.xlarge = 4).
If it's 0, request an increase (can take hours–1 day). CLI check:
```bash
aws service-quotas get-service-quota --service-code ec2 --quota-code L-3819A6DF --region us-east-1 \
  --query 'Quota.Value'   # G/VT Spot vCPUs; if 0 -> request-service-quota-increase --desired-value 4
```

## Step 2 — launch (spot, terminate-on-shutdown)
Find the current Deep Learning AMI (CUDA+PyTorch preinstalled) via SSM, then launch:
```bash
AMI=$(aws ssm get-parameter --region us-east-1 \
  --name /aws/service/deeplearning/ami/x86_64/oss-nvidia-driver-gpu-pytorch-2.4-ubuntu22.04/latest/ami-id \
  --query 'Parameter.Value' --output text)

aws ec2 run-instances --region us-east-1 \
  --image-id "$AMI" --instance-type g5.xlarge \
  --key-name YOUR_KEYPAIR --security-group-ids sg-XXXX \
  --instance-market-options '{"MarketType":"spot"}' \
  --instance-initiated-shutdown-behavior terminate \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":50,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=absorption-pythia}]' \
  --count 1
```
Two cost-critical flags: **`--instance-market-options spot`** (~$0.40/hr vs $1.00 on-demand; our runner is
resumable so a spot interruption just resumes) and **`--instance-initiated-shutdown-behavior terminate`**
— so the driver's `sudo shutdown` at the end **terminates** the box (EBS released, **zero** lingering
cost), not merely stops it. `sg-XXXX` = a security group allowing inbound SSH (port 22) from your IP.

Get the IP: `aws ec2 describe-instances --filters Name=tag:Name,Values=absorption-pythia \
Name=instance-state-name,Values=running --query 'Reservations[].Instances[].PublicIpAddress' --output text`

## Step 3 — setup (on the box, ~5 min)
```bash
ssh -i YOUR_KEYPAIR.pem ubuntu@<IP>
# dead-man's switch: auto-terminate in 3h no matter what happens (belt & suspenders)
sudo shutdown -h +180 &

sudo apt-get update -y && sudo apt-get install -y git python3.11-venv
git clone https://github.com/meowmaryexe/SAEBench-Reliability.git && cd SAEBench-Reliability
git checkout alor/ravel-abs-unlearning

# two pinned venvs
python3.11 -m venv ~/venv-060 && ~/venv-060/bin/pip -q install "sae-bench" "transformers>=4.51,<5"
python3.11 -m venv ~/venv-032 && ~/venv-032/bin/pip -q install \
  "sae-bench @ git+https://github.com/adamkarvonen/SAEBench.git@141aff72928f7588c1451bed47c401e1d565d471" \
  "sae_lens==5.3.1" "transformers>=4.40,<5"
```
Each venv keeps its own upstream cache (`RESULTS_DIR`/`PROBES_DIR` are package-relative), so the two
versions never cross-contaminate even in the same working dir. Pythia-160M is ungated → **no HF login**.

## Step 4 — run (one command; resumable; auto-terminates on success)
```bash
tmux new -s abs   # so it survives an SSH drop
VENV_032=~/venv-032/bin/python VENV_060=~/venv-060/bin/python \
  DEVICE=cuda AUTO_SHUTDOWN=1 \
  bash scripts/run_absorption_suite.sh 2>&1 | tee suite.log
```
This runs all 42 SAEs under both versions (resumable — safe to re-run after a spot interruption), fetches
the published numbers, aggregates, prints + saves the report, writes the **run record**, and — because
`AUTO_SHUTDOWN=1` + the terminate-on-shutdown flag — **terminates the box on success** (60 s grace).
~1.5 GPU-h. Runs **Pythia only** by default; for Gemma later, uncomment the `SUITES=(...)` line in
`scripts/run_absorption_suite.sh` (and `huggingface-cli login` first — Gemma is gated).

## Step 5 — recordkeeping & get results back (before it shuts down)
Every run writes a durable, git-tracked record to **`docs/run_records/absorption/<suite>_<UTC>/`**:
- `run_record.md` — human summary: per-version provenance (sae-bench/sae_lens/transformer_lens/
  transformers/torch versions, host, device, GPU, git SHA+dirty, UTC timestamps), a per-architecture
  results table (published vs 0.3.2 vs 0.6.0, fraction + full), and the embedded report.
- `run_record.json` — the same, machine-readable (self-contained: embeds the by-arch results, so it
  survives even though the bulky raw outputs under `results/` are gitignored).
- `report.txt` — the raw report text.

Commit it from the box (this is the audit trail):
```bash
git config user.email you@example.com && git config user.name "you"
git add docs/run_records/absorption/  && git commit -m "Pythia absorption suite run record (both versions)" && git push
```
Optionally archive raw outputs too: `aws s3 sync results/ s3://YOUR_BUCKET/absorption-results/` (or
`scp -r ubuntu@<IP>:~/SAEBench-Reliability/results ./`). Raw per-SAE outputs + `suite.log` are gitignored
by design; the committed `run_record.*` is the durable record.

## Step 6 — teardown (verify!)
With `AUTO_SHUTDOWN=1` + terminate-on-shutdown, the box self-terminates on success. **Confirm:**
```bash
aws ec2 describe-instances --filters Name=tag:Name,Values=absorption-pythia \
  --query 'Reservations[].Instances[].State.Name' --output text   # expect: terminated (or empty)
```
If a run failed before auto-shutdown, terminate manually:
`aws ec2 terminate-instances --instance-ids <id>`.

## Cost & guardrails — why this can't run away
| lever | setting |
|---|---|
| estimate | 42 SAEs × ~1 min × 2 versions ≈ ~1.5 GPU-h + setup ≈ **~$2 spot / ~$3 on-demand** |
| auto-terminate on completion | `AUTO_SHUTDOWN=1` + `--instance-initiated-shutdown-behavior terminate` |
| dead-man's switch | `sudo shutdown -h +180 &` at login → auto-terminate in 3 h regardless |
| budget alarm | AWS Budgets $20, email at 80/100% (Step 0) |
| spot | ~60% cheaper; resumable runner tolerates interruptions |
| terminate (not stop) | EBS released on shutdown → zero lingering charge |
| right-size | Pythia needs no GPU — `c6i.2xlarge` CPU is cheaper if not rehearsing Gemma |

## Verify it worked
Open the committed `docs/run_records/absorption/<suite>_<UTC>/run_record.md`. For `absorption_fraction`:
**0.3.2 within ~0.05 of published** across archs (reproduced) and **0.6.0 well below** (the redefinition);
`full_absorption` reproduces under both; and the Spearman ρ answers the ranking-stability question (only
meaningful once all 42 ran — the report warns on partial runs). Then confirm the instance is terminated.

## Later: Gemma
Same box, bigger model. Changes: `huggingface-cli login` (accept the Gemma license); uncomment the Gemma
suites in `run_absorption_suite.sh`; run with `DEVICE=cuda LLM_DTYPE=bfloat16`. Cost ~$4–24 per
width/version (see `configs/gpu/absorption_gpu.yaml`). Everything else — record, guardrails — is identical.
