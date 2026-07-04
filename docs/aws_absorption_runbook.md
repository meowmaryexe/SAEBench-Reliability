# AWS runbook — absorption suite (Pythia-160M now; Gemma later)

Run the full 42-SAE absorption suite under **both** sae-bench versions (0.3.2 = matches published,
0.6.0 = current code) on an AWS GPU box, produce a **durable run record**, pull it back, and **terminate**.
Pythia-160M is tiny and ungated → **~$2–4, ~1 hour**. This is the dress rehearsal for the (later) Gemma
run. The only real cost risk is a forgotten instance; the guardrails below make that near-impossible.

> **Cloud-agnostic:** the code has no AWS-specific bits. Any CUDA box (RunPod/Lambda/Colab) works — jump
> to **Step 3 (Setup)**. Pythia needs no GPU at all (a `c6i` CPU box is cheaper), but a GPU box rehearses
> the exact Gemma path. Commands below use the AWS CLI; every step has a Console equivalent.

---

# ⭐ Pythia now — CPU path (no GPU quota needed)

**Use this to run Pythia today** without requesting a GPU quota. Pythia-160M is tiny and runs on CPU;
a standard `c6i` instance uses the ordinary "Running On-Demand Standard instances" quota every account
already has. Slower than GPU (it's an overnight run) but ~$5 and zero quota hassle. The GPU sections
further down are the reference for the (later) Gemma run.

### 1. Budget alarm — do **Step 0** below first (5 min).

### 2. Launch a CPU box
```bash
# Ubuntu 22.04 AMI id for your region
AMI=$(aws ssm get-parameter --region us-east-1 \
  --name /aws/service/canonical/ubuntu/server/22.04/stable/current/amd64/hvm/ebs-gp2/ami-id \
  --query Parameter.Value --output text)

aws ec2 run-instances --region us-east-1 \
  --image-id "$AMI" --instance-type c6i.4xlarge \
  --key-name YOUR_KEYPAIR --security-group-ids sg-XXXX \
  --instance-initiated-shutdown-behavior terminate \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":40,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=absorption-pythia-cpu}]' --count 1
```
- **`c6i.4xlarge`** = 16 vCPU, 32 GB, ~$0.68/hr on-demand → the run is ~$5. (Cheaper: `c6i.2xlarge` = 8
  vCPU ~$0.34/hr but ~1.5× slower. If a brand-new account caps you at 5 vCPU, use `c6i.xlarge` (4 vCPU)
  or request a small Standard-quota bump.)
- **On-demand** (not spot) here: the run is long (~overnight), so predictable > cheap. (Spot works too —
  the runner is resumable — but a mid-run interruption means you re-launch and resume.)
- **`--instance-initiated-shutdown-behavior terminate`** → the driver's auto-shutdown *terminates* the box
  (EBS released, zero lingering cost).

### 3. Setup (on the box, ~5–10 min)
```bash
ssh -i YOUR_KEYPAIR.pem ubuntu@<IP>
sudo shutdown -h +960 &   # dead-man's switch: auto-terminate in 16 h (> the ~10 h run) no matter what

sudo apt-get update -y && sudo apt-get install -y git python3-venv python3-pip
git clone https://github.com/meowmaryexe/SAEBench-Reliability.git && cd SAEBench-Reliability
git checkout alor/ravel-abs-unlearning

python3 -m venv ~/venv-060 && ~/venv-060/bin/pip -q install -U pip \
  && ~/venv-060/bin/pip -q install "sae-bench" "transformers>=4.51,<5"
python3 -m venv ~/venv-032 && ~/venv-032/bin/pip -q install -U pip \
  && ~/venv-032/bin/pip -q install \
     "sae-bench @ git+https://github.com/adamkarvonen/SAEBench.git@141aff72928f7588c1451bed47c401e1d565d471" \
     "sae_lens==5.3.1" "transformers>=4.40,<5"
```
(No HF login — Pythia is ungated. Each venv keeps its own cache, so the two versions never contaminate.)

### 4. Run (in tmux, `DEVICE=cpu`, auto-terminates on success)
```bash
tmux new -s abs
VENV_032=~/venv-032/bin/python VENV_060=~/venv-060/bin/python \
  DEVICE=cpu AUTO_SHUTDOWN=1 \
  bash scripts/run_absorption_suite.sh 2>&1 | tee suite.log
# detach with Ctrl-b d ; reattach later with: tmux attach -t abs
```
Runs all 42 SAEs × both versions (resumable), fetches published numbers, aggregates, writes the run
record, and **terminates the box on completion**. Expect **~8–10 h wall, ~$5**.

### 5–6. Records & teardown — identical to **Step 5** and **Step 6** below (commit
`docs/run_records/absorption/`, then confirm the instance is `terminated`).

> Prefer GPU (≈1.5 h instead of ~10 h)? You need the G/VT quota — see **Step 1** below — then follow the
> GPU steps (`g5.xlarge`, `DEVICE=cuda`). You'll need a GPU for Gemma anyway, so requesting that quota now
> is worthwhile even if you run Pythia on CPU today.

---

# GPU path (faster; required for Gemma) — reference

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
