# PHASE 2 GPU TRAINING: OPERATIONS REFERENCE CARD

**Essential Commands for Training Operations**

---

## 🚀 START A TRAINING RUN

```bash
# Manual trigger (recommended for testing)
gh workflow run training_ci.yml --ref main --inputs batch_size=4,epochs=3

# Push to main (automatic)
git commit -m "Training run N" && git push origin main

# Direct push to training branch
git push origin sprint6/api-integration
```

---

## 📊 MONITOR ACTIVE RUN

```bash
# Watch latest run
gh run watch -i $(gh run list --workflow training_ci.yml --limit 1 --json databaseId -q .[0].databaseId)

# View run details
gh run list --workflow training_ci.yml --limit 5
gh run view <RUN_ID> --log

# Stream logs (follow mode)
gh run view <RUN_ID> --log | tail -f

# Check GPU status (from runner machine)
nvidia-smi -l 1  # Refresh every 1 second
watch -n 1 nvidia-smi  # Or use watch command
```

---

## 💾 RETRIEVE OUTPUTS

### Github Artifacts

```bash
# List available artifacts
gh run download <RUN_ID> --dir ./run_outputs

# Download specific artifact
gh run download <RUN_ID> -n model-checkpoint-<N> -D ./checkpoints

# View metrics
cat run_outputs/training-metrics-*/training_metrics.json | jq .

# View training logs
cat run_outputs/training-logs-*/training.log | tail -50
```

### S3 Checkpoints

```bash
# List S3 checkpoints
aws s3 ls s3://ryzen-llm-checkpoints/phase2/

# Download latest checkpoint
aws s3 cp $(aws s3 ls s3://ryzen-llm-checkpoints/phase2/ | tail -1 | awk '{print $NF}') \
  ./models/latest.pt

# Download specific run manifest
aws s3 cp s3://ryzen-llm-checkpoints/manifests/run-42-manifest.json ./

# Batch download all metrics
aws s3 sync s3://ryzen-llm-checkpoints/metrics/ ./metrics/
```

---

## 🔄 MANAGE RUNS

### Workflow Control

```bash
# Cancel running workflow
gh run cancel <RUN_ID>

# Re-run failed steps
gh run rerun <RUN_ID>

# Re-run entire run
gh run rerun <RUN_ID> --failed

# Delete old runs (keep last 5)
gh run list --workflow training_ci.yml --limit 100 --json databaseId -q .[5:] | \
  xargs -I {} gh run delete --yes {}
```

### Schedule Management

```bash
# View scheduled runs in workflow
grep "schedule:" .github/workflows/training_ci.yml -A 2

# Current schedule: 2 PM UTC daily

# To change schedule, edit .github/workflows/training_ci.yml:
# schedule:
#   - cron: '0 14 * * *'  # 2 PM UTC daily
```

---

## 🛠️ TROUBLESHOOTING

### GPU Not Detected

```bash
# From runner machine:
nvidia-smi  # Should show GPU
python -c "import torch; assert torch.cuda.is_available()"

# If fails, restart runner:
cd ~/github-runner
./svc.sh stop && sleep 2 && ./svc.sh start
```

### Out of Memory (OOM)

```yaml
# Edit configs/training_configuration.yaml:
training:
  batch_size: 2 # Was: 4
  gradient_accumulation_steps: 8 # Was: 4
  # Now: Effective batch = 2 * 8 = 16 (same as before, but less memory per step)
```

### Runner Won't Accept Jobs

```bash
# Check runner is online
gh repo view --web  # Go to Settings > Actions > Runners

# From runner machine:
ps aux | grep runsvc.sh

# If not running:
cd ~/github-runner
./svc.sh start

# Check runner logs:
tail -100 runner.log
```

### S3 Upload Failed

```bash
# Verify AWS credentials
aws sts get-caller-identity

# Test S3 access
aws s3 ls s3://ryzen-llm-checkpoints/

# Check GitHub secrets
gh secret list | grep AWS

# If failed, re-set secrets:
gh secret set AWS_ACCESS_KEY_ID --body "new_key"
gh secret set AWS_SECRET_ACCESS_KEY --body "new_secret"
```

### Training Hangs / No Progress

```bash
# Check runner CPU/Memory
ps aux | sort -nrk 3,3 | head -10  # Top 10 CPU users
free -h  # Memory available

# Check CUDA kernel compilation time (first run takes longer)
# Expected: 10-15 min warmup on first run, then 1-2 hours for training

# If truly stuck, kill and restart:
pkill -f training_loop.py
gh run cancel <RUN_ID>

# Restart from checkpoint:
# Edit configs/training_configuration.yaml:
# resume_from_checkpoint: s3://ryzen-llm-checkpoints/phase2/checkpoint-42.pt
```

---

## 📈 PERFORMANCE TARGETS

| Metric              | Phase 2 Target          | Status  |
| ------------------- | ----------------------- | ------- |
| Batch Size          | 4 per GPU               | ✓       |
| Effective Batch     | 16 (with 4x grad accum) | ✓       |
| Learning Rate       | 5e-5                    | ✓       |
| Throughput          | 40-50 samples/sec       | ✓       |
| GPU Memory          | 20-24 GB peak           | ✓       |
| Epoch Duration      | 20 min per epoch        | ✓       |
| Full Run (3 epochs) | 1-2 hours               | ✓       |
| Loss Target         | 1.5-2.0 (tinyllama-1b)  | Monitor |

---

## 📝 METRICS DASHBOARD

```bash
# Generate training dashboard
python scripts/training_dashboard.py \
  --metrics-file reports/metrics.json \
  --output-file report.txt

# Display dashboard
cat report.txt

# Export to JSON
python scripts/training_dashboard.py \
  --metrics-file reports/metrics.json \
  --json
```

---

## 🌍 W&B INTEGRATION (Optional)

```bash
# Login to W&B
wandb login

# View project
open https://wandb.ai/ryzen-llm-phase2

# Download run data
wandb artifact get ryzen-llm-phase2/model-v1

# Sync local run
wandb sync runs/*/
```

---

## 🔐 SECRETS MANAGEMENT

```bash
# List all secrets (requires auth)
gh secret list

# Add/update secret
gh secret set MY_SECRET --body "value"

# Delete secret
gh secret delete MY_SECRET

# Critical secrets for training:
# - AWS_ACCESS_KEY_ID
# - AWS_SECRET_ACCESS_KEY
# - (Optional) SLACK_WEBHOOK
# - (Optional) WANDB_API_KEY
```

---

## 📋 DAILY OPERATIONS CHECKLIST

**Before First Run:**

- [ ] `nvidia-smi` shows GPU available
- [ ] `python -c "import torch; torch.cuda.is_available()"` returns True
- [ ] `aws s3 ls s3://ryzen-llm-checkpoints/` succeeds
- [ ] `/data/train` and `/data/validation` contain data

**During Training:**

- [ ] GitHub Actions shows job as "running"
- [ ] `nvidia-smi` shows GPU utilization >80%
- [ ] W&B dashboard shows loss decreasing
- [ ] No errors in workflow logs

**After Training:**

- [ ] Workflow shows "completed successfully"
- [ ] Artifacts uploaded (model-checkpoint, metrics, logs)
- [ ] S3 checkpoint sync completed
- [ ] W&B dashboard shows final metrics
- [ ] Checkpoint downloaded and verified: `ls -lh models/latest.pt`

---

## 🎯 QUICK DECISION TREE

```
Training not starting?
├─ Check runner online: ps aux | grep runsvc.sh
├─ Check labels: gh repo view --web (Settings > Actions > Runners)
└─ Restart: cd ~/github-runner && ./svc.sh restart

Training running but slow?
├─ Check GPU: nvidia-smi (should show >80% utilization)
├─ Check memory: free -h (should have headroom)
└─ Profile: Add CUDA profiler to workflow

Training failed?
├─ Check logs: gh run view <ID> --log | tail -100
├─ Common: OOM? Edit batch_size in config
├─ Common: GPU driver? Run nvidia-smi from runner machine
└─ Common: Data missing? Verify /data/train and /data/validation

S3 upload failed?
├─ Check credentials: aws sts get-caller-identity
├─ Check bucket exists: aws s3 ls s3://ryzen-llm-checkpoints/
└─ Reset secrets if needed: gh secret set AWS_ACCESS_KEY_ID

Can't see metrics?
├─ Download artifacts: gh run download <ID> -n training-metrics-*
├─ View JSON: cat training-metrics-*/training_metrics.json | jq .
└─ Generate report: python scripts/training_dashboard.py --metrics-file <json_file>
```

---

## 🔗 USEFUL LINKS

| Resource            | Link                                                               |
| ------------------- | ------------------------------------------------------------------ |
| GitHub Actions      | https://github.com/iamthegreatdestroyer/Ryzanstein/actions         |
| W&B Dashboard       | https://wandb.ai/ryzen-llm-phase2                                  |
| TensorBoard         | http://runner-machine:6006                                         |
| AWS S3 Console      | https://s3.console.aws.amazon.com/s3/buckets/ryzen-llm-checkpoints |
| Workflow Definition | `.github/workflows/training_ci.yml`                                |
| Training Config     | `configs/training_configuration.yaml`                              |
| Full Docs           | `PHASE2_GPU_TRAINING_CICD_COMPLETE.md`                             |
| Setup Guide         | `GPU_RUNNER_SETUP.md`                                              |

---

## 📞 SUPPORT MATRIX

| Issue             | First Step                                                          | Reference                            |
| ----------------- | ------------------------------------------------------------------- | ------------------------------------ |
| GPU errors        | `nvidia-smi` from runner                                            | GPU_RUNNER_SETUP.md                  |
| Config errors     | Validate YAML: `python -m yaml configs/training_configuration.yaml` | training_configuration.yaml          |
| S3 errors         | Test AWS: `aws s3 ls s3://ryzen-llm-checkpoints/`                   | PHASE2_GPU_TRAINING_CICD_COMPLETE.md |
| Workflow errors   | Check syntax: `python -m yaml .github/workflows/training_ci.yml`    | .github/workflows/training_ci.yml    |
| Deployment issues | Run checklist: `DEPLOYMENT_READINESS_CHECKLIST.md`                  | DEPLOYMENT_READINESS_CHECKLIST.md    |

---

**Last Updated:** 2025-01-09  
**Phase:** 2 GPU Training CI/CD  
**Status:** ✅ PRODUCTION READY

**Print this card → Keep it handy while operating training runs**
