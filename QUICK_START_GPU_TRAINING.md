# PHASE 2 GPU TRAINING: QUICK START GUIDE

**TL;DR** - Get GPU training running in 15 minutes

---

## 1️⃣ Prerequisites Check (5 min)

```bash
# Verify CUDA 12.1
nvcc --version
nvidia-smi  # Should show GPU with ~24GB memory

# Verify Python 3.11
python --version

# Verify GitHub runner active
ps aux | grep runsvc.sh  # Linux
Get-Process | Select-String "RunService"  # Windows
```

---

## 2️⃣ Setup S3 Access (3 min)

Add to GitHub repo secrets:

```
AWS_ACCESS_KEY_ID=your_key_here
AWS_SECRET_ACCESS_KEY=your_secret_here
```

Via GitHub CLI:

```bash
gh secret set AWS_ACCESS_KEY_ID --body "your_key"
gh secret set AWS_SECRET_ACCESS_KEY --body "your_secret"
```

---

## 3️⃣ Configure GPU Runner (5 min)

Verify runner is registered:

```bash
cd ~/github-runner
./run.sh  # Or svc.sh status
```

Expected output:

```
✓ Connected to GitHub
✓ Runner version: 2.312.0
✓ Labels: self-hosted, gpu, cuda-12
```

---

## 4️⃣ Trigger Training (2 min)

### Option A: Manual Dispatch

```bash
gh workflow run training_ci.yml --ref main
```

### Option B: Push to Trigger

```bash
git push origin main
# Training starts automatically
```

### Option C: View Workflow

```bash
gh run list --workflow training_ci.yml --limit 5
gh run view <RUN_ID> --log
```

---

## 5️⃣ Monitor Training

### Watch Live Logs

```bash
gh run watch <RUN_ID>
```

### Check GPU Status

```bash
nvidia-smi -l 1  # Refresh every 1 second
```

### View Metrics

```bash
# After training completes:
gh run download <RUN_ID> -n training-metrics-<RUN_NUM>
cat training_metrics.json | jq .
```

---

## 🔍 Quick Troubleshooting

### Issue: GPU Not Found

```bash
# Solution:
export CUDA_VISIBLE_DEVICES=0
nvidia-smi
python -c "import torch; assert torch.cuda.is_available()"
```

### Issue: Out of Memory

```yaml
# Edit configs/training_configuration.yaml:
training:
  batch_size: 2
  gradient_accumulation_steps: 8
```

### Issue: S3 Upload Fails

```bash
# Verify credentials:
aws s3 ls s3://ryzen-llm-checkpoints/
# If fails, check GitHub secrets
```

### Issue: Runner Not Picking Up Jobs

```bash
# Restart runner:
cd ~/github-runner
./svc.sh stop
./svc.sh start

# Or check runner logs:
cat ./runner.log | tail -50
```

---

## 📊 Expected Results

After ~1 hour, you should see:

✅ GitHub Artifacts:

- `model-checkpoint-N` (latest.pt)
- `training-metrics-N` (training_metrics.json)
- `training-logs-N` (training.log)

✅ S3 Bucket:

- `s3://ryzen-llm-checkpoints/phase2/checkpoint-N.pt`
- `s3://ryzen-llm-checkpoints/metrics/metrics-N.json`
- `s3://ryzen-llm-checkpoints/manifests/run-N-manifest.json`

✅ W&B Dashboard:

- Real-time loss curves
- GPU memory tracking
- Training velocity

---

## 🚀 Next Steps

1. **Schedule Daily Runs**

   ```bash
   # Already configured in .github/workflows/training_ci.yml
   # Default: 2 PM UTC daily
   # Edit cron to change
   ```

2. **Setup W&B Dashboard**
   - Visit: https://wandb.ai/ryzen-llm-phase2
   - Connect: `wandb login`
   - View metrics in real-time

3. **Download Checkpoints**

   ```bash
   aws s3 cp s3://ryzen-llm-checkpoints/phase2/checkpoint-42.pt ./models/
   ```

4. **Resume Training**
   ```yaml
   # Edit configs/training_configuration.yaml:
   checkpointing:
     resume_from_checkpoint: path/to/checkpoint-42.pt
   ```

---

## 📞 Full Documentation

See complete docs:

- `GPU_RUNNER_SETUP.md` - Detailed runner setup
- `PHASE2_GPU_TRAINING_CICD_COMPLETE.md` - Full execution details
- `.github/workflows/training_ci.yml` - Workflow definition
- `configs/training_configuration.yaml` - Training config reference

---

**Ready?** Run: `gh workflow run training_ci.yml --ref main`
