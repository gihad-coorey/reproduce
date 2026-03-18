# SmolVLA LIBERO Checkpoint Testing Guide

## Current Status
- ✅ Pipeline diagnostics: **All checks PASS**
- ❌ Current checkpoint (`godnpeter`): **0% success** (100 steps)
- → **Diagnosis: Checkpoint quality issue, NOT pipeline mismatch**

## Next: Test HuggingFace Checkpoints

### Option A: Search for available checkpoints
```bash
python scripts/search_hf_models.py
```
This will list all SmolVLA + LIBERO models on HuggingFace.

### Option B: Test a specific checkpoint
```bash
# Replace <model_id> with checkpoint name (e.g., HuggingFaceVLA/smolvla_libero)
python scripts/test_checkpoint.py --model-path <model_id> --steps 100
```

**Output interpretation:**
- `Max reward: X.XXXX` — Higher is better. 0.0 = same as current checkpoint
- `Success: YES/NO` — Task completed within episode
- `Non-zero rewards: N/50` — How often model got partial credit

### Option C: Automatic comparison
```bash
python scripts/compare_checkpoints.py
```
Tests current checkpoint vs discovered HF alternatives side-by-side.

## Expected Candidates

Search will likely find one of these (if they exist):
```
HuggingFaceVLA/smolvla_libero
lerobot/smolvla_libero
tinyrobotics/smolvla_libero
SmolVLA/smolvla_libero_tuned
<user_org>/smolvla_libero_finetuned
```

## If No HF Checkpoint Works

**Option 1: Fine-tune from scratch**
```bash
# Requires LeRobot installed and LIBERO data at data/libero
lerobot-train \
  --dataset.repo_id=lerobot/libero_combined \
  --policy.type=smolvla \
  --policy.device=mps \
  --output_dir=models/smolvla_libero_finetuned \
  --training.num_steps=200000
```

**Option 2: Check base model on Open-X tasks**
```bash
python scripts/run_eval.py --model-path lerobot/smolvla_base --smoke
```
If base model → 0% on Open-X too, then environment or fundamental pipeline issue remains.

## Tracking Results

Test results are saved to:
- `results/checkpoint_comparison.json` — A/B test results
- `results/hf_checkpoint_results.json` — Full eval results (5 tasks, 20 episodes)

## Quick Decision Tree

```
1. Run: python scripts/search_hf_models.py
   ├─ Found LIBERO checkpoints? → Go to 2a
   └─ No results? → Go to 3

2a. For each found checkpoint:
    Run: python scripts/test_checkpoint.py --model-path <id> --steps 100
    ├─ Max reward > 0? → Run full eval, update report
    └─ Max reward = 0? → Try next checkpoint

3. No candidates found:
   ├─ Option A: Fine-tune from scratch
   ├─ Option B: Search GitHub/Papers for checkpoint links
   └─ Option C: Use base model + domain adaptation approach
```

## Monitoring

After each checkpoint test:
1. Check max_reward value
2. Compare to baseline (current = 0.0)
3. If better: run full eval `python scripts/run_eval.py --model-path <id>`
4. Update session memory with results
