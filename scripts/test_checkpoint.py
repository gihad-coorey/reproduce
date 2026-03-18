"""
Test a specific checkpoint path and report diagnostics.
Usage: python scripts/test_checkpoint.py --model-path <checkpoint> [--steps 100]
"""
import os, sys
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, "external/lerobot/src")

import argparse
import numpy as np
import torch
from pathlib import Path

from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs import libero as libero_module
from lerobot.envs.utils import preprocess_observation, add_envs_task
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", required=True, help="Path or HuggingFace ID of checkpoint")
parser.add_argument("--steps", type=int, default=100, help="Steps to run in rollout")
parser.add_argument("--task", default="libero_spatial_task_0", help="Task to evaluate")
args = parser.parse_args()

# Patch action dim
orig_step = libero_module.LiberoEnv.step
def patched_step(self, action):
    a = np.asarray(action)
    if a.ndim == 2 and a.shape[1] == 6:
        zeros = np.zeros((a.shape[0], 1), dtype=a.dtype)
        a = np.concatenate([a, zeros], axis=1)
    elif a.ndim == 1 and a.shape[0] == 6:
        a = np.concatenate([a, np.zeros(1, dtype=a.dtype)])
    return orig_step(self, a)
libero_module.LiberoEnv.step = patched_step

model_path = args.model_path
print(f"Testing checkpoint: {model_path}")
print(f"Device: {DEVICE}\n")

try:
    print("Loading model...")
    model = SmolVLAPolicy.from_pretrained(model_path)
    model.to(torch.device(DEVICE)).eval()
    print(f"✓ Model loaded successfully")
except Exception as e:
    print(f"❌ FAILED to load model: {e}")
    sys.exit(1)

# Run single forward pass
suite_name, task_id = args.task.rsplit("_task_", 1)
task_id = int(task_id)

env_cfg = LiberoEnvConfig(task=suite_name, task_ids=[task_id], control_mode="relative")
envs = make_env(cfg=env_cfg, n_envs=1, use_async_envs=False, trust_remote_code=False)
env = envs[suite_name][task_id]
env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=model.config)

# Rename map
feature_keys = set(model.config.input_features.keys())
rename_map = {}
if "observation.images.wrist_image" in feature_keys:
    rename_map["observation.images.image2"] = "observation.images.wrist_image"
if "observation.images.image" in feature_keys:
    rename_map.pop("observation.images.image", None)

preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=model.config,
    pretrained_path=model_path,
    preprocessor_overrides={
        "device_processor": {"device": DEVICE},
        "rename_observations_processor": {"rename_map": rename_map},
    },
)

print(f"\n{'='*80}")
print(f"SINGLE FORWARD PASS TEST")
print(f"{'='*80}")
model.reset()
obs, info = env.reset(seed=[42])
obs = preprocess_observation(obs)
obs = add_envs_task(env, obs)
obs = env_pre(obs)
obs = preprocessor(obs)

try:
    with torch.no_grad():
        action = model.select_action(obs)
    action = postprocessor(action)
    action_np = action.float().cpu().numpy()
    
    if action_np.ndim == 2:
        action_np = action_np[0]
    
    print(f"✓ Forward pass successful")
    print(f"  Action shape: {action_np.shape}")
    print(f"  Action values: {action_np}")
    print(f"  Action finite: {np.isfinite(action_np).all()}")
except Exception as e:
    print(f"❌ Forward pass FAILED: {e}")
    env.close()
    sys.exit(1)

print(f"\n{'='*80}")
print(f"MINI ROLLOUT: {args.steps} steps")
print(f"{'='*80}")

model.reset()
obs, info = env.reset(seed=[42])
done = np.array([False])
step = 0
max_steps = env.call("_max_episode_steps")[0]
rewards = []
action_mags = []
success = False

while not np.all(done) and step < min(max_steps, args.steps):
    obs = preprocess_observation(obs)
    obs = add_envs_task(env, obs)
    obs = env_pre(obs)
    obs = preprocessor(obs)
    
    with torch.no_grad():
        action = model.select_action(obs)
    
    action = postprocessor(action)
    action_transition = {"action": action}
    action_transition = env_post(action_transition)
    action_np = action_transition["action"].to("cpu").numpy()
    
    obs, reward, terminated, truncated, info = env.step(action_np)
    done = terminated | truncated
    
    rewards.append(float(reward[0]))
    if action_np.ndim == 2:
        a = action_np[0]
    else:
        a = action_np
    action_mags.append(np.linalg.norm(a[:6]))
    
    if "final_info" in info and info["final_info"].get("is_success", [False])[0]:
        success = True
    
    if step < 3 or step % 30 == 0 or reward[0] > 0:
        print(f"  step={step:3d} | reward={reward[0]:+.3f} | action_mag={action_mags[-1]:.4f}")
    
    step += 1

env.close()

print(f"\n{'='*80}")
print(f"RESULTS")
print(f"{'='*80}")
print(f"Total steps: {step} / {args.steps}")
print(f"Max reward: {max(rewards) if rewards else 0.0:.4f}")
print(f"Sum rewards: {sum(rewards):.4f}")
print(f"Non-zero rewards: {sum(1 for r in rewards if r > 0)}")
print(f"Success achieved: {'🎯 YES' if success else '❌ NO'}")
print(f"Action magnitude: min={min(action_mags):.4f} max={max(action_mags):.4f} mean={np.mean(action_mags):.4f}")

if success or max(rewards) > 0:
    print(f"\n✅ Checkpoint shows POSITIVE signals!")
else:
    print(f"\n⚠️  Checkpoint shows 0% success (same as tuned checkpoint)")
