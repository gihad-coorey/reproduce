"""
Compare two checkpoints on LIBERO task 0.
Usage: python scripts/compare_checkpoints.py
"""
import os, sys
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, "external/lerobot/src")

import json
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

def test_checkpoint(model_path, steps=50, task="libero_spatial_task_0"):
    """Test a checkpoint and return metrics."""
    print(f"\nTesting: {model_path}")
    print(f"  Loading model...")
    
    try:
        model = SmolVLAPolicy.from_pretrained(model_path)
        model.to(torch.device(DEVICE)).eval()
    except Exception as e:
        print(f"  ❌ Load failed: {e}")
        return None
    
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
    
    suite_name, task_id = task.rsplit("_task_", 1)
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
    
    try:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=model.config,
            pretrained_path=model_path,
            preprocessor_overrides={
                "device_processor": {"device": DEVICE},
                "rename_observations_processor": {"rename_map": rename_map},
            },
        )
    except Exception as e:
        print(f"  ❌ Preprocessor failed: {e}")
        env.close()
        return None
    
    # Run rollout
    model.reset()
    obs, info = env.reset(seed=[42])
    done = np.array([False])
    step = 0
    max_steps = env.call("_max_episode_steps")[0]
    rewards = []
    action_mags = []
    success = False
    
    while not np.all(done) and step < min(max_steps, steps):
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
        action_mags.append(float(np.linalg.norm(a[:6])))
        
        if "final_info" in info and info["final_info"].get("is_success", [False])[0]:
            success = True
        
        step += 1
    
    env.close()
    
    results = {
        "checkpoint": model_path,
        "steps_run": step,
        "max_reward": float(max(rewards)) if rewards else 0.0,
        "sum_rewards": float(sum(rewards)),
        "non_zero_rewards": int(sum(1 for r in rewards if r > 0)),
        "success": bool(success),
        "action_mag_min": float(min(action_mags)) if action_mags else 0.0,
        "action_mag_max": float(max(action_mags)) if action_mags else 0.0,
        "action_mag_mean": float(np.mean(action_mags)) if action_mags else 0.0,
    }
    
    print(f"  ✓ Ran {step} steps")
    print(f"    Max reward: {results['max_reward']:.4f}")
    print(f"    Success: {results['success']}")
    print(f"    Non-zero rewards: {results['non_zero_rewards']}")
    
    return results

# Test both checkpoints
print("="*80)
print("CHECKPOINT COMPARISON TEST")
print("="*80)

results = {}

# Test current tuned checkpoint
current = test_checkpoint("models/smolvla_libero_tuned_migrated", steps=50)
if current:
    results["current_tuned"] = current

# Test HuggingFace checkpoint
print("\nAttempting to load HuggingFace checkpoint...")
hf_checkpoint = test_checkpoint("HuggingFaceVLA/smolvla_libero", steps=50)
if hf_checkpoint:
    results["huggingface_smolvla_libero"] = hf_checkpoint
else:
    print("  Trying alternative name: lerobot/smolvla_libero")
    hf_checkpoint = test_checkpoint("lerobot/smolvla_libero", steps=50)
    if hf_checkpoint:
        results["lerobot_smolvla_libero"] = hf_checkpoint

# Summary
print("\n" + "="*80)
print("COMPARISON SUMMARY")
print("="*80)

for name, metrics in results.items():
    print(f"\n{name}:")
    print(f"  Max reward: {metrics['max_reward']:.4f}")
    print(f"  Success: {'✅ YES' if metrics['success'] else '❌ NO'}")
    print(f"  Non-zero rewards: {metrics['non_zero_rewards']} / {metrics['steps_run']}")

# Save results
result_file = Path("results/checkpoint_comparison.json")
result_file.parent.mkdir(parents=True, exist_ok=True)
with open(result_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to {result_file}")

# Recommend next step
if results:
    best_checkpoint = max(results.items(), key=lambda x: x[1]["max_reward"])
    print(f"\nBest checkpoint so far: {best_checkpoint[0]}")
    print(f"  Max reward: {best_checkpoint[1]['max_reward']:.4f}")
