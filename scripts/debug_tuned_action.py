"""
Quick diagnostic: run 1 episode with tuned model, print action stats + observation keys.
Usage: python scripts/debug_tuned_action.py [--verbose]
Enhanced with per-stage tensor diagnostics and validation checks.
"""
import os, sys
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import json
import numpy as np
import torch
import argparse

# Make sure lerobot is importable
sys.path.insert(0, "external/lerobot/src")

from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs import libero as libero_module
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

MODEL_PATH = "models/smolvla_libero_tuned_migrated"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {DEVICE}")

def print_tensor_stats(name, tensor, verbose=False):
    """Print diagnostics for a tensor."""
    if isinstance(tensor, torch.Tensor):
        t = tensor.float().detach().cpu()
        has_nan = t.isnan().any().item()
        has_inf = t.isinf().any().item()
        summary = f"{name}: shape={tuple(t.shape)} dtype={tensor.dtype} min={t.min().item():.6f} max={t.max().item():.6f} mean={t.mean().item():.6f} std={t.std().item():.6f}"
        if has_nan or has_inf:
            summary += f" ⚠️ NaN={has_nan} Inf={has_inf}"
        print(f"  {summary}")
        if verbose and tensor.numel() <= 16:
            print(f"    values: {t.reshape(-1).tolist()}")
    else:
        print(f"  {name}: {type(tensor).__name__}")

def print_obs_dict(obs_dict, name="", verbose=False):
    """Print diagnostics for observation dictionary."""
    if name:
        print(f"\n--- {name} ---")
    for k, v in sorted(obs_dict.items()):
        if isinstance(v, torch.Tensor):
            print_tensor_stats(k, v, verbose=verbose)
        elif isinstance(v, dict):
            print(f"  {k}: dict with keys {list(v.keys())}")
        else:
            print(f"  {k}: {type(v).__name__}")

# Patch action dim
original_step = libero_module.LiberoEnv.step
def patched_step(self, action):
    action_np = np.asarray(action)
    if action_np.ndim == 1 and action_np.shape[0] == 6:
        action_np = np.concatenate([action_np, np.array([0.0], dtype=action_np.dtype)])
    elif action_np.ndim == 2 and action_np.shape[1] == 6:
        zeros = np.zeros((action_np.shape[0], 1), dtype=action_np.dtype)
        action_np = np.concatenate([action_np, zeros], axis=1)
    return original_step(self, action_np)
libero_module.LiberoEnv.step = patched_step

parser = argparse.ArgumentParser()
parser.add_argument("--verbose", action="store_true", help="Print full tensor values")
args = parser.parse_args()

print("Loading tuned model...")
model = SmolVLAPolicy.from_pretrained(MODEL_PATH)
model.to(torch.device(DEVICE))
model.eval()

# Build rename map from model config
feature_keys = set(model.config.input_features.keys())
print(f"\nModel input features: {sorted(feature_keys)}")
print(f"Model output features: {sorted(model.config.output_features.keys())}")

# Check required action feature
action_feature = model.config.output_features.get("action", {})
action_shape = getattr(action_feature, "shape", "unknown") if hasattr(action_feature, "shape") else str(action_feature)
print(f"Model action output: {action_feature}")

rename_map = {}
if "observation.images.camera1" in feature_keys:
    rename_map["observation.images.image"] = "observation.images.camera1"
if "observation.images.camera2" in feature_keys:
    rename_map["observation.images.image2"] = "observation.images.camera2"
if "observation.images.wrist_image" in feature_keys:
    rename_map["observation.images.image2"] = "observation.images.wrist_image"
if "observation.images.image" in feature_keys:
    rename_map.pop("observation.images.image", None)
print(f"Image rename map: {rename_map}")

preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=model.config,
    pretrained_path=MODEL_PATH,
    preprocessor_overrides={
        "device_processor": {"device": DEVICE},
        "rename_observations_processor": {"rename_map": rename_map},
    },
)

# Build env
env_cfg = LiberoEnvConfig(task="libero_spatial", task_ids=[0], control_mode="relative")
envs = make_env(cfg=env_cfg, n_envs=1, use_async_envs=False, trust_remote_code=False)
env = envs["libero_spatial"][0]
env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=model.config)

# These are needed to replicate what rollout() does
from lerobot.envs.utils import preprocess_observation, add_envs_task

# Reset env with deterministic seed
print("\n" + "="*80)
print("STEP 1: Raw environment observation")
print("="*80)
obs, info = env.reset(seed=[42])
print(f"Reset seed: 42 | Info: {info}")

for k, v in sorted(obs.items()):
    if isinstance(v, np.ndarray):
        print(f"  {k}: shape={v.shape} dtype={v.dtype} min={v.min():.4f} max={v.max():.4f}")
    elif isinstance(v, dict):
        print(f"  {k}: dict with keys {list(v.keys())}")
    else:
        print(f"  {k}: {type(v).__name__}")

print("\n" + "="*80)
print("STEP 2: After preprocess_observation (numpy→tensor)")
print("="*80)
obs = preprocess_observation(obs)
print_obs_dict(obs, verbose=args.verbose)

print("\n" + "="*80)
print("STEP 3: Add task language instruction")
print("="*80)
obs = add_envs_task(env, obs)
if 'task' in obs:
    task_text = obs['task'][0] if isinstance(obs['task'], list) else obs['task']
    print(f"✓ Task instruction: {str(task_text)[:100]}")
    assert isinstance(task_text, str) and len(task_text) > 0, "❌ Task instruction is empty or not a string!"
else:
    print("❌ FAILURE: 'task' key missing from observation!")
    sys.exit(1)

print("\n" + "="*80)
print("STEP 4: After env_preprocessor (LiberoProcessorStep)")
print("="*80)
obs_proc = env_preprocessor(obs)
print_obs_dict(obs_proc, verbose=args.verbose)

print("\n" + "="*80)
print("STEP 5: After policy preprocessor (rename, tokenize, normalize)")
print("="*80)
obs_final = preprocessor(obs_proc)
print_obs_dict(obs_final, verbose=args.verbose)

# Explicit validation checks
print("\n" + "="*80)
print("VALIDATION CHECKS")
print("="*80)

# Check required image keys
required_images = set(model.config.image_features.keys())
actual_images = {k for k in obs_final.keys() if k.startswith("observation.images.")}
missing = required_images - actual_images
if missing:
    print(f"❌ Missing image keys: {missing}")
    print(f"   Available: {actual_images}")
    sys.exit(1)
else:
    print(f"✓ All required image keys present: {required_images}")

# Check state normalization
if "observation.state" in obs_final:
    state = obs_final["observation.state"].float().cpu()
    state_val = state.mean().item()
    state_std = state.std().item()
    print(f"✓ observation.state: mean={state_val:.6f} std={state_std:.6f}")
    if state_val > 10 or state_std > 10:
        print(f"  ⚠️  Warning: State values appear unnormalized (too large)")
else:
    print("⚠️  observation.state not in preprocessor output")

print("\n" + "="*80)
print("STEP 6: Model.select_action() with one forward pass")
print("="*80)
print("Running select_action...")
with torch.no_grad():
    action = model.select_action(obs_final)

print_tensor_stats("action output", action, verbose=args.verbose)

print("\n" + "="*80)
print("STEP 7: After postprocessor (unnormalization)")
print("="*80)
action_post = postprocessor(action)
print_tensor_stats("action unnormalized", action_post, verbose=args.verbose)

# Validate postprocessed action
print("\n" + "="*80)
print("ACTION VALIDATION")
print("="*80)

action_np = action_post.float().cpu().numpy()
if action_np.ndim == 2:
    action_np = action_np[0]  # Remove batch if present

print(f"Action shape: {action_np.shape}")
print(f"Action values: {action_np}")

# Check for NaN/Inf
if np.isnan(action_np).any() or np.isinf(action_np).any():
    print("❌ FAILURE: Action contains NaN or Inf!")
    sys.exit(1)
else:
    print("✓ Action is finite")

# Check magnitude (rough sanity check)
action_mag = np.linalg.norm(action_np[:6])  # Position components
gripper = action_np[6] if len(action_np) > 6 else 0
print(f"  Position magnitude: {action_mag:.4f}")
print(f"  Gripper value: {gripper:.4f}")

if action_mag > 10 or action_mag < -10:
    print("  ⚠️  Warning: Position action magnitude seems extreme")

env.close()
print("\n" + "="*80)
print("✓ All checks passed. Pipeline is healthy.")
print("="*80)
