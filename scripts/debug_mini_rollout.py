"""
Mini rollout diagnostic: run 100 steps with the tuned model, print action/reward/done trace.
Usage: python scripts/debug_mini_rollout.py [--steps N] [--verbose]
With per-step action statistics and success/termination diagnostics.
"""
import os, sys
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, "external/lerobot/src")

import argparse
import numpy as np
import torch
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs import libero as libero_module
from lerobot.envs.utils import preprocess_observation, add_envs_task
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

MODEL_PATH = "models/smolvla_libero_tuned_migrated"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=100, help="Max steps per episode")
parser.add_argument("--verbose", action="store_true", help="Print full action values")
args = parser.parse_args()

# Patch just in case (tuned model is 7-dim so this is a no-op)
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

print("Loading model...")
model = SmolVLAPolicy.from_pretrained(MODEL_PATH)
model.to(torch.device(DEVICE)).eval()
print(f"✓ Model loaded. Device={DEVICE}")

feature_keys = set(model.config.input_features.keys())
rename_map = {}
if "observation.images.wrist_image" in feature_keys:
    rename_map["observation.images.image2"] = "observation.images.wrist_image"
if "observation.images.image" in feature_keys:
    rename_map.pop("observation.images.image", None)
print(f"Rename map: {rename_map}")

preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=model.config,
    pretrained_path=MODEL_PATH,
    preprocessor_overrides={
        "device_processor": {"device": DEVICE},
        "rename_observations_processor": {"rename_map": rename_map},
    },
)

env_cfg = LiberoEnvConfig(task="libero_spatial", task_ids=[0], control_mode="relative")
envs = make_env(cfg=env_cfg, n_envs=1, use_async_envs=False, trust_remote_code=False)
env = envs["libero_spatial"][0]
env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=model.config)

print("\n" + "="*100)
print("MINI ROLLOUT: Running episode with tuned model")
print("="*100)

model.reset()
obs, info = env.reset(seed=[42])
done = np.array([False])
step = 0
max_steps = env.call("_max_episode_steps")[0]
print(f"Task max_steps: {max_steps}")
print(f"Target steps for diagnostic: {args.steps}")
print()

rewards = []
action_mags = []
success_flag = False

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
    action_tensor = action_transition["action"]
    action_np = action_tensor.to("cpu").numpy()
    
    obs, reward, terminated, truncated, info = env.step(action_np)
    done = terminated | truncated
    
    # Diagnostics
    rewards.append(float(reward[0]))
    if action_np.ndim == 2:
        action_vec = action_np[0]
    else:
        action_vec = action_np
    
    action_mag = np.linalg.norm(action_vec[:6])
    action_mags.append(action_mag)
    
    # Check if success was achieved
    if "final_info" in info and info["final_info"].get("is_success", [False])[0]:
        success_flag = True
    
    # Detailed output for first 5 steps, sparse afterwards, and any non-zero reward
    if step < 5 or step % 20 == 0 or reward[0] > 0 or success_flag:
        line = f"  step={step:3d} reward={reward[0]:+.3f} done={int(done[0])} "
        line += f"action=[{action_vec[0]:+.4f},{action_vec[1]:+.4f},{action_vec[2]:+.4f},"
        line += f"{action_vec[3]:+.4f},{action_vec[4]:+.4f},{action_vec[5]:+.4f},"
        line += f"{action_vec[6]:+.4f}]"
        if args.verbose:
            line += f" mag={action_mag:.4f}"
        if reward[0] > 0:
            line += " ⭐ REWARD"
        if success_flag:
            line += " 🎯 SUCCESS"
        print(line)
    
    step += 1

print()
print("="*100)
print("EPISODE SUMMARY")
print("="*100)
print(f"Total steps: {step} / {min(max_steps, args.steps)}")
print(f"Episode completed: {bool(done[0])}")
print(f"Max reward: {max(rewards):.4f}")
print(f"Sum rewards: {sum(rewards):.4f}")
print(f"Mean reward: {np.mean(rewards):.6f}")
print(f"Reward freq (>0): {sum(1 for r in rewards if r > 0)} / {len(rewards)}")

print(f"\nAction statistics:")
print(f"  Pos magnitude - min={min(action_mags):.4f} max={max(action_mags):.4f} mean={np.mean(action_mags):.4f}")

if success_flag:
    print("\n🎯 ✓ Task completed successfully!")
else:
    print("\n❌ Task did not complete successfully.")
    if max(rewards) == 0:
        print("  (No positive rewards received - possible pipeline issue)")
    else:
        print(f"  (Achieved max reward {max(rewards):.4f} but target may be higher)")

print("="*100)
env.close()
