import json
import os
import argparse

import numpy as np
import torch
from tqdm import tqdm

from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs import libero as libero_module
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.scripts.lerobot_eval import eval_one


TASK_FILE = "configs/tasks.json"
DATA_PATH = "data/libero"
DEFAULT_MODEL_PATH = "models/smolvla"
DEFAULT_RESULT_PATH = "results/baseline_results.json"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
EPISODES_PER_TASK = 20


def patch_libero_action_dim():
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


def parse_task_name(task_name):
	if "_task_" not in task_name:
		raise ValueError(f"Unexpected task format: {task_name}")
	suite_name, task_id = task_name.rsplit("_task_", 1)
	return suite_name, int(task_id)


def load_model(model_path):
	model = SmolVLAPolicy.from_pretrained(model_path)
	model.to(torch.device(DEVICE))
	model.eval()
	return model


def build_rename_map(model):
	feature_keys = set(model.config.input_features.keys())
	rename_map = {}

	# Common LIBERO env outputs are image and image2; map them to policy-specific names.
	if "observation.images.camera1" in feature_keys:
		rename_map["observation.images.image"] = "observation.images.camera1"
	if "observation.images.camera2" in feature_keys:
		rename_map["observation.images.image2"] = "observation.images.camera2"
	if "observation.images.wrist_image" in feature_keys:
		rename_map["observation.images.image2"] = "observation.images.wrist_image"

	# Some tuned checkpoints use observation.images.image directly.
	if "observation.images.image" in feature_keys:
		rename_map.pop("observation.images.image", None)

	return rename_map


def run_task(model, preprocessor, postprocessor, task_name, debug=False, debug_steps=5):
	suite_name, task_id = parse_task_name(task_name)

	env_cfg = LiberoEnvConfig(task=suite_name, task_ids=[task_id], control_mode="relative")
	envs = make_env(cfg=env_cfg, n_envs=1, use_async_envs=False, trust_remote_code=False)
	env = envs[suite_name][task_id]

	env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=model.config)

	if debug:
		print(f"  [DEBUG] {task_name}: Model image keys={set(model.config.image_features.keys())}")

	metrics = eval_one(
		env,
		policy=model,
		env_preprocessor=env_preprocessor,
		env_postprocessor=env_postprocessor,
		preprocessor=preprocessor,
		postprocessor=postprocessor,
		n_episodes=EPISODES_PER_TASK,
		max_episodes_rendered=0,
		videos_dir=None,
		return_episode_data=False,
		start_seed=0,
	)

	env.close()
	successes = [int(s) for s in metrics["successes"]]
	return float(np.mean(successes))


def parse_args():
	parser = argparse.ArgumentParser(description="Run SmolVLA LIBERO evaluation")
	parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to local SmolVLA checkpoint")
	parser.add_argument("--result-path", default=DEFAULT_RESULT_PATH, help="Path to write JSON results")
	parser.add_argument("--debug", action="store_true", help="Print per-task debug info (first N steps)")
	parser.add_argument("--debug-steps", type=int, default=5, help="Number of steps to debug per task")
	parser.add_argument("--smoke", action="store_true", help="Run only first task with fewer episodes (smoke test)")
	parser.add_argument("--smoke-episodes", type=int, default=3, help="Episodes for smoke test")
	return parser.parse_args()


def main():
	args = parse_args()
	model_path = args.model_path
	result_path = args.result_path

	os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
	patch_libero_action_dim()

	with open(TASK_FILE) as f:
		tasks = json.load(f)

	# If smoke test, only run first task
	if args.smoke:
		tasks = tasks[:1]
		episodes_per_task = args.smoke_episodes
		print(f"🔥 SMOKE TEST MODE: Running {len(tasks)} task(s) with {episodes_per_task} episodes each")
	else:
		episodes_per_task = EPISODES_PER_TASK

	model = load_model(model_path)
	rename_map = build_rename_map(model)
	
	if args.debug:
		print(f"\n[DEBUG MODE] Rename map: {rename_map}")
		print(f"[DEBUG MODE] Model input features: {sorted(model.config.input_features.keys())}")
		print(f"[DEBUG MODE] Model image features: {sorted(model.config.image_features.keys())}\n")
	
	preprocessor, postprocessor = make_pre_post_processors(
		policy_cfg=model.config,
		pretrained_path=model_path,
		preprocessor_overrides={
			"device_processor": {"device": DEVICE},
			"rename_observations_processor": {
				"rename_map": rename_map
			},
		},
	)

	results = {}
	
	# Temporarily override EPISODES_PER_TASK for this run
	original_episodes = EPISODES_PER_TASK
	globals()["EPISODES_PER_TASK"] = episodes_per_task

	for task in tqdm(tasks, desc="Tasks"):
		print("Running task:", task)
		sr = run_task(model, preprocessor, postprocessor, task, debug=args.debug, debug_steps=args.debug_steps)
		results[task] = float(sr)

	# Restore original
	globals()["EPISODES_PER_TASK"] = original_episodes

	result_path_obj = os.path.dirname(result_path)
	if result_path_obj and not os.path.exists(result_path_obj):
		os.makedirs(result_path_obj, exist_ok=True)

	with open(result_path, "w") as f:
		json.dump(results, f, indent=2)

	print("\n===== EVALUATION RESULTS =====")
	for k, v in results.items():
		status = "✓" if v > 0 else "✗"
		print(f"{k:40} | success_rate: {v:.3f} {status}")
	
	# Print summary
	mean_sr = np.mean(list(results.values()))
	print(f"\nMean success rate: {mean_sr:.3f}")
	
	if args.smoke:
		print("\n🔥 Smoke test complete. Use --smoke=false to run full evaluation.")


if __name__ == "__main__":
	main()

