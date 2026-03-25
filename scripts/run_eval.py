import json
import os
import sys
import argparse
import time
import uuid
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

import imageio
import numpy as np
import torch
from tqdm import tqdm

from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import add_envs_task, preprocess_observation
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.scripts.lerobot_eval import eval_one
from lerobot.utils.constants import ACTION
from lerobot.utils.random_utils import set_seed

TASK_FILE = "configs/tasks.json"
OFFICIAL_MODEL_ID = "HuggingFaceVLA/smolvla_libero"
DEFAULT_RESULT_PATH = "results/official_eval_results.json"
DEFAULT_FRAMES_ROOT = "results/headless_frame_dumps"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
EPISODES_PER_TASK = 4
SMOKE_EPISODES = 1
N_ENVS_PER_TASK = 4
USE_ASYNC_ENVS = False
# Set to "0" in your shell to test whether MPS fallback is hiding slow CPU fallbacks.
# Example: PYTORCH_ENABLE_MPS_FALLBACK=0 python ./scripts/run_eval.py --smoke
MPS_FALLBACK = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "1")
DEFAULT_SEED = 0
DETERMINISTIC_MODE = True


POLICY_REGISTRY = {
	"Huggingface SmolVLA Libero": {
		"build_model": lambda: SmolVLAPolicy.from_pretrained(OFFICIAL_MODEL_ID),
	},
}


def build_policy_choices():
	"""Build selectable policies from hardcoded registry."""
	choices = []
	for name, entry in POLICY_REGISTRY.items():
		choices.append(
			{
				"id": name,
				"label": name,
				"build_model": entry["build_model"],
			}
		)
	return choices


def _print_policy_choices(choices):
	print("\nAvailable policies:")
	for idx, choice in enumerate(choices, start=1):
		print(f"  [{idx}] {choice['label']}")


def _prompt_for_policy_choice(choices):
	while True:
		selection = input("Select policy number (Enter for 1): ").strip()
		if selection == "":
			return choices[0]
		if selection.isdigit():
			idx = int(selection)
			if 1 <= idx <= len(choices):
				return choices[idx - 1]
		print(f"Invalid selection: {selection}. Enter a number from 1 to {len(choices)}.")


def select_policy_runtime(args):
	"""Resolve policy builder from args and optional user prompts."""
	choices = build_policy_choices()

	if args.list_policies:
		_print_policy_choices(choices)
		sys.exit(0)

	if args.policy:
		for choice in choices:
			if args.policy == choice["id"]:
				return choice["build_model"], choice["label"]

		valid_names = ", ".join(choice["id"] for choice in choices)
		raise ValueError(f"Unknown --policy '{args.policy}'. Valid options: {valid_names}")

	if sys.stdin.isatty():
		_print_policy_choices(choices)
		selected = _prompt_for_policy_choice(choices)
		return selected["build_model"], selected["label"]

	default_choice = choices[0]
	return default_choice["build_model"], default_choice["label"]


def emit_event(event, run_id, **data):
	entry = {
		"ts": time.time(),
		"event": event,
		"run_id": run_id,
	}
	entry.update(data)
	print(json.dumps(entry, separators=(",", ":")), flush=True)


def parse_task_name(task_name):
	if "_task_" not in task_name:
		raise ValueError(f"Unexpected task format: {task_name}")
	suite_name, task_id = task_name.rsplit("_task_", 1)
	return suite_name, int(task_id)


def load_model(build_model):
	model = build_model()
	model.to(torch.device(DEVICE))
	model.eval()
	return model


def configure_reproducibility(seed: int, deterministic: bool) -> None:
	"""Configure Python/NumPy/Torch RNG and deterministic algorithm behavior."""
	set_seed(seed)
	random.seed(seed)

	if deterministic:
		# warn_only avoids hard failures for unsupported deterministic ops on some backends.
		torch.use_deterministic_algorithms(True, warn_only=True)
		if torch.cuda.is_available():
			torch.backends.cudnn.deterministic = True
			torch.backends.cudnn.benchmark = False
	else:
		torch.use_deterministic_algorithms(False)
		if torch.cuda.is_available():
			torch.backends.cudnn.deterministic = False
			torch.backends.cudnn.benchmark = True


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


def extract_render_frame(env):
	"""Safely extract a rendered RGB frame from the vectorized env."""
	try:
		frames = env.call("render")
		if isinstance(frames, (list, tuple)) and frames:
			frame = frames[0]
			if frame is not None:
				return frame
		if isinstance(frames, np.ndarray):
			return frames
	except Exception as exc:
		print(f"Warning: failed to extract render frame: {exc}")
		return None
	return None


def normalize_frame_for_saving(frame):
	"""Normalize a frame into uint8 HWC RGB for PNG saving."""
	if isinstance(frame, torch.Tensor):
		frame = frame.detach().cpu().numpy()
	if not isinstance(frame, np.ndarray):
		return None

	while frame.ndim >= 4:
		frame = frame[0]

	if frame.ndim == 2:
		frame = np.repeat(frame[..., None], 3, axis=2)
	elif frame.ndim == 3 and frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
		frame = np.transpose(frame, (1, 2, 0))

	if frame.ndim != 3:
		return None

	if frame.shape[2] == 1:
		frame = np.repeat(frame, 3, axis=2)
	elif frame.shape[2] == 4:
		frame = frame[:, :, :3]

	if frame.dtype != np.uint8:
		frame = frame.astype(np.float32)
		if frame.max() <= 1.0:
			frame *= 255.0
		frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)

	return frame


def extract_preview_from_observation(observation):
	"""Fallback frame source from raw LIBERO observations."""
	try:
		pixels = observation.get("pixels")
		if isinstance(pixels, dict):
			for key in ("image", "image2"):
				frame = pixels.get(key)
				normalized = normalize_frame_for_saving(frame)
				if normalized is not None:
					return normalized
	except Exception as exc:
		print(f"Warning: failed to extract preview frame: {exc}")
		return None
	return None


def maybe_save_frame(frame, episode_dir, frame_index):
	normalized = normalize_frame_for_saving(frame)
	if normalized is None:
		return False
	frame_path = episode_dir / f"frame_{frame_index:07d}.png"
	imageio.imwrite(frame_path, normalized)
	return True


def run_task(model, preprocessor, postprocessor, task_name, n_episodes):
	build_start = time.time()
	suite_name, task_id = parse_task_name(task_name)

	env_cfg = LiberoEnvConfig(task=suite_name, task_ids=[task_id], control_mode="relative")
	envs = make_env(
		cfg=env_cfg,
		n_envs=N_ENVS_PER_TASK,
		use_async_envs=USE_ASYNC_ENVS,
		trust_remote_code=False,
	)
	env = envs[suite_name][task_id]

	env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=model.config)
	build_s = time.time() - build_start

	eval_start = time.time()
	metrics = eval_one(
		env,
		policy=model,
		env_preprocessor=env_preprocessor,
		env_postprocessor=env_postprocessor,
		preprocessor=preprocessor,
		postprocessor=postprocessor,
		n_episodes=n_episodes,
		max_episodes_rendered=0,
		videos_dir=None,
		return_episode_data=False,
		start_seed=0,
	)
	eval_s = time.time() - eval_start

	env.close()
	successes = [int(s) for s in metrics["successes"]]
	return float(np.mean(successes)), build_s, eval_s


def run_task_with_frames(
	model,
	preprocessor,
	postprocessor,
	task_name,
	n_episodes,
	frames_dir,
	render_every_n_steps,
	run_id,
):
	build_start = time.time()
	suite_name, task_id = parse_task_name(task_name)

	env_cfg = LiberoEnvConfig(task=suite_name, task_ids=[task_id], control_mode="relative")
	envs = make_env(
		cfg=env_cfg,
		n_envs=1,
		use_async_envs=False,
		trust_remote_code=False,
	)
	env = envs[suite_name][task_id]

	env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=model.config)
	build_s = time.time() - build_start

	task_frames_dir = Path(frames_dir) / run_id / task_name
	task_frames_dir.mkdir(parents=True, exist_ok=True)

	eval_start = time.time()
	success_count = 0
	saved_frames = 0

	try:
		for ep in range(n_episodes):
			model.reset()
			observation, info = env.reset(seed=[ep])
			episode_dir = task_frames_dir / f"episode_{ep + 1:03d}"
			episode_dir.mkdir(parents=True, exist_ok=True)

			frame_index = 0
			if 0 % render_every_n_steps == 0:
				frame = extract_render_frame(env)
				if frame is None:
					frame = extract_preview_from_observation(observation)
				if frame is not None and maybe_save_frame(frame, episode_dir, frame_index):
					saved_frames += 1
					frame_index += 1

			done = np.array([False])
			max_steps = env.call("_max_episode_steps")[0]
			step = 0

			while (not bool(done[0])) and step < max_steps:
				obs = preprocess_observation(observation)
				obs = add_envs_task(env, obs)
				obs = env_preprocessor(obs)
				obs = preprocessor(obs)

				with torch.inference_mode():
					action = model.select_action(obs)

				action = postprocessor(action)
				action_transition = {ACTION: action}
				action_transition = env_postprocessor(action_transition)
				action = action_transition[ACTION]
				action_np = action.to("cpu").numpy()

				observation, _reward, terminated, truncated, info = env.step(action_np)
				done = terminated | truncated | done
				step += 1

				if step % render_every_n_steps == 0:
					frame = extract_render_frame(env)
					if frame is None:
						frame = extract_preview_from_observation(observation)
					if frame is not None and maybe_save_frame(frame, episode_dir, frame_index):
						saved_frames += 1
						frame_index += 1

				if "final_info" in info and info["final_info"]["is_success"][0]:
					success_count += 1
					break
	finally:
		env.close()

	eval_s = time.time() - eval_start
	return success_count / float(n_episodes), build_s, eval_s, saved_frames


def parse_args():
	parser = argparse.ArgumentParser(
		description="Run SmolVLA LIBERO evaluation from the terminal.",
		epilog=(
			"Examples:\n"
			"  python scripts/run_eval.py                      # interactive policy prompt\n"
			"  python scripts/run_eval.py --list-policies\n"
			"  python scripts/run_eval.py --policy MyPolicy --smoke\n"
			"  python scripts/run_eval.py --policy Official SmolVLA --smoke\n"
			"  python scripts/run_eval.py --smoke\n"
			"  python scripts/run_eval.py --save-frames\n"
			"  python scripts/run_eval.py --save-frames 10"
		),
		formatter_class=argparse.RawTextHelpFormatter,
	)
	parser.add_argument("--smoke", action="store_true", help="Run only first task with fewer episodes")
	parser.add_argument(
		"--policy",
		type=str,
		default=None,
		help=(
			"Policy name from hardcoded POLICY_REGISTRY (e.g., 'MyPolicy', 'Official SmolVLA'). "
			"If omitted, you will be prompted at runtime in interactive terminals."
		),
	)
	parser.add_argument(
		"--list-policies",
		action="store_true",
		help="Print configured policy options and exit.",
	)
	parser.add_argument(
		"--save-frames",
		nargs="?",
		type=int,
		const=5,
		default=0,
		metavar="N",
		help=(
			"Enable headless frame dumping. If provided without a value, saves one frame every 5 steps. "
			"You can override with N, e.g. --save-frames 10."
		),
	)
	return parser.parse_args()


def main():
	args = parse_args()
	if args.save_frames < 0:
		raise ValueError("--save-frames expects a positive integer when a value is provided")

	save_frames_enabled = args.save_frames > 0
	render_every_n_steps = args.save_frames if save_frames_enabled else 0

	run_id = str(uuid.uuid4())
	configure_reproducibility(seed=DEFAULT_SEED, deterministic=DETERMINISTIC_MODE)
	build_model, policy_label = select_policy_runtime(args)
	model_path = OFFICIAL_MODEL_ID
	result_path = DEFAULT_RESULT_PATH

	os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = MPS_FALLBACK

	with open(TASK_FILE) as f:
		tasks = json.load(f)

	# If smoke test, only run first task
	if args.smoke:
		tasks = tasks[:1]
		episodes_per_task = SMOKE_EPISODES
		print(f"🔥 SMOKE TEST MODE: Running {len(tasks)} task(s) with {episodes_per_task} episodes each")
	else:
		episodes_per_task = EPISODES_PER_TASK

	emit_event(
		"run_start",
		run_id,
		mode="smoke" if args.smoke else "full",
		tasks_total=len(tasks),
		episodes_per_task=episodes_per_task,
		device=DEVICE,
		n_envs=(1 if save_frames_enabled else N_ENVS_PER_TASK),
		use_async_envs=USE_ASYNC_ENVS,
		mps_fallback=MPS_FALLBACK,
		save_frames=save_frames_enabled,
		render_every_n_steps=int(render_every_n_steps),
		frames_root=DEFAULT_FRAMES_ROOT,
		seed=int(DEFAULT_SEED),
		deterministic=bool(DETERMINISTIC_MODE),
		policy=policy_label,
		model_path=model_path,
	)

	print(f"Selected policy: {policy_label}")
	print(f"Selected checkpoint: {model_path}")

	model = load_model(build_model)
	rename_map = build_rename_map(model)
	
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
	total_build_s = 0.0
	total_eval_s = 0.0
	total_saved_frames = 0
	tasks_completed = 0

	n_envs_runtime = 1 if save_frames_enabled else N_ENVS_PER_TASK
	print(
		f"Eval config: device={DEVICE}, n_envs={n_envs_runtime}, "
		f"use_async_envs={USE_ASYNC_ENVS}, mps_fallback={MPS_FALLBACK}, "
		f"seed={DEFAULT_SEED}, deterministic={DETERMINISTIC_MODE}, "
		f"episodes_per_task={episodes_per_task}, save_frames={save_frames_enabled}, "
		f"render_every_n_steps={render_every_n_steps if save_frames_enabled else 'off'}"
	)
	if save_frames_enabled:
		print(f"Frame dump directory: {Path(DEFAULT_FRAMES_ROOT) / run_id}")

	with torch.inference_mode():
		for task_idx, task in enumerate(tqdm(tasks, desc="Tasks"), start=1):
			print("Running task:", task)
			emit_event("task_start", run_id, task=task, task_index=task_idx, tasks_total=len(tasks))
			if save_frames_enabled:
				sr, build_s, eval_s, saved_frames = run_task_with_frames(
					model,
					preprocessor,
					postprocessor,
					task,
					n_episodes=episodes_per_task,
					frames_dir=DEFAULT_FRAMES_ROOT,
					render_every_n_steps=render_every_n_steps,
					run_id=run_id,
				)
				total_saved_frames += saved_frames
			else:
				sr, build_s, eval_s = run_task(
					model,
					preprocessor,
					postprocessor,
					task,
					n_episodes=episodes_per_task,
				)
				saved_frames = 0
			total_build_s += build_s
			total_eval_s += eval_s
			tasks_completed += 1
			print(f"  timing: env_build={build_s:.2f}s eval={eval_s:.2f}s")
			if save_frames_enabled:
				print(f"  frames_saved: {saved_frames}")
			results[task] = float(sr)
			emit_event(
				"task_end",
				run_id,
				task=task,
				success_rate=float(sr),
				build_s=float(build_s),
				eval_s=float(eval_s),
				tasks_completed=tasks_completed,
				running_mean_sr=float(np.mean(list(results.values()))),
			)

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
	print(f"Total env build time: {total_build_s:.2f}s")
	print(f"Total eval time: {total_eval_s:.2f}s")
	if save_frames_enabled:
		print(f"Total saved frames: {total_saved_frames}")
		print(f"Saved frame root: {Path(DEFAULT_FRAMES_ROOT) / run_id}")
	emit_event(
		"run_end",
		run_id,
		mean_success_rate=float(mean_sr),
		tasks_completed=tasks_completed,
		total_build_s=float(total_build_s),
		total_eval_s=float(total_eval_s),
		total_saved_frames=total_saved_frames,
	)
	
	if args.smoke:
		print("\n🔥 Smoke test complete. Use --smoke=false to run full evaluation.")


if __name__ == "__main__":
	main()

