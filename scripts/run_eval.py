import json
import os
import sys
import argparse
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

import imageio
import numpy as np
import torch
from tqdm import tqdm

from scripts.common import (
	DEFAULT_EPISODES_PER_TASK,
	DEFAULT_FRAMES_ROOT,
	DEFAULT_N_ENVS_PER_TASK,
	DEFAULT_SMOKE_EPISODES,
	DEFAULT_TASK_FILE,
	OFFICIAL_MODEL_ID,
	POLICY_REGISTRY,
	RUNTIME_DETERMINISTIC,
	RUNTIME_MPS_FALLBACK,
	RUNTIME_SEED,
	build_default_result_path,
	configure_reproducibility,
	emit_event,
	extract_preview_from_observation,
	extract_render_frame,
	load_policy_and_processors,
	normalize_frame_uint8,
	resolve_project_path,
	resolve_runtime_device,
	run_task_stepwise,
	run_task_vectorized,
)

DEFAULT_RESULT_PATH = ""


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
		continue


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

	valid_names = ", ".join(choice["id"] for choice in choices)
	raise ValueError(
		"--policy is required in non-interactive environments. "
		f"Valid options: {valid_names}"
	)
def maybe_save_frame(frame, episode_dir, frame_index):
	normalized = normalize_frame_uint8(frame)
	if normalized is None:
		return False
	frame_path = episode_dir / f"frame_{frame_index:07d}.png"
	imageio.imwrite(frame_path, normalized)
	return True


def run_task_with_frames(
	model,
	preprocessor,
	postprocessor,
	task_name,
	n_episodes,
	frames_dir,
	render_every_n_steps,
	run_id,
	start_seed,
	event_log_path,
):
	task_frames_dir = Path(frames_dir) / run_id / task_name
	task_frames_dir.mkdir(parents=True, exist_ok=True)
	saved_frames = 0
	frame_indices: dict[int, int] = {}

	def on_episode_reset(_env, _observation, _info, episode_index: int):
		episode_dir = task_frames_dir / f"episode_{episode_index + 1:03d}"
		episode_dir.mkdir(parents=True, exist_ok=True)
		frame_indices[episode_index] = 0

	def on_frame(env, observation, _info, step: int, episode_index: int):
		nonlocal saved_frames
		frame = extract_render_frame(env)
		if frame is None:
			frame = extract_preview_from_observation(observation)
		episode_dir = task_frames_dir / f"episode_{episode_index + 1:03d}"
		frame_index = frame_indices.get(episode_index, 0)
		if frame is not None and maybe_save_frame(frame, episode_dir, frame_index):
			saved_frames += 1
			frame_indices[episode_index] = frame_index + 1

	def on_step(step: int, max_steps: int, _info, episode_index: int):
		del _info
		if step % 10 == 0:
			emit_event(
				"step_update",
				run_id,
				event_log_path=event_log_path,
				task=task_name,
				episode=episode_index + 1,
				step=step,
				max_steps=max_steps,
			)

	sr, build_s, eval_s = run_task_stepwise(
		model=model,
		preprocessor=preprocessor,
		postprocessor=postprocessor,
		task_name=task_name,
		n_episodes=n_episodes,
		start_seed=start_seed,
		render_every_n_steps=render_every_n_steps,
		on_frame=on_frame,
		on_step=on_step,
		on_episode_reset=on_episode_reset,
	)

	return sr, build_s, eval_s, saved_frames


def parse_args():
	parser = argparse.ArgumentParser(
		description="Run SmolVLA LIBERO evaluation from the terminal.",
		epilog=(
			"Examples:\n"
			"  python scripts/run_eval.py                      # interactive policy prompt\n"
			"  python scripts/run_eval.py --list-policies\n"
			"  python scripts/run_eval.py --policy 'MyPolicy FlowOnly' --smoke\n"
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
			"Policy name from hardcoded POLICY_REGISTRY (e.g., 'MyPolicy FlowOnly'). "
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
	parser.add_argument(
		"--episodes-per-task",
		type=int,
		default=int(os.environ.get("EPISODES_PER_TASK", DEFAULT_EPISODES_PER_TASK)),
		help="Episodes per task in normal mode.",
	)
	parser.add_argument(
		"--n-envs",
		type=int,
		default=int(os.environ.get("N_ENVS", DEFAULT_N_ENVS_PER_TASK)),
		help="Number of vectorized envs for non-frame-dump runs.",
	)
	parser.add_argument(
		"--task-file",
		type=str,
		default=DEFAULT_TASK_FILE,
		help="Task list JSON path (relative paths are resolved from project root).",
	)
	return parser.parse_args()


def main():
	args = parse_args()

	if args.save_frames < 0:
		raise ValueError("--save-frames expects a positive integer when a value is provided")
	if args.episodes_per_task <= 0:
		raise ValueError("--episodes-per-task must be a positive integer")
	if args.n_envs <= 0:
		raise ValueError("--n-envs must be a positive integer")

	save_frames_enabled = args.save_frames > 0
	render_every_n_steps = args.save_frames if save_frames_enabled else 0
	project_root = PROJECT_ROOT
	device = resolve_runtime_device()
	seed = int(RUNTIME_SEED)
	deterministic = bool(RUNTIME_DETERMINISTIC)
	task_file = resolve_project_path(args.task_file, project_root)
	frames_root = resolve_project_path(DEFAULT_FRAMES_ROOT, project_root)
	n_envs_runtime = 1 if save_frames_enabled else int(args.n_envs)

	run_id = str(uuid.uuid4())
	configure_reproducibility(seed=seed, deterministic=deterministic)
	_build_model, policy_label = select_policy_runtime(args)
	model_path = OFFICIAL_MODEL_ID
	if str(DEFAULT_RESULT_PATH).strip():
		result_path = resolve_project_path(DEFAULT_RESULT_PATH, project_root)
	else:
		result_path = build_default_result_path(project_root, policy_label, task_file, prefix="official_eval")
	event_log_path = result_path.with_suffix(".jsonl")
	model, preprocessor, postprocessor, rename_map, router_metadata = load_policy_and_processors(policy_label, device)

	os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = RUNTIME_MPS_FALLBACK

	with task_file.open() as f:
		tasks = json.load(f)

	# If smoke test, only run first task
	if args.smoke:
		tasks = tasks[:1]
		episodes_per_task = DEFAULT_SMOKE_EPISODES
	else:
		episodes_per_task = int(args.episodes_per_task)

	mode = "smoke" if args.smoke else "full"
	base_event_fields = {
		"mode": mode,
		"policy": policy_label,
		"task_file": str(task_file),
		"device": device,
		"n_envs": n_envs_runtime,
		"episodes_per_task": episodes_per_task,
		"seed": seed,
	}

	emit_event(
		"run_start",
		run_id,
		event_log_path=event_log_path,
		**base_event_fields,
		tasks_total=len(tasks),
		mps_fallback=RUNTIME_MPS_FALLBACK,
		save_frames=save_frames_enabled,
		render_every_n_steps=int(render_every_n_steps),
		frames_root=str(frames_root),
		model_path=model_path,
		results_file=str(result_path),
		**router_metadata,
	)

	results = {}
	total_build_s = 0.0
	total_eval_s = 0.0
	total_saved_frames = 0
	tasks_completed = 0

	with torch.inference_mode():
		for task_idx, task in enumerate(tqdm(tasks, desc="Tasks", disable=not sys.stdout.isatty()), start=1):
			if save_frames_enabled:
				sr, build_s, eval_s, saved_frames = run_task_with_frames(
					model,
					preprocessor,
					postprocessor,
					task,
					n_episodes=episodes_per_task,
					frames_dir=frames_root,
					render_every_n_steps=render_every_n_steps,
					run_id=run_id,
					start_seed=seed,
					event_log_path=event_log_path,
				)
				total_saved_frames += saved_frames
			else:
				sr, build_s, eval_s = run_task_vectorized(
					model,
					preprocessor,
					postprocessor,
					task,
					n_episodes=episodes_per_task,
					n_envs=n_envs_runtime,
					use_async_envs=False,
					start_seed=seed,
				)
				saved_frames = 0
			total_build_s += build_s
			total_eval_s += eval_s
			tasks_completed += 1
			results[task] = float(sr)
			router_task_metadata = {}
			if hasattr(model, "model") and hasattr(model.model, "last_router_decision"):
				router_task_metadata = dict(model.model.last_router_decision)
			emit_event(
				"task_end",
				run_id,
				event_log_path=event_log_path,
				**base_event_fields,
				task=task,
				task_index=task_idx,
				tasks_total=len(tasks),
				success_rate=float(sr),
				build_s=float(build_s),
				eval_s=float(eval_s),
				tasks_completed=tasks_completed,
				running_mean_sr=float(np.mean(list(results.values()))),
				**router_task_metadata,
			)

	result_path.parent.mkdir(parents=True, exist_ok=True)

	with result_path.open("w") as f:
		json.dump(results, f, indent=2)

	# Print summary
	mean_sr = np.mean(list(results.values()))
	emit_event(
		"run_end",
		run_id,
		event_log_path=event_log_path,
		**base_event_fields,
		mean_success_rate=float(mean_sr),
		tasks_completed=tasks_completed,
		tasks_total=len(tasks),
		total_build_s=float(total_build_s),
		total_eval_s=float(total_eval_s),
		total_saved_frames=total_saved_frames,
		results_file=str(result_path),
	)


if __name__ == "__main__":
	main()

