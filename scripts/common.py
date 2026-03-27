from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import add_envs_task, preprocess_observation
from lerobot.policies.factory import make_pre_post_processors
from lerobot.scripts.lerobot_eval import eval_one
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import ACTION
from lerobot.utils.random_utils import set_seed
from my_policies import build_mypolicy_binning_only, build_mypolicy_flow_only

OFFICIAL_MODEL_ID = "HuggingFaceVLA/smolvla_libero"
DEFAULT_TASK_FILE = "configs/tasks.json"
DEFAULT_FRAMES_ROOT = "results/headless_frame_dumps"
DEFAULT_EPISODES_PER_TASK = 10
DEFAULT_SMOKE_EPISODES = 1
DEFAULT_N_ENVS_PER_TASK = 4
RUNTIME_SEED = 0
RUNTIME_DETERMINISTIC = True
RUNTIME_MPS_FALLBACK = "1"


def resolve_runtime_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"

    if not sys.stdin.isatty():
        raise RuntimeError(
            "No CUDA/MPS device available. CPU fallback requires interactive confirmation. "
            "Non-interactive runs fail fast by design."
        )

    answer = input("No CUDA/MPS device found. Continue on CPU? [y/N]: ").strip().lower()
    if answer in {"y", "yes"}:
        return "cpu"

    raise RuntimeError("CPU fallback not confirmed. Aborting run.")


def _to_json_serializable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    return value


def build_policy_registry() -> dict[str, dict[str, object]]:
    return {
        "HF": {
            "build_model": lambda: SmolVLAPolicy.from_pretrained(OFFICIAL_MODEL_ID),
        },
        "FlowOnly": {
            "build_model": build_mypolicy_flow_only,
        },
        "BinOnly": {
            "build_model": build_mypolicy_binning_only,
        },
    }


POLICY_REGISTRY = build_policy_registry()


def resolve_project_path(path_value: str, project_root: Path) -> Path:
    path_obj = Path(path_value)
    if path_obj.is_absolute():
        return path_obj
    return project_root / path_obj


def _sanitize_name_for_path(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_")
    return cleaned.lower() or "default"


def get_result_group_dir(project_root: Path, policy_name: str, task_file: Path) -> Path:
    policy_dir = _sanitize_name_for_path(policy_name)
    task_stem = _sanitize_name_for_path(task_file.stem)
    return project_root / "results" / policy_dir / task_stem


def build_default_result_path(
    project_root: Path,
    policy_name: str,
    task_file: Path,
    *,
    prefix: str = "eval",
) -> Path:
    group_dir = get_result_group_dir(project_root, policy_name, task_file)
    return group_dir / f"{prefix}.json"


def emit_event(
    event: str,
    run_id: str | None = None,
    *,
    event_log_path: str | Path | None = None,
    **data,
) -> None:
    payload = {
        "event": event,
        "ts": time.time(),
    }
    if run_id is not None:
        payload["run_id"] = run_id
    payload.update(data)
    normalized = {key: _to_json_serializable(value) for key, value in payload.items()}
    line = json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str)

    if event_log_path is not None:
        log_path = Path(event_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.write("\n")

    summary = f"[{event}]"
    if run_id is not None:
        summary += f" run_id={run_id}"
    if "task" in normalized:
        summary += f" task={normalized['task']}"
    if "success_rate" in normalized:
        summary += f" success_rate={normalized['success_rate']}"
    if "mean_success_rate" in normalized:
        summary += f" mean_success_rate={normalized['mean_success_rate']}"
    if "status" in normalized:
        summary += f" status={normalized['status']}"
    print(summary, flush=True)


def parse_task_name(task_name: str) -> tuple[str, int]:
    if "_task_" not in task_name:
        raise ValueError(f"Unexpected task format: {task_name}")
    suite_name, task_id = task_name.rsplit("_task_", 1)
    return suite_name, int(task_id)


def build_rename_map(model) -> dict[str, str]:
    feature_keys = set(model.config.input_features.keys())
    rename_map: dict[str, str] = {}

    if "observation.images.camera1" in feature_keys:
        rename_map["observation.images.image"] = "observation.images.camera1"
    if "observation.images.camera2" in feature_keys:
        rename_map["observation.images.image2"] = "observation.images.camera2"
    if "observation.images.wrist_image" in feature_keys:
        rename_map["observation.images.image2"] = "observation.images.wrist_image"

    if "observation.images.image" in feature_keys:
        rename_map.pop("observation.images.image", None)

    return rename_map


def extract_render_frame(env):
    try:
        frames = env.call("render")
        if isinstance(frames, (list, tuple)) and frames:
            frame = frames[0]
            if frame is not None:
                return frame
        if isinstance(frames, np.ndarray):
            return frames
    except Exception:
        return None
    return None


def normalize_frame_uint8(frame):
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
    try:
        pixels = observation.get("pixels")
        if isinstance(pixels, dict):
            for key in ("image", "image2"):
                frame = pixels.get(key)
                normalized = normalize_frame_uint8(frame)
                if normalized is not None:
                    return normalized
    except Exception:
        return None
    return None


def configure_reproducibility(seed: int, deterministic: bool) -> None:
    set_seed(seed)
    random.seed(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True


def load_policy_and_processors(policy_name: str, device: str):
    registry = POLICY_REGISTRY
    if policy_name not in registry:
        valid = ", ".join(registry)
        raise ValueError(f"Unknown policy '{policy_name}'. Valid options: {valid}")

    model = registry[policy_name]["build_model"]()
    model.to(torch.device(device))
    model.eval()

    router_metadata = {}
    if hasattr(model, "get_routing_metadata"):
        router_metadata = model.get_routing_metadata()

    rename_map = build_rename_map(model)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=model.config,
        pretrained_path=OFFICIAL_MODEL_ID,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": rename_map},
        },
    )

    return model, preprocessor, postprocessor, rename_map, router_metadata


def run_task_vectorized(
    model,
    preprocessor,
    postprocessor,
    task_name: str,
    n_episodes: int,
    *,
    n_envs: int,
    use_async_envs: bool,
    start_seed: int = 0,
):
    build_start = time.time()
    suite_name, task_id = parse_task_name(task_name)

    env_cfg = LiberoEnvConfig(task=suite_name, task_ids=[task_id], control_mode="relative")
    envs = make_env(
        cfg=env_cfg,
        n_envs=n_envs,
        use_async_envs=use_async_envs,
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
        start_seed=start_seed,
    )
    eval_s = time.time() - eval_start

    env.close()
    successes = [int(s) for s in metrics["successes"]]
    return float(np.mean(successes)), build_s, eval_s


def run_task_stepwise(
    model,
    preprocessor,
    postprocessor,
    task_name: str,
    n_episodes: int,
    *,
    start_seed: int = 0,
    render_every_n_steps: int,
    on_frame: Callable | None = None,
    on_step: Callable | None = None,
    on_episode_reset: Callable | None = None,
    on_episode_end: Callable | None = None,
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

    eval_start = time.time()
    success_count = 0

    try:
        for ep in range(n_episodes):
            model.reset()
            observation, info = env.reset(seed=[start_seed + ep])
            if on_episode_reset is not None:
                on_episode_reset(env, observation, info, ep)

            if on_frame is not None and 0 % render_every_n_steps == 0:
                on_frame(env, observation, info, step=0, episode_index=ep)

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

                if on_step is not None:
                    on_step(step, max_steps, info, ep)

                if on_frame is not None and step % render_every_n_steps == 0:
                    on_frame(env, observation, info, step=step, episode_index=ep)

                if "final_info" in info and info["final_info"]["is_success"][0]:
                    success_count += 1
                    break

            if on_episode_end is not None:
                on_episode_end(info, step, max_steps, ep)
    finally:
        env.close()

    eval_s = time.time() - eval_start
    return success_count / float(n_episodes), build_s, eval_s
