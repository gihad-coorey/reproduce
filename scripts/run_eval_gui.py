import json
import os
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np
import torch
from PIL import Image, ImageTk

from lerobot.envs import libero as libero_module
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import add_envs_task, preprocess_observation
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import ACTION


TASK_FILE = Path("configs/tasks.json")
DEFAULT_RESULTS_FILE = Path("results/gui_eval_results.json")
MODEL_OPTIONS = {
    "SmolVLA Base (models/smolvla)": "models/smolvla",
    "SmolVLA LIBERO Tuned (models/smolvla_libero_tuned_migrated)": "models/smolvla_libero_tuned_migrated",
}


def build_rename_map(model):
    feature_keys = set(model.config.input_features.keys())
    rename_map = {}
    if "observation.images.camera1" in feature_keys:
        rename_map["observation.images.image"] = "observation.images.camera1"
    if "observation.images.camera2" in feature_keys:
        rename_map["observation.images.image2"] = "observation.images.camera2"
    if "observation.images.wrist_image" in feature_keys:
        rename_map["observation.images.image2"] = "observation.images.wrist_image"
    if "observation.images.image" in feature_keys:
        rename_map.pop("observation.images.image", None)
    return rename_map


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
    except Exception:
        # Keep evaluation running even if rendering momentarily fails.
        return None
    return None


def normalize_frame_for_display(frame):
    """Normalize a frame into uint8 HWC RGB for Tk preview."""
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    if not isinstance(frame, np.ndarray):
        return None

    # Handle batch dimensions by selecting the first item.
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
    """Fallback preview source from raw LIBERO observations."""
    try:
        pixels = observation.get("pixels")
        if isinstance(pixels, dict):
            for key in ("image", "image2"):
                frame = pixels.get(key)
                normalized = normalize_frame_for_display(frame)
                if normalized is not None:
                    return normalized
    except Exception:
        return None
    return None


def run_single_task(
    model,
    preprocessor,
    postprocessor,
    task_name,
    episodes,
    progress_cb,
    render_frame_cb=None,
    ui_pump_cb=None,
    fallback_preview_cb=None,
):
    suite_name, task_id = parse_task_name(task_name)

    env_cfg = LiberoEnvConfig(task=suite_name, task_ids=[task_id], control_mode="relative")
    envs = make_env(cfg=env_cfg, n_envs=1, use_async_envs=False, trust_remote_code=False)
    env = envs[suite_name][task_id]

    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=model.config)

    success_count = 0
    try:
        for ep in range(episodes):
            model.reset()
            observation, _info = env.reset(seed=[ep])
            if render_frame_cb is not None:
                frame = extract_render_frame(env)
                if frame is not None:
                    render_frame_cb(frame)
                elif fallback_preview_cb is not None:
                    fallback = extract_preview_from_observation(observation)
                    if fallback is not None:
                        fallback_preview_cb(fallback)
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
                if render_frame_cb is not None:
                    frame = extract_render_frame(env)
                    if frame is not None:
                        render_frame_cb(frame)
                    elif fallback_preview_cb is not None:
                        fallback = extract_preview_from_observation(observation)
                        if fallback is not None:
                            fallback_preview_cb(fallback)
                done = terminated | truncated | done
                step += 1

                # Keep Tk responsive while rollout runs on main thread.
                if ui_pump_cb is not None and (step % 5 == 0):
                    ui_pump_cb()

                if "final_info" in info and info["final_info"]["is_success"][0]:
                    success_count += 1
                    break

            progress_cb(f"{task_name}: episode {ep + 1}/{episodes} complete")

    finally:
        env.close()

    return success_count / float(episodes)


class EvalGuiApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SmolVLA LIBERO Eval GUI")
        self.root.geometry("980x820")

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model_var = tk.StringVar(value=list(MODEL_OPTIONS.keys())[0])
        self.episodes_var = tk.StringVar(value="2")
        self.results_var = tk.StringVar(value=str(DEFAULT_RESULTS_FILE))
        self._preview_photo = None
        self._latest_frame = None
        self._is_running = False
        self._preview_errors = 0
        self._first_frame_seen = False
        self._fallback_preview_used = False

        self._build_ui()
        self._load_tasks()
        self.root.after(50, self._update_preview)

    def _build_ui(self):
        frm = ttk.Frame(self.root, padding=12)
        frm.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frm, text="Model").pack(anchor=tk.W)
        ttk.Combobox(
            frm,
            values=list(MODEL_OPTIONS.keys()),
            textvariable=self.model_var,
            state="readonly",
        ).pack(fill=tk.X, pady=(0, 8))

        ttk.Label(frm, text="LIBERO Task List (multi-select)").pack(anchor=tk.W)
        self.task_list = tk.Listbox(frm, selectmode=tk.EXTENDED, height=12)
        self.task_list.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        row = ttk.Frame(frm)
        row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(row, text="Episodes per task").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.episodes_var, width=8).pack(side=tk.LEFT, padx=(8, 16))
        ttk.Label(row, text="Results JSON path").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.results_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))

        btn_row = ttk.Frame(frm)
        btn_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(btn_row, text="Select All", command=self._select_all).pack(side=tk.LEFT)
        self.run_button = ttk.Button(btn_row, text="Run Eval", command=self._start_run)
        self.run_button.pack(side=tk.LEFT, padx=8)

        ttk.Label(frm, text="Live Preview").pack(anchor=tk.W)
        self.preview_status_var = tk.StringVar(value="No frame yet")
        ttk.Label(frm, textvariable=self.preview_status_var).pack(anchor=tk.W, pady=(0, 4))
        preview_frame = ttk.Frame(frm, width=512, height=512)
        preview_frame.pack(fill=tk.NONE, expand=False, pady=(0, 8))
        preview_frame.pack_propagate(False)
        self.preview_label = ttk.Label(preview_frame, text="No frame yet")
        self.preview_label.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frm, text="Log").pack(anchor=tk.W)
        self.log = tk.Text(frm, height=10)
        self.log.pack(fill=tk.BOTH, expand=False)

    def _load_tasks(self):
        if TASK_FILE.exists():
            with TASK_FILE.open() as f:
                tasks = json.load(f)
        else:
            tasks = [
                "libero_spatial_task_0",
                "libero_spatial_task_1",
                "libero_object_task_0",
                "libero_object_task_1",
                "libero_goal_task_0",
            ]

        self.task_list.delete(0, tk.END)
        for t in tasks:
            self.task_list.insert(tk.END, t)

    def _select_all(self):
        self.task_list.select_set(0, tk.END)

    def _append_log(self, text):
        self.log.insert(tk.END, text + "\n")
        self.log.see(tk.END)

    def _set_latest_frame(self, frame):
        if frame is None:
            return
        self._latest_frame = frame
        if not self._first_frame_seen:
            self._first_frame_seen = True
            self._append_log("Preview: received first render frame")

    def _set_latest_frame_fallback(self, frame):
        if frame is None:
            return
        self._latest_frame = frame
        if not self._first_frame_seen:
            self._first_frame_seen = True
        if not self._fallback_preview_used:
            self._fallback_preview_used = True
            self._append_log("Preview: using observation camera fallback (env.render() unavailable)")

    def _update_preview(self):
        frame = self._latest_frame
        if frame is not None:
            try:
                display = normalize_frame_for_display(frame)
                if display is not None:
                    img = Image.fromarray(display).resize((512, 512), Image.Resampling.BILINEAR)
                    self._preview_photo = ImageTk.PhotoImage(img)
                    self.preview_label.configure(image=self._preview_photo, text="")
                    h, w = display.shape[:2]
                    self.preview_status_var.set(f"Preview active: {w}x{h}")
                else:
                    self.preview_status_var.set("Preview waiting: unsupported frame format")
                    if self._preview_errors < 3:
                        self._preview_errors += 1
                        self._append_log("Preview warning: unsupported frame format from env.render()")
            except Exception:
                self.preview_status_var.set("Preview error: failed to convert frame")
                if self._preview_errors < 3:
                    self._preview_errors += 1
                    self._append_log("Preview warning: failed to convert render frame")
        self.root.after(50, self._update_preview)

    def _validate_inputs(self):
        selected = [self.task_list.get(i) for i in self.task_list.curselection()]
        if not selected:
            messagebox.showerror("Missing tasks", "Select at least one task.")
            return None

        try:
            episodes = int(self.episodes_var.get())
            if episodes <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Bad episodes", "Episodes per task must be a positive integer.")
            return None

        model_path = MODEL_OPTIONS[self.model_var.get()]
        result_path = Path(self.results_var.get())

        return selected, episodes, model_path, result_path

    def _start_run(self):
        if self._is_running:
            return
        validated = self._validate_inputs()
        if validated is None:
            return

        selected, episodes, model_path, result_path = validated
        self._is_running = True
        self.run_button.configure(state=tk.DISABLED)
        # On macOS, creating LIBERO/MuJoCo envs in a worker thread can trigger
        # hard crashes (Trace/BPT trap). Run eval on the Tk main thread.
        self.root.after(0, lambda: self._run_eval(selected, episodes, model_path, result_path))

    def _run_eval(self, selected, episodes, model_path, result_path):
        try:
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            patch_libero_action_dim()

            self._append_log(f"Device: {self.device}")
            self._append_log(f"Loading model from {model_path}...")
            model = SmolVLAPolicy.from_pretrained(model_path)
            model.to(torch.device(self.device))
            model.eval()

            rename_map = build_rename_map(model)
            self._append_log(f"Image rename map: {rename_map}")
            preprocessor, postprocessor = make_pre_post_processors(
                policy_cfg=model.config,
                pretrained_path=model_path,
                preprocessor_overrides={
                    "device_processor": {"device": self.device},
                    "rename_observations_processor": {"rename_map": rename_map},
                },
            )

            results = {}
            for task in selected:
                self._append_log(f"Running: {task}")
                sr = run_single_task(
                    model=model,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    task_name=task,
                    episodes=episodes,
                    progress_cb=self._append_log,
                    render_frame_cb=self._set_latest_frame,
                    ui_pump_cb=self.root.update,
                    fallback_preview_cb=self._set_latest_frame_fallback,
                )
                results[task] = float(sr)
                self._append_log(f"Done: {task} success_rate={sr:.3f}")

            if not self._first_frame_seen:
                self._append_log("Preview: no frames received; env.render() may be unavailable in this run")

            result_path.parent.mkdir(parents=True, exist_ok=True)
            with result_path.open("w") as f:
                json.dump(results, f, indent=2)

            self._append_log("\n===== GUI EVAL RESULTS =====")
            for k, v in results.items():
                self._append_log(f"{k}: {v:.3f}")
            self._append_log(f"Saved results to {result_path}")

        except Exception as exc:
            self._append_log(f"ERROR: {exc}")
            messagebox.showerror("Evaluation failed", str(exc))
        finally:
            self._is_running = False
            self.run_button.configure(state=tk.NORMAL)


def main():
    root = tk.Tk()
    app = EvalGuiApp(root)
    app._append_log("Ready. Select model and tasks, then click Run Eval.")
    root.mainloop()


if __name__ == "__main__":
    main()
