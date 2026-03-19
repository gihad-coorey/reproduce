import json
import os
import time
import argparse
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np
import torch
from PIL import Image, ImageTk

from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import add_envs_task, preprocess_observation
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import ACTION


TASK_FILE = Path("configs/tasks.json")
DEFAULT_RESULTS_FILE = Path("results/gui_eval_results.json")
OFFICIAL_MODEL_ID = "HuggingFaceVLA/smolvla_libero"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch the SmolVLA LIBERO evaluation GUI.",
        epilog=(
            "Examples:\n"
            "  python scripts/run_eval_gui.py\n"
            "  python scripts/run_eval_gui.py --smoke"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--smoke", action="store_true", help="Preselect first task and set episodes=1")
    return parser.parse_args()


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
    except Exception as exc:
        # Keep evaluation running even if rendering momentarily fails.
        print(f"Warning: failed to extract render frame: {exc}")
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
    except Exception as exc:
        print(f"Warning: failed to extract preview frame: {exc}")
        return None
    return None


def extract_task_instruction(env, reset_info, task_name):
    """Best-effort extraction of a human-readable task instruction."""
    if isinstance(reset_info, dict):
        instruction = reset_info.get("instruction")
        if isinstance(instruction, (list, tuple)) and instruction:
            instruction = instruction[0]
        if isinstance(instruction, str) and instruction.strip():
            return instruction.strip()

    # Try common env APIs without failing evaluation if unavailable.
    for method_name in ("get_task_instruction", "get_language_instruction"):
        try:
            value = env.call(method_name)
            if isinstance(value, (list, tuple)) and value:
                value = value[0]
            if isinstance(value, str) and value.strip():
                return value.strip()
        except Exception:
            pass

    return task_name


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
    status_cb=None,
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
            observation, info = env.reset(seed=[ep])

            instruction = extract_task_instruction(env, info, task_name)
            progress_cb(f"Instruction: {instruction}")

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
            last_progress_log_step = 0
            spinner_chars = "|/-\\"

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

                # Lightweight heartbeat in status line every 5 steps.
                if status_cb is not None and (step % 5 == 0):
                    spinner = spinner_chars[(step // 5) % len(spinner_chars)]
                    status_cb(
                        f"Running {task_name} ep {ep + 1}/{episodes}: step {step}/{max_steps} {spinner}"
                    )

                # Log progress more sparsely to avoid Tk Text widget stalls.
                if (step - last_progress_log_step) >= 50 or step == max_steps:
                    last_progress_log_step = step
                    progress_cb(f"Progress: {task_name} ep {ep + 1}/{episodes} step {step}/{max_steps}")

                # Keep Tk responsive while rollout runs on main thread.
                if ui_pump_cb is not None and (step % 20 == 0):
                    ui_pump_cb()

                if "final_info" in info and info["final_info"]["is_success"][0]:
                    success_count += 1
                    break

            progress_cb(f"{task_name}: episode {ep + 1}/{episodes} complete (steps: {step}/{max_steps})")

    finally:
        env.close()

    return success_count / float(episodes)


class EvalGuiApp:
    def __init__(self, root, smoke_mode=False):
        self.root = root
        self.root.title("SmolVLA LIBERO Eval GUI")
        self.root.geometry("980x820")

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model_var = tk.StringVar(value=OFFICIAL_MODEL_ID)
        self.episodes_var = tk.StringVar(value="2")
        self.results_var = tk.StringVar(value=str(DEFAULT_RESULTS_FILE))
        self._preview_photo = None
        self._latest_frame = None
        self._is_running = False
        self._preview_errors = 0
        self._first_frame_seen = False
        self._fallback_preview_used = False
        self._last_ui_pump_time = 0.0
        self._smoke_mode = smoke_mode

        self._build_ui()
        self._load_tasks()
        if self._smoke_mode:
            self.episodes_var.set("1")
            if self.task_list.size() > 0:
                self.task_list.selection_clear(0, tk.END)
                self.task_list.select_set(0)
        self.root.after(50, self._update_preview)

    def _build_ui(self):
        frm = ttk.Frame(self.root, padding=12)
        frm.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frm, text="Model").pack(anchor=tk.W)
        ttk.Entry(frm, textvariable=self.model_var, state="readonly").pack(fill=tk.X, pady=(0, 8))

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

        self.run_status_var = tk.StringVar(value="Idle")
        ttk.Label(frm, textvariable=self.run_status_var).pack(anchor=tk.W, pady=(0, 8))

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
        # Mirror to terminal so progress is visible even if Tk lags.
        print(text, flush=True)
        self.root.update_idletasks()

    def _set_run_status(self, text):
        self.run_status_var.set(text)

    def _pump_ui(self):
        now = time.time()
        if (now - self._last_ui_pump_time) < 0.15:
            return
        self._last_ui_pump_time = now
        self.root.update_idletasks()

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

        model_path = self.model_var.get()
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

            self._set_run_status("Starting evaluation...")
            self._append_log(f"Device: {self.device}")
            self._append_log(f"Loading model from {model_path}...")
            self._pump_ui()
            model = SmolVLAPolicy.from_pretrained(model_path)
            model.to(torch.device(self.device))
            model.eval()

            rename_map = build_rename_map(model)
            self._append_log(f"Image rename map: {rename_map}")
            self._set_run_status("Preparing preprocessors...")
            self._pump_ui()
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
                self._set_run_status(f"Running {task}...")
                self._pump_ui()
                sr = run_single_task(
                    model=model,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    task_name=task,
                    episodes=episodes,
                    progress_cb=self._append_log,
                    render_frame_cb=self._set_latest_frame,
                    ui_pump_cb=self._pump_ui,
                    fallback_preview_cb=self._set_latest_frame_fallback,
                    status_cb=self._set_run_status,
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
            self._set_run_status("Done")

        except Exception as exc:
            self._append_log(f"ERROR: {exc}")
            messagebox.showerror("Evaluation failed", str(exc))
            self._set_run_status("Failed")
        finally:
            self._is_running = False
            self.run_button.configure(state=tk.NORMAL)


def main():
    args = parse_args()
    root = tk.Tk()
    app = EvalGuiApp(root, smoke_mode=args.smoke)
    app._append_log("Ready. Select model and tasks, then click Run Eval.")
    if args.smoke:
        app._append_log("Smoke mode enabled: first task selected and episodes set to 1")
    root.mainloop()


if __name__ == "__main__":
    main()
