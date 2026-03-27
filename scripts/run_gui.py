import json
import sys
import os
import time
import tkinter as tk
import uuid
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np
from PIL import Image, ImageTk
from libero.libero import benchmark

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))
from scripts.common import (
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
    parse_task_name,
    resolve_project_path,
    resolve_runtime_device,
    run_task_stepwise,
)

TASK_FILE = resolve_project_path(DEFAULT_TASK_FILE, PROJECT_ROOT)
DEFAULT_POLICY_NAME = next(iter(POLICY_REGISTRY.keys()))
DEFAULT_RESULTS_FILE = build_default_result_path(PROJECT_ROOT, DEFAULT_POLICY_NAME, TASK_FILE, prefix="gui_eval")


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
    run_id,
    progress_cb,
    render_frame_cb=None,
    ui_pump_cb=None,
    fallback_preview_cb=None,
    status_cb=None,
    render_every_n_steps=1,
    event_log_path=None,
):
    spinner_chars = "|/-\\"

    def on_episode_reset(env, observation, info, ep_index):
        ep_start = time.time()
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
        episode_starts[ep_index] = ep_start

    def on_step(step, max_steps, info, ep_index):
        del info
        if status_cb is not None and (step % 5 == 0):
            spinner = spinner_chars[(step // 5) % len(spinner_chars)]
            status_cb(f"Running {task_name} ep {ep_index + 1}/{episodes}: step {step}/{max_steps} {spinner}")
        if ui_pump_cb is not None and (step % 5 == 0):
            ui_pump_cb()

    def on_frame(env, observation, _info, step, episode_index):
        del _info, step, episode_index  # Unused, but keep to match expected callback signature in `run_task_stepwise`
        frame = extract_render_frame(env)
        if frame is not None:
            if render_frame_cb is not None:
                render_frame_cb(frame)
        elif fallback_preview_cb is not None:
            fallback = extract_preview_from_observation(observation)
            if fallback is not None:
                fallback_preview_cb(fallback)

    def on_episode_end(info, step, max_steps, ep_index):
        progress_cb(f"{task_name}: episode {ep_index + 1}/{episodes} complete (steps: {step}/{max_steps})")
        episode_success = int("final_info" in info and info["final_info"]["is_success"][0])
        emit_event(
            "episode_end",
            run_id,
            event_log_path=event_log_path,
            task=task_name,
            episode=ep_index + 1,
            steps=step,
            max_steps=max_steps,
            success=episode_success,
            episode_s=float(time.time() - episode_starts.get(ep_index, time.time())),
        )

    episode_starts: dict[int, float] = {}
    sr, _build_s, _eval_s = run_task_stepwise(
        model=model,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        task_name=task_name,
        n_episodes=episodes,
        start_seed=RUNTIME_SEED,
        render_every_n_steps=render_every_n_steps,
        on_frame=on_frame,
        on_step=on_step,
        on_episode_reset=on_episode_reset,
        on_episode_end=on_episode_end,
    )
    return sr


class EvalGuiApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SmolVLA LIBERO Eval GUI")
        self.root.geometry("980x820")

        self.device = resolve_runtime_device()
        self.policy_var = tk.StringVar(value="HF")
        self.episodes_var = tk.StringVar(value="1")
        self.render_every_var = tk.StringVar(value="10")
        self.results_var = tk.StringVar(value=str(DEFAULT_RESULTS_FILE))
        self._preview_photo = None
        self._latest_frame = None
        self._is_running = False
        self._preview_errors = 0
        self._first_frame_seen = False
        self._fallback_preview_used = False
        self._last_ui_pump_time = 0.0
        self._last_preview_draw_time = 0.0
        self.task_instruction_map = {}

        self._build_ui()
        self._load_tasks()
        self.root.after(50, self._update_preview)

    def _build_ui(self):
        frm = ttk.Frame(self.root, padding=12)
        frm.pack(fill=tk.BOTH, expand=True)

        policy_row = ttk.Frame(frm)
        policy_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(policy_row, text="Policy").pack(side=tk.LEFT)
        policy_combo = ttk.Combobox(
            policy_row,
            textvariable=self.policy_var,
            values=list(POLICY_REGISTRY.keys()),
            state="readonly",
            width=24,
        )
        policy_combo.pack(side=tk.LEFT, padx=(8, 16))
        ttk.Label(policy_row, text=f"Checkpoint: {OFFICIAL_MODEL_ID}").pack(side=tk.LEFT)

        ttk.Label(frm, text="LIBERO Task List (multi-select)").pack(anchor=tk.W)
        self.task_list = tk.Listbox(frm, selectmode=tk.EXTENDED, height=12)
        self.task_list.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        self.task_list.bind("<<ListboxSelect>>", self._on_task_selection_changed)

        ttk.Label(frm, text="Selected Task Instruction").pack(anchor=tk.W)
        self.task_instruction_var = tk.StringVar(value="Select a task to view its language instruction.")
        ttk.Label(
            frm,
            textvariable=self.task_instruction_var,
            justify=tk.LEFT,
            wraplength=920,
        ).pack(fill=tk.X, pady=(0, 8))

        row = ttk.Frame(frm)
        row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(row, text="Episodes per task").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.episodes_var, width=8).pack(side=tk.LEFT, padx=(8, 16))
        ttk.Label(row, text="Render every N steps").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.render_every_var, width=8).pack(side=tk.LEFT, padx=(8, 16))
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
        self.task_instruction_map = self._build_task_instruction_map(tasks)
        self._on_task_selection_changed()

    def _build_task_instruction_map(self, tasks):
        instruction_map = {}
        suites = {}
        for task_name in tasks:
            instruction_map[task_name] = task_name
            try:
                suite_name, task_id = parse_task_name(task_name)
            except Exception:
                continue

            suite = suites.get(suite_name)
            if suite is None:
                try:
                    suite_ctor = benchmark.get_benchmark_dict()[suite_name]
                    suite = suite_ctor()
                    suites[suite_name] = suite
                except Exception:
                    continue

            try:
                instruction = suite.tasks[task_id].language
                if isinstance(instruction, str) and instruction.strip():
                    instruction_map[task_name] = instruction.strip()
            except Exception:
                continue

        return instruction_map

    def _on_task_selection_changed(self, _event=None):
        selected = [self.task_list.get(i) for i in self.task_list.curselection()]
        if not selected:
            self.task_instruction_var.set("Select a task to view its language instruction.")
            return

        if len(selected) == 1:
            task = selected[0]
            instruction = self.task_instruction_map.get(task, task)
            self.task_instruction_var.set(f"{task}: {instruction}")
            return

        lines = []
        for task in selected:
            instruction = self.task_instruction_map.get(task, task)
            lines.append(f"- {task}: {instruction}")
        self.task_instruction_var.set("\n".join(lines))

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
        # `update()` processes timer callbacks (e.g. `after`) so live preview
        # continues updating during long rollout loops on the main thread.
        self.root.update()

    def _draw_preview_frame(self, frame):
        display = normalize_frame_uint8(frame)
        if display is None:
            self.preview_status_var.set("Preview waiting: unsupported frame format")
            if self._preview_errors < 3:
                self._preview_errors += 1
                self._append_log("Preview warning: unsupported frame format from env.render()")
            return

        img = Image.fromarray(display).resize((512, 512), Image.Resampling.BILINEAR)
        self._preview_photo = ImageTk.PhotoImage(img)
        self.preview_label.configure(image=self._preview_photo, text="")
        h, w = display.shape[:2]
        self.preview_status_var.set(f"Preview active: {w}x{h}")
        # Flush pending widget draws so preview appears immediately during rollout.
        self.root.update_idletasks()

    def _draw_preview_if_due(self):
        frame = self._latest_frame
        if frame is None:
            return
        now = time.time()
        # Limit redraws to avoid excessive image conversions during inference.
        if (now - self._last_preview_draw_time) < 0.06:
            return
        self._last_preview_draw_time = now
        try:
            self._draw_preview_frame(frame)
        except Exception:
            self.preview_status_var.set("Preview error: failed to convert frame")
            if self._preview_errors < 3:
                self._preview_errors += 1
                self._append_log("Preview warning: failed to convert render frame")

    def _set_latest_frame(self, frame):
        if frame is None:
            return
        self._latest_frame = frame
        if not self._first_frame_seen:
            self._first_frame_seen = True
            self._append_log("Preview: received first render frame")
        self._draw_preview_if_due()

    def _set_latest_frame_fallback(self, frame):
        if frame is None:
            return
        self._latest_frame = frame
        if not self._first_frame_seen:
            self._first_frame_seen = True
        if not self._fallback_preview_used:
            self._fallback_preview_used = True
            self._append_log("Preview: using observation camera fallback (env.render() unavailable)")
        self._draw_preview_if_due()

    def _update_preview(self):
        self._draw_preview_if_due()
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

        try:
            render_every_n_steps = int(self.render_every_var.get())
            if render_every_n_steps <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Bad render frequency", "Render every N steps must be a positive integer.")
            return None

        policy_name = self.policy_var.get().strip()
        if policy_name not in POLICY_REGISTRY:
            messagebox.showerror("Bad policy", f"Unknown policy '{policy_name}'.")
            return None

        result_path = Path(self.results_var.get())
        return selected, episodes, render_every_n_steps, policy_name, result_path

    def _start_run(self):
        if self._is_running:
            return
        validated = self._validate_inputs()
        if validated is None:
            return

        selected, episodes, render_every_n_steps, policy_name, result_path = validated
        self._is_running = True
        self.run_button.configure(state=tk.DISABLED)
        # On macOS, creating LIBERO/MuJoCo envs in a worker thread can trigger
        # hard crashes (Trace/BPT trap). Run eval on the Tk main thread.
        self.root.after(
            0,
            lambda: self._run_eval(
                selected,
                episodes,
                render_every_n_steps,
                policy_name,
                result_path,
            ),
        )

    def _run_eval(
        self,
        selected,
        episodes,
        render_every_n_steps,
        policy_name,
        result_path,
    ):
        run_id = str(uuid.uuid4())
        event_log_path = result_path.with_suffix(".events.jsonl")
        try:
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = RUNTIME_MPS_FALLBACK
            configure_reproducibility(seed=RUNTIME_SEED, deterministic=RUNTIME_DETERMINISTIC)
            if result_path == DEFAULT_RESULTS_FILE:
                result_path = build_default_result_path(
                    PROJECT_ROOT,
                    policy_name,
                    TASK_FILE,
                    prefix="gui_eval",
                )
            event_log_path = result_path.with_suffix(".jsonl")
            self._set_run_status("Starting evaluation...")
            self._append_log(f"Device: {self.device}")
            self._append_log(f"Render frame frequency: every {render_every_n_steps} step(s)")
            self._append_log(f"Selected policy: {policy_name}")
            self._append_log(f"Loading model from {OFFICIAL_MODEL_ID}...")
            self._pump_ui()
            model, preprocessor, postprocessor, rename_map, router_metadata = load_policy_and_processors(
                policy_name, self.device
            )

            emit_event(
                "run_start",
                run_id,
                event_log_path=event_log_path,
                mode="gui",
                tasks_total=len(selected),
                episodes_per_task=episodes,
                device=self.device,
                policy=policy_name,
                model_path=OFFICIAL_MODEL_ID,
                **router_metadata,
            )

            self._append_log(f"Image rename map: {rename_map}")

            results = {}
            for task_idx, task in enumerate(selected, start=1):
                self._append_log(f"Running: {task}")
                self._set_run_status(f"Running {task}...")
                self._pump_ui()
                sr = run_single_task(
                    model=model,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    task_name=task,
                    episodes=episodes,
                    run_id=run_id,
                    progress_cb=self._append_log,
                    render_frame_cb=self._set_latest_frame,
                    ui_pump_cb=self._pump_ui,
                    fallback_preview_cb=self._set_latest_frame_fallback,
                    status_cb=self._set_run_status,
                    render_every_n_steps=render_every_n_steps,
                    event_log_path=event_log_path,
                )
                results[task] = float(sr)
                self._append_log(f"Done: {task} success_rate={sr:.3f}")
                router_task_metadata = {}
                if hasattr(model, "model") and hasattr(model.model, "last_router_decision"):
                    router_task_metadata = dict(model.model.last_router_decision)
                emit_event(
                    "task_end",
                    run_id,
                    event_log_path=event_log_path,
                    task=task,
                    success_rate=float(sr),
                    **router_task_metadata,
                )

            if not self._first_frame_seen:
                self._append_log("Preview: no frames received; env.render() may be unavailable in this run")

            result_path.parent.mkdir(parents=True, exist_ok=True)
            with result_path.open("w") as f:
                json.dump(results, f, indent=2)

            self._append_log("\n===== GUI EVAL RESULTS =====")
            for k, v in results.items():
                self._append_log(f"{k}: {v:.3f}")
            self._append_log(f"Saved results to {result_path}")
            emit_event(
                "run_end",
                run_id,
                event_log_path=event_log_path,
                status="ok",
                mean_success_rate=float(np.mean(list(results.values()))),
                tasks_completed=len(results),
            )
            self._set_run_status("Done")

        except Exception as exc:
            self._append_log(f"ERROR: {exc}")
            emit_event("run_end", run_id, event_log_path=event_log_path, status="failed", error=str(exc))
            messagebox.showerror("Evaluation failed", str(exc))
            self._set_run_status("Failed")
        finally:
            self._is_running = False
            self.run_button.configure(state=tk.NORMAL)


def main():
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        raise SystemExit("GUI mode requires a display server on Linux. Use CLI mode instead.")
    root = tk.Tk()
    app = EvalGuiApp(root)
    app._append_log("Ready. Select policy and tasks, then click Run Eval.")
    root.mainloop()


if __name__ == "__main__":
    main()
