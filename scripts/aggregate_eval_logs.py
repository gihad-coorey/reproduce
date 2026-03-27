#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunRecord:
    run_id: str
    policy: str = ""
    task_file: str = ""
    mode: str = ""
    device: str = ""
    slurm_job_id: str = ""
    slurm_array_task_id: str = ""
    episodes_per_task: int | None = None
    seed: int | None = None
    started: bool = False
    ended: bool = False
    tasks_total: int | None = None
    tasks_completed: int = 0
    total_build_s: float = 0.0
    total_eval_s: float = 0.0
    reported_mean_success_rate: float | None = None
    task_success: dict[str, float] = field(default_factory=dict)

    def apply_common(self, event: dict[str, Any]) -> None:
        self.policy = str(event.get("policy", self.policy) or self.policy)
        self.task_file = str(event.get("task_file", self.task_file) or self.task_file)
        self.mode = str(event.get("mode", self.mode) or self.mode)
        self.device = str(event.get("device", self.device) or self.device)
        self.slurm_job_id = str(event.get("slurm_job_id", self.slurm_job_id) or self.slurm_job_id)
        self.slurm_array_task_id = str(event.get("slurm_array_task_id", self.slurm_array_task_id) or self.slurm_array_task_id)

        if "episodes_per_task" in event and event["episodes_per_task"] is not None:
            self.episodes_per_task = int(event["episodes_per_task"])
        if "seed" in event and event["seed"] is not None:
            self.seed = int(event["seed"])
        if "tasks_total" in event and event["tasks_total"] is not None:
            self.tasks_total = int(event["tasks_total"])

    def apply_event(self, event: dict[str, Any]) -> None:
        self.apply_common(event)
        event_name = event.get("event")

        if event_name == "run_start":
            self.started = True
            return

        if event_name == "task_end":
            task = event.get("task")
            success_rate = event.get("success_rate")
            if task is not None and success_rate is not None:
                self.task_success[str(task)] = float(success_rate)

            if "tasks_completed" in event and event["tasks_completed"] is not None:
                self.tasks_completed = int(event["tasks_completed"])
            if "build_s" in event and event["build_s"] is not None:
                self.total_build_s += float(event["build_s"])
            if "eval_s" in event and event["eval_s"] is not None:
                self.total_eval_s += float(event["eval_s"])
            return

        if event_name == "run_end":
            self.ended = True
            if "tasks_completed" in event and event["tasks_completed"] is not None:
                self.tasks_completed = int(event["tasks_completed"])
            if "tasks_total" in event and event["tasks_total"] is not None:
                self.tasks_total = int(event["tasks_total"])
            if "total_build_s" in event and event["total_build_s"] is not None:
                self.total_build_s = float(event["total_build_s"])
            if "total_eval_s" in event and event["total_eval_s"] is not None:
                self.total_eval_s = float(event["total_eval_s"])
            if "mean_success_rate" in event and event["mean_success_rate"] is not None:
                self.reported_mean_success_rate = float(event["mean_success_rate"])

    @property
    def computed_mean_success_rate(self) -> float | None:
        if not self.task_success:
            return None
        return sum(self.task_success.values()) / float(len(self.task_success))

    @property
    def mean_success_rate(self) -> float | None:
        if self.reported_mean_success_rate is not None:
            return self.reported_mean_success_rate
        return self.computed_mean_success_rate


@dataclass
class ParseStats:
    files_seen: int = 0
    lines_seen: int = 0
    events_seen: int = 0
    malformed_lines: int = 0
    non_event_json_lines: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate eval performance from JSONL events in SLURM log files."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["logs/**/*.out"],
        help="Input file paths or glob patterns.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on malformed JSON lines instead of warning and continuing.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional path to write full aggregation JSON.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Summary output format printed to stdout.",
    )
    return parser.parse_args()


def resolve_input_files(patterns: list[str]) -> list[Path]:
    files: set[Path] = set()
    for pattern in patterns:
        p = Path(pattern)
        if p.exists() and p.is_file():
            files.add(p)
            continue

        for match in glob.glob(pattern, recursive=True):
            match_path = Path(match)
            if match_path.is_file():
                files.add(match_path)

    return sorted(files)


def maybe_parse_event_json(line: str) -> dict[str, Any] | None:
    stripped = line.strip()
    if not stripped.startswith("{") or not stripped.endswith("}"):
        return None

    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        return None
    if "event" not in parsed or "run_id" not in parsed:
        return None
    return parsed


def parse_events(files: list[Path], strict: bool) -> tuple[dict[str, RunRecord], ParseStats]:
    runs: dict[str, RunRecord] = {}
    stats = ParseStats(files_seen=len(files))

    for file_path in files:
        with file_path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw_line in handle:
                stats.lines_seen += 1
                line = raw_line.rstrip("\n")

                if not line.strip():
                    continue

                if not line.lstrip().startswith("{"):
                    continue

                try:
                    maybe_event = maybe_parse_event_json(line)
                except json.JSONDecodeError:
                    stats.malformed_lines += 1
                    if strict:
                        raise
                    continue

                if maybe_event is None:
                    stats.non_event_json_lines += 1
                    continue

                stats.events_seen += 1
                run_id = str(maybe_event["run_id"])
                run = runs.get(run_id)
                if run is None:
                    run = RunRecord(run_id=run_id)
                    runs[run_id] = run
                run.apply_event(maybe_event)

    return runs, stats


def mean(values: list[float]) -> float:
    return sum(values) / float(len(values))


def stddev(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = mean(values)
    var = sum((v - m) ** 2 for v in values) / float(len(values) - 1)
    return math.sqrt(var)


def build_group_rollups(runs: dict[str, RunRecord]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[RunRecord]] = {}
    for run in runs.values():
        key = (run.policy, run.task_file)
        grouped.setdefault(key, []).append(run)

    rollups: list[dict[str, Any]] = []
    for (policy, task_file), records in sorted(grouped.items()):
        mean_srs = [r.mean_success_rate for r in records if r.mean_success_rate is not None]
        total_eval_s = [r.total_eval_s for r in records]
        total_build_s = [r.total_build_s for r in records]

        rollups.append(
            {
                "policy": policy,
                "task_file": task_file,
                "runs": len(records),
                "runs_completed": sum(1 for r in records if r.ended),
                "mean_success_rate_mean": mean(mean_srs) if mean_srs else None,
                "mean_success_rate_std": stddev(mean_srs) if len(mean_srs) > 1 else 0.0 if mean_srs else None,
                "mean_success_rate_min": min(mean_srs) if mean_srs else None,
                "mean_success_rate_max": max(mean_srs) if mean_srs else None,
                "mean_total_eval_s": mean(total_eval_s) if total_eval_s else None,
                "mean_total_build_s": mean(total_build_s) if total_build_s else None,
            }
        )

    return rollups


def build_run_summaries(runs: dict[str, RunRecord]) -> list[dict[str, Any]]:
    records = sorted(runs.values(), key=lambda r: (r.policy, r.task_file, r.run_id))
    summaries: list[dict[str, Any]] = []
    for run in records:
        summaries.append(
            {
                "run_id": run.run_id,
                "policy": run.policy,
                "task_file": run.task_file,
                "mode": run.mode,
                "device": run.device,
                "slurm_job_id": run.slurm_job_id,
                "slurm_array_task_id": run.slurm_array_task_id,
                "episodes_per_task": run.episodes_per_task,
                "seed": run.seed,
                "started": run.started,
                "ended": run.ended,
                "tasks_total": run.tasks_total,
                "tasks_completed": run.tasks_completed,
                "mean_success_rate": run.mean_success_rate,
                "mean_success_rate_computed": run.computed_mean_success_rate,
                "mean_success_rate_reported": run.reported_mean_success_rate,
                "total_build_s": run.total_build_s,
                "total_eval_s": run.total_eval_s,
                "tasks_observed": len(run.task_success),
            }
        )
    return summaries


def print_text_summary(payload: dict[str, Any]) -> None:
    parse_stats = payload["parse_stats"]
    run_summaries = payload["run_summaries"]
    group_rollups = payload["group_rollups"]

    print("=== Eval Log Aggregation ===")
    print(f"files_seen={parse_stats['files_seen']} lines_seen={parse_stats['lines_seen']} events_seen={parse_stats['events_seen']}")
    print(
        "malformed_lines="
        f"{parse_stats['malformed_lines']} non_event_json_lines={parse_stats['non_event_json_lines']}"
    )
    print(f"runs_found={len(run_summaries)} groups_found={len(group_rollups)}")
    print("")

    if group_rollups:
        print("-- Group Rollups (policy x task_file) --")
        for row in group_rollups:
            print(
                "policy={policy} task_file={task_file} runs={runs} completed={runs_completed} "
                "mean_sr={mean_success_rate_mean} std_sr={mean_success_rate_std}"
                .format(**row)
            )
        print("")

    if run_summaries:
        print("-- Per-Run Summary --")
        for row in run_summaries:
            print(
                "run_id={run_id} policy={policy} task_file={task_file} "
                "mean_sr={mean_success_rate} tasks={tasks_completed}/{tasks_total} "
                "eval_s={total_eval_s} build_s={total_build_s} ended={ended}"
                .format(**row)
            )


def main() -> None:
    args = parse_args()
    files = resolve_input_files(args.inputs)

    if not files:
        raise FileNotFoundError("No input log files found. Check --inputs patterns.")

    runs, parse_stats = parse_events(files=files, strict=args.strict)
    run_summaries = build_run_summaries(runs)
    group_rollups = build_group_rollups(runs)

    payload = {
        "parse_stats": {
            "files_seen": parse_stats.files_seen,
            "lines_seen": parse_stats.lines_seen,
            "events_seen": parse_stats.events_seen,
            "malformed_lines": parse_stats.malformed_lines,
            "non_event_json_lines": parse_stats.non_event_json_lines,
        },
        "run_summaries": run_summaries,
        "group_rollups": group_rollups,
    }

    if args.output_json.strip():
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print_text_summary(payload)


if __name__ == "__main__":
    main()
