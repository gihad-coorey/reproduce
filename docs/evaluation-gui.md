# GUI Evaluation

The GUI entrypoint is `scripts/run_gui.py`.

## Launch

```bash
python scripts/run_gui.py
```

## What the GUI Provides

- Policy dropdown from the same shared policy registry as CLI
- Multi-select task list from task file
- Episodes and render frequency controls
- Live frame preview during rollout
- Per-task run logs and summary

## Outputs

By default, GUI writes grouped results to:

- `results/<sanitized_policy>/<sanitized_task_file_stem>/gui_eval.json`
- Event log JSONL alongside results:
  - `results/<sanitized_policy>/<sanitized_task_file_stem>/gui_eval.jsonl`

## Notes

- On Linux, GUI mode requires a display server.
- The GUI pumps Tk callbacks during rollout to keep preview responsive.
- Runtime policy loading and preprocessors are shared with CLI via `scripts/common.py`.
