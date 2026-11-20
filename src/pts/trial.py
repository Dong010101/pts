"""Trial protocol v1, vendored identically in FreeSpeed and PTS.

The clock starts in the executor before its first task observation. Compilation,
ready-pose motion and finalization are outside it; first real inference is inside.
Success is an explicit operator/evaluator decision, never inferred from exit code.
"""

import argparse
import asyncio
import csv
import json
import math
import os
from pathlib import Path
import signal
import time
import uuid


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class Trial:
    def __init__(self, directory, *, task, method, mode, duration_s, action_hz=15,
                 setup_id="0", repeat_id="0", criteria="operator assessment"):
        if not math.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("duration must be finite and positive")
        if not math.isfinite(action_hz) or action_hz <= 0:
            raise ValueError("action_hz must be finite and positive")
        self.directory = Path(directory)
        self.limit = duration_s
        self.started = asyncio.Event()
        self.document = {
            "schema_version": "trial-v1", "trial_id": self.directory.name,
            "control_token": uuid.uuid4().hex, "task": task, "method": method,
            "mode": mode, "setup_id": str(setup_id), "repeat_id": str(repeat_id),
            "criteria": criteria, "action_hz": action_hz, "timeout_s": duration_s,
            "timing": "first_task_observation_to_stop_decision_including_first_inference",
            "started_monotonic_s": None, "started_wall_s": None,
            "outcome": "prepared", "task_success": None,
        }
        self._save()

    def _save(self):
        atomic_json(self.directory / "trial_result.json", self.document)

    def start(self):
        if self.started.is_set():
            raise RuntimeError("trial clock was already started")
        self.document.update(started_monotonic_s=time.monotonic(), started_wall_s=time.time(),
                             outcome="running")
        self._save()
        self.started.set()

    def finish(self, outcome, reason, *, evaluator=None):
        now = time.monotonic()
        start = self.document["started_monotonic_s"]
        duration = None if start is None else max(0.0, now - start)
        success = (outcome == "success") if start is not None and outcome != "unscored" else None
        self.document.update(outcome=outcome, task_success=success, reason=str(reason),
                             evaluator=evaluator, stopped_monotonic_s=now, stopped_wall_s=time.time(),
                             duration_s=duration,
                             control_steps=None if duration is None else duration * self.document["action_hz"])
        self._save()

    def read_decision(self):
        path = self.directory / "trial_control.json"
        if not path.exists():
            return None
        command = json.loads(path.read_text())
        if command.get("control_token") != self.document["control_token"]:
            raise ValueError("trial control belongs to another run")
        if command.get("outcome") not in {"success", "failure", "unscored"}:
            raise ValueError("invalid trial outcome")
        if not command.get("reason") or not command.get("evaluator"):
            raise ValueError("trial decision needs reason and evaluator")
        return command


async def supervise(managed, trial, *, interrupt=None, startup_timeout=30):
    """Start a prepared/ready runner and finalize all stop paths consistently."""
    interrupt = interrupt or asyncio.Event()
    managed.runner.trial = trial
    failure = None
    try:
        await managed.start_at(time.time())
        startup_deadline = time.monotonic() + startup_timeout
        while True:
            status = await managed.status()
            if status["error"]:
                raise RuntimeError(status["error"])
            if interrupt.is_set():
                trial.finish("aborted", "operator_interrupt")
                break
            if status["state"] in {"stopped", "error"}:
                # A debug cycle/step limit is never a task-success signal.
                trial.finish("unscored", "executor_finished_without_task_decision")
                break
            if trial.started.is_set():
                if time.monotonic() - trial.document["started_monotonic_s"] >= trial.limit:
                    trial.finish("timeout", "wall_time_limit")
                    break
                decision = trial.read_decision()
                if decision:
                    trial.finish(decision["outcome"], decision["reason"], evaluator=decision["evaluator"])
                    break
            elif time.monotonic() >= startup_deadline:
                raise TimeoutError("executor did not start the trial clock")
            await asyncio.sleep(0.01)
    except BaseException as exc:
        failure = exc
        trial.finish("aborted" if isinstance(exc, asyncio.CancelledError) else "error", str(exc))
    finally:
        # Preserve the stop-decision timestamp; draining SDK/logging is measured separately.
        try:
            await managed.stop(trial.document["outcome"])
            status = await managed.status()
            if status["error"]:
                raise RuntimeError(status["error"])
            await managed.finalize()
        except BaseException as exc:
            failure = failure or exc
            trial.document.update(outcome="error", task_success=False if trial.started.is_set() else None,
                                  cleanup_error=f"{type(exc).__name__}: {exc}")
        trial.document["finalized_wall_s"] = time.time()
        trial.document["finalization_s"] = time.monotonic() - trial.document["stopped_monotonic_s"]
        trial._save()
    if failure is not None:
        raise failure
    return trial.document


def add_trial_arguments(parser):
    parser.add_argument("--session", help="Unique trial directory name")
    parser.add_argument("--duration", type=float, default=60.0, help="Task timeout, excluding ready/warmup")
    parser.add_argument("--setup-id", default="0")
    parser.add_argument("--repeat-id", default="0")
    parser.add_argument("--criteria", help="Versioned task success/quality criteria")


def session_name(value, method):
    name = value or f"{method}_{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}_{uuid.uuid4().hex[:8]}"
    if not name or name in {".", ".."} or Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError("session must be a single directory name")
    return name


def install_interrupt():
    event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, event.set)
    return event


def remove_interrupt():
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.remove_signal_handler(sig)


def send_decision(directory, outcome, reason, evaluator):
    directory = Path(directory)
    result = json.loads((directory / "trial_result.json").read_text())
    if result["outcome"] != "running":
        raise ValueError("decisions are accepted only during a running trial")
    atomic_json(directory / "trial_control.json", {
        "control_token": result["control_token"], "outcome": outcome,
        "reason": reason, "evaluator": evaluator, "requested_wall_s": time.time(),
    })


def score(directory, success, reason, evaluator):
    directory = Path(directory)
    result = json.loads((directory / "trial_result.json").read_text())
    if (result["outcome"] in {"prepared", "running"} or result.get("duration_s") is None
            or "finalized_wall_s" not in result):
        raise ValueError("only a completed, started attempt can be scored")
    if success and result["outcome"] not in {"success", "unscored"}:
        raise ValueError("failed, aborted, timed-out or errored attempts cannot be promoted to success")
    path = directory / "trial_score.json"
    if path.exists():
        raise FileExistsError("score already exists; preserve the original adjudication")
    # Exclusive creation avoids two reviewers silently overwriting one another.
    with path.open("x") as stream:
        json.dump({"schema_version": "trial-v1", "control_token": result["control_token"],
                   "task_success": bool(success), "reason": reason, "evaluator": evaluator,
                   "scored_wall_s": time.time()}, stream, indent=2)


def export_rows(directories):
    rows, seen = [], set()
    for directory in map(Path, directories):
        result = json.loads((directory / "trial_result.json").read_text())
        if (result["outcome"] in {"prepared", "running"} or result.get("duration_s") is None
                or "finalized_wall_s" not in result):
            raise ValueError(f"not a completed attempt: {directory}")
        score_path = directory / "trial_score.json"
        if score_path.exists():
            adjudication = json.loads(score_path.read_text())
            if adjudication["control_token"] != result["control_token"]:
                raise ValueError("score belongs to another run")
            result["task_success"] = adjudication["task_success"]
        if result["task_success"] is None:
            raise ValueError(f"unscored trial: {directory}; use score before exporting")
        if result["control_token"] in seen:
            raise ValueError(f"duplicate trial: {directory}")
        seen.add(result["control_token"])
        rows.append({key: result[key] for key in
                     ("method", "trial_id", "task", "mode", "setup_id", "repeat_id",
                      "duration_s", "control_steps", "outcome", "criteria")})
        rows[-1]["success"] = int(result["task_success"])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    decision = sub.add_parser("signal", help="Stop a running attempt with an explicit decision")
    decision.add_argument("directory", type=Path)
    decision.add_argument("--outcome", choices=("success", "failure", "unscored"), required=True)
    scoring = sub.add_parser("score", help="Adjudicate a stopped attempt, e.g. polish surface quality")
    scoring.add_argument("directory", type=Path)
    scoring.add_argument("--success", type=int, choices=(0, 1), required=True)
    for p in (decision, scoring):
        p.add_argument("--reason", required=True)
        p.add_argument("--evaluator", required=True)
    export = sub.add_parser("export", help="Export all selected attempts; never silently drop failures")
    export.add_argument("directories", nargs="+", type=Path)
    export.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "signal":
        send_decision(args.directory, args.outcome, args.reason, args.evaluator)
    elif args.command == "score":
        score(args.directory, args.success, args.reason, args.evaluator)
    else:
        rows = export_rows(args.directories)
        with args.output.open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
