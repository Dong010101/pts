import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from pts.trial import Trial, atomic_json, export_rows, score, send_decision, supervise


class Managed:
    """A running executor, including a cancellable outstanding policy request."""
    def __init__(self, *, startup_delay=0, cleanup_error=False, finish_early=False):
        self.runner = SimpleNamespace()
        self.state, self.error = "ready", None
        self.startup_delay, self.cleanup_error = startup_delay, cleanup_error
        self.finish_early = finish_early
        self.stopped = self.finalized = False

    async def start_at(self, timestamp):
        self.state = "running"
        async def execute():
            await asyncio.sleep(self.startup_delay)
            self.runner.trial.start()
            if self.finish_early:
                self.state = "stopped"
                return
            await asyncio.Event().wait()
        self.task = asyncio.create_task(execute())

    async def status(self):
        return {"state": self.state, "error": self.error}

    async def stop(self, reason):
        self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)
        self.stopped = True
        self.state = "stopped"

    async def finalize(self):
        if self.cleanup_error:
            raise RuntimeError("video finalize failed")
        self.finalized = True


def make_trial(directory, **kwargs):
    directory.mkdir()
    return Trial(directory, task="flip", method="method", mode="mock", duration_s=kwargs.pop("duration_s", .06), **kwargs)


def test_timeout_includes_waiting_for_first_inference_but_excludes_startup(tmp_path):
    async def scenario():
        trial = make_trial(tmp_path / "timeout")
        managed = Managed(startup_delay=.08)
        before = time.monotonic()
        result = await supervise(managed, trial)
        assert result["outcome"] == "timeout" and result["task_success"] is False
        assert .06 <= result["duration_s"] < .13
        assert time.monotonic() - before >= .14
        assert result["control_steps"] == pytest.approx(result["duration_s"] * 15)
        assert managed.stopped and managed.finalized
    asyncio.run(scenario())


def test_explicit_success_cancels_inflight_execution_and_records_evaluator(tmp_path):
    async def scenario():
        trial = make_trial(tmp_path / "success", duration_s=60)
        managed = Managed()
        async def evaluate():
            await trial.started.wait()
            await asyncio.sleep(.02)
            send_decision(trial.directory, "success", "target face stable", "operator-A")
        evaluation = asyncio.create_task(evaluate())
        result = await supervise(managed, trial)
        await evaluation
        assert result["task_success"] is True and result["duration_s"] < 1
        assert result["evaluator"] == "operator-A" and managed.stopped
        assert export_rows([trial.directory])[0]["success"] == 1
    asyncio.run(scenario())


def test_debug_completion_requires_scoring_and_quality_can_fail(tmp_path):
    async def scenario():
        trial = make_trial(tmp_path / "unscored")
        result = await supervise(Managed(finish_early=True), trial)
        assert result["task_success"] is None
        with pytest.raises(ValueError, match="unscored"):
            export_rows([trial.directory])
        score(trial.directory, False, "surface quality below criterion", "reviewer")
        assert export_rows([trial.directory])[0]["success"] == 0
        with pytest.raises(FileExistsError):
            score(trial.directory, True, "overwrite", "reviewer")
    asyncio.run(scenario())


def test_timeout_cannot_be_scored_as_success_and_is_kept_in_export(tmp_path):
    async def scenario():
        trial = make_trial(tmp_path / "timeout")
        await supervise(Managed(), trial)
        with pytest.raises(ValueError, match="promoted"):
            score(trial.directory, True, "bad", "reviewer")
        row = export_rows([trial.directory])[0]
        assert row["success"] == 0 and row["duration_s"] >= .06
        with pytest.raises(ValueError, match="duplicate"):
            export_rows([trial.directory, trial.directory])
    asyncio.run(scenario())


def test_stale_control_file_is_an_error_not_success(tmp_path):
    async def scenario():
        trial = make_trial(tmp_path / "stale")
        atomic_json(trial.directory / "trial_control.json", {"control_token": "old", "outcome": "success"})
        managed = Managed()
        with pytest.raises(ValueError, match="another run"):
            await supervise(managed, trial)
        assert trial.document["task_success"] is False and managed.stopped
    asyncio.run(scenario())


def test_finalization_error_downgrades_success_and_preserves_stop_time(tmp_path):
    async def scenario():
        trial = make_trial(tmp_path / "broken")
        managed = Managed(cleanup_error=True)
        async def evaluate():
            await trial.started.wait()
            send_decision(trial.directory, "success", "confirmed", "operator")
        evaluator = asyncio.create_task(evaluate())
        with pytest.raises(RuntimeError, match="video"):
            await supervise(managed, trial)
        await evaluator
        result = json.loads((trial.directory / "trial_result.json").read_text())
        assert result["outcome"] == "error" and result["task_success"] is False
        assert "stopped_monotonic_s" in result and managed.stopped
    asyncio.run(scenario())
