"""Exercise resource contention with real threads, without loading model weights."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
import time

import pytest

from dsapi import api, audio, pipeline, scheduling, score, tools


def wait_for(predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < deadline, "scheduler did not reach expected state"
        time.sleep(.005)


@pytest.mark.parametrize("gpu_slots", [1, 2])
def test_gpu_budget_shared_across_stages_while_cpu_can_overlap(gpu_slots):
    scheduler = scheduling.Scheduler(3, gpu_slots)
    release, separating, synthesizing, extracting = (Event() for _ in range(4))

    def work(stage, gpu, entered):
        with scheduler.resource(stage, gpu=gpu):
            entered.set()
            assert release.wait(5)

    with ThreadPoolExecutor(max_workers=3) as pool:
        try:
            first = pool.submit(work, "separate", True, separating)
            assert separating.wait(5)
            second = pool.submit(work, "synthesis", True, synthesizing)
            third = pool.submit(work, "f0", False, extracting)
            assert extracting.wait(5)
            if gpu_slots == 1:
                wait_for(lambda: scheduler.gpu.status()["waiting"] == 1)
                assert not synthesizing.is_set()
            else:
                assert synthesizing.wait(5)
        finally:
            release.set()
        for future in (first, second, third):
            future.result(timeout=5)
    assert scheduler.gpu.status() == {"capacity": gpu_slots, "active": 0, "waiting": 0}


def test_same_stage_is_serial_even_with_multiple_gpu_slots():
    scheduler = scheduling.Scheduler(3, 3)
    release, first_entered, second_entered = Event(), Event(), Event()

    def work(entered):
        with scheduler.resource("synthesis", gpu=True):
            entered.set()
            assert release.wait(5)

    with ThreadPoolExecutor(max_workers=2) as pool:
        try:
            first = pool.submit(work, first_entered)
            assert first_entered.wait(5)
            second = pool.submit(work, second_entered)
            wait_for(lambda: scheduler.status()["stages"]["synthesis"]["waiting"] == 1)
            assert not second_entered.is_set()
        finally:
            release.set()
        first.result(timeout=5)
        second.result(timeout=5)


def test_fifo_gpu_admission_and_failure_release():
    scheduler = scheduling.Scheduler(3, 1)
    order = []

    def work(name, fail=False):
        with scheduler.resource(name, gpu=True):
            order.append(name)
            if fail:
                raise RuntimeError("model failed")

    with ThreadPoolExecutor(max_workers=2) as pool:
        with scheduler.resource("occupied", gpu=True):
            first = pool.submit(work, "align", True)
            wait_for(lambda: scheduler.gpu.status()["waiting"] == 1)
            second = pool.submit(work, "synthesis")
            wait_for(lambda: scheduler.gpu.status()["waiting"] == 2)
        with pytest.raises(RuntimeError, match="model failed"):
            first.result(timeout=5)
        second.result(timeout=5)
    assert order == ["align", "synthesis"]
    with scheduler.resource("align", gpu=True):
        assert scheduler.gpu.status()["active"] == 1


def test_job_limit_preserves_queued_state_and_allows_pipeline_overlap(tmp_path, monkeypatch):
    scheduler = scheduling.Scheduler(2, 1)
    monkeypatch.setattr(scheduling, "scheduler", scheduler)
    monkeypatch.setattr(pipeline.settings, "WORK", tmp_path)
    release = Event()
    entered = [Event() for _ in range(3)]
    jobs = [pipeline.Job(f"job-{i}", pipeline.JobSpec()) for i in range(3)]

    def work(i):
        entered[i].set()
        assert release.wait(5)

    for i, job in enumerate(jobs):
        job._run = lambda i=i: work(i)
    with ThreadPoolExecutor(max_workers=3) as pool:
        try:
            first = pool.submit(jobs[0].run)
            assert entered[0].wait(5)
            second = pool.submit(jobs[1].run)
            assert entered[1].wait(5), "whole-job lock prevented overlap"
            third = pool.submit(jobs[2].run)
            wait_for(lambda: scheduler.jobs.status()["waiting"] == 1)
            assert jobs[2].state.status == "queued"
            assert jobs[2].state.started_at is None
            assert not entered[2].is_set()
        finally:
            release.set()
        for future in (first, second, third):
            assert future.result(timeout=5).status == "done"
    assert scheduler.jobs.status()["active"] == 0


def test_sync_render_uses_same_admission_budget(monkeypatch):
    scheduler = scheduling.Scheduler(1, 1)
    monkeypatch.setattr(scheduling, "scheduler", scheduler)
    monkeypatch.setattr(api, "validate_voice", lambda _: None)
    entered = Event()
    monkeypatch.setattr(api, "_render", lambda _: entered.set())
    with ThreadPoolExecutor(max_workers=1) as pool:
        with scheduler.jobs.enter():
            pending = pool.submit(api.render, api.ScoreIn(phrases=[]))
            wait_for(lambda: scheduler.jobs.status()["waiting"] == 1)
            assert not entered.is_set()
        pending.result(timeout=5)
    assert entered.is_set()


def test_parent_budgets_child_alignment_without_blocking_cpu_f0(monkeypatch, tmp_path):
    scheduler = scheduling.Scheduler(3, 1)
    monkeypatch.setattr(scheduling, "scheduler", scheduler)
    monkeypatch.setattr(tools, "_workers", {})
    monkeypatch.setattr(tools.settings, "torch_device", lambda: "cpu")
    entered = set()
    monkeypatch.setattr(tools._Worker, "run", lambda worker, args, log: entered.add(worker.script))
    with ThreadPoolExecutor(max_workers=2) as pool:
        with scheduler.resource("synthesis", gpu=True):
            align = pool.submit(tools._run, "HubertFA", "hubertfa_runner.py", log=tmp_path / "align.log")
            wait_for(lambda: scheduler.gpu.status()["waiting"] == 1)
            f0 = pool.submit(tools._run, "SOME", "f0_runner.py", log=tmp_path / "f0.log")
            f0.result(timeout=5)
            assert entered == {"f0_runner.py"}
        align.result(timeout=5)
    assert entered == {"f0_runner.py", "hubertfa_runner.py"}


def test_score_and_mimic_share_synthesis_slot(monkeypatch):
    from types import SimpleNamespace
    import numpy as np

    scheduler = scheduling.Scheduler(3, 2)
    monkeypatch.setattr(scheduling, "scheduler", scheduler)
    monkeypatch.setattr(score, "score_inputs", lambda *args: ([], [], [], [], None))
    first_entered, release, second_entered = Event(), Event(), Event()

    def first_render(*args):
        first_entered.set()
        assert release.wait(5)
        return np.ones(100)

    def second_render(*args):
        second_entered.set()
        return np.ones(100)

    first_vb = SimpleNamespace(render=first_render)
    second_vb = SimpleNamespace(render=second_render, predict_durations=lambda *args: [],
                                predict_pitch=lambda *args: [])
    phrase = score.Phrase([], phones=[("i", 0, .1)], f0_midi=np.ones(100))
    with ThreadPoolExecutor(max_workers=2) as pool:
        try:
            first = pool.submit(score.render_mimic, first_vb, phrase)
            assert first_entered.wait(5)
            second = pool.submit(score.render_score, second_vb, phrase)
            wait_for(lambda: scheduler.status()["stages"]["synthesis"]["waiting"] == 1)
            assert not second_entered.is_set()
        finally:
            release.set()
        assert len(first.result(timeout=5)) == len(second.result(timeout=5)) == 100


def test_shared_separator_keeps_output_directories_isolated(tmp_path, monkeypatch):
    from types import ModuleType, SimpleNamespace
    import sys

    scheduler = scheduling.Scheduler(3, 2)
    monkeypatch.setattr(scheduling, "scheduler", scheduler)
    monkeypatch.setattr(audio, "_separators", {})
    entered, release = Event(), Event()
    loads = []

    class Separator:
        def __init__(self, **kwargs):
            self.model_instance = SimpleNamespace(output_dir=kwargs["output_dir"])

        def load_model(self, **kwargs):
            loads.append(kwargs)

        def separate(self, src):
            if src == "first":
                entered.set()
                assert release.wait(5)
            # Read shared mutable state after the competing call was submitted.
            from pathlib import Path
            path = Path(self.model_instance.output_dir) / "stem_(Vocals).wav"
            assert self.output_dir == str(path.parent)
            path.write_text(src)
            return [str(path)]

    module = ModuleType("audio_separator.separator")
    module.Separator = Separator
    monkeypatch.setitem(sys.modules, "audio_separator.separator", module)
    folders = [tmp_path / "one", tmp_path / "two"]
    for folder in folders:
        folder.mkdir()
    with ThreadPoolExecutor(max_workers=2) as pool:
        try:
            first = pool.submit(audio._separate, "model", "first", folders[0])
            assert entered.wait(5)
            second = pool.submit(audio._separate, "model", "second", folders[1])
            wait_for(lambda: scheduler.status()["stages"]["separate"]["waiting"] == 1)
        finally:
            release.set()
        assert first.result(timeout=5)["vocals"].read_text() == "first"
        assert second.result(timeout=5)["vocals"].read_text() == "second"
    assert len(loads) == 1
