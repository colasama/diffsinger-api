"""Progress and partial output must remain usable while rendering continues."""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from dsapi import api, pipeline


@pytest.fixture
def job_env(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline.settings, "WORK", tmp_path)
    monkeypatch.setattr(api, "_jobs", {})
    monkeypatch.setattr(api, "API_KEY", "")
    clock = SimpleNamespace(now=100.0)
    timer = SimpleNamespace(time=lambda: clock.now, perf_counter=lambda: clock.now)
    monkeypatch.setattr(pipeline, "time", timer)
    monkeypatch.setattr(api, "time", timer)
    return clock


@pytest.mark.parametrize("fail", [False, True])
def test_elapsed_includes_initialization_and_stops_at_completion(job_env, fail):
    clock = job_env
    job = pipeline.Job("timing", pipeline.JobSpec())
    api._jobs["timing"] = job
    clock.now = 130.0  # Time spent queued is excluded.
    assert api.query_result("timing")["elapsed_seconds"] is None

    def run():
        saved = json.loads((job.dir / "job.json").read_text())
        assert saved["status"] == "running"
        assert saved["started_at"] == 130.0
        clock.now = 135.0  # Includes initialization before the first stage.
        with job.stage("render_mimic"):
            assert api.query_result("timing")["elapsed_seconds"] == 5.0
            clock.now = 138.0
            assert api.query_result("timing")["elapsed_seconds"] == 8.0
            if fail:
                raise RuntimeError("render failed")

    job._run = run
    job.run()
    clock.now = 200.0
    result = api.query_result("timing")
    assert result["status"] == ("failed" if fail else "done")
    assert result["elapsed_seconds"] == 8.0
    assert result["timings"] == {"render_mimic": 3.0}
    api._jobs.clear()  # The result is also stable after reloading from disk.
    assert api.query_result("timing")["elapsed_seconds"] == 8.0


def test_mimic_can_be_streamed_while_score_is_running(job_env):
    job = pipeline.Job("partial", pipeline.JobSpec())
    api._jobs["partial"] = job
    content = b"ID3" + bytes(range(256))

    def run():
        with job.stage("render_mimic"):
            path = job.dir / "mimic.mp3"
            path.write_bytes(content)
            job.state.outputs["mimic"] = str(path)
        with job.stage("render_score"):
            with TestClient(api.app) as client:
                state = client.get("/query_result?task_id=partial").json()
                assert state["status"] == "running"
                assert state["stage"] == "render_score"
                assert "score" not in state["outputs"]
                url = state["outputs"]["mimic"]
                response = client.get(url)
                assert response.status_code == 200
                assert response.content == content
                response = client.get(url, headers={"Range": "bytes=0-9"})
                assert response.status_code == 206
                assert response.content == content[:10]

    job._run = run
    assert job.run().status == "done"


def test_old_saved_jobs_without_timestamps_still_load(job_env):
    job = pipeline.Job("legacy", pipeline.JobSpec())
    saved = json.loads((job.dir / "job.json").read_text())
    del saved["started_at"], saved["finished_at"]
    saved.update(status="done", timings={"fetch": 3.5})
    (job.dir / "job.json").write_text(json.dumps(saved))
    result = api.query_result("legacy")
    assert result["elapsed_seconds"] is None
    assert result["timings"] == {"fetch": 3.5}
