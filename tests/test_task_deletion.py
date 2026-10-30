from concurrent.futures import Future

import pytest
from fastapi.testclient import TestClient

from dsapi import api, pipeline


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(api.settings, "WORK", tmp_path / "work")
    monkeypatch.setattr(api, "API_KEY", "test-key")
    monkeypatch.setattr(api, "_jobs", {})
    monkeypatch.setattr(api, "_job_futures", {})
    with TestClient(api.app, headers={"Authorization": "Bearer test-key"}) as client:
        yield client


@pytest.mark.parametrize("status", ["done", "failed"])
@pytest.mark.parametrize("live", [False, True])
def test_delete_cleans_task_and_upload_but_preserves_other_files(client, tmp_path, status, live):
    source = tmp_path / "original.wav"
    source.write_bytes(b"original")
    job = pipeline.Job("remove", pipeline.JobSpec(audio_path=str(source)))
    job.state.status = status
    job.save()
    (job.dir / "audio.mp3").write_bytes(b"audio")
    uploads = api.settings.WORK / "uploads" / "remove"
    uploads.mkdir(parents=True)
    (uploads / "input.wav").write_bytes(b"upload")
    keep = pipeline.Job("keep", pipeline.JobSpec())
    if live:
        api._jobs["remove"] = job
        future = Future()
        future.set_result(job.state)
        api._job_futures["remove"] = future
    assert client.delete("/tasks/remove").json() == {"task_id": "remove", "deleted": True}
    assert not job.dir.exists()
    assert not uploads.exists()
    assert source.read_bytes() == b"original"
    assert (keep.dir / "job.json").is_file()
    assert "remove" not in api._jobs and "remove" not in api._job_futures
    assert client.get("/query_result?task_id=remove").status_code == 404
    assert client.delete("/tasks/remove").status_code == 404
    assert [t["task_id"] for t in client.get("/tasks").json()] == ["keep"]


@pytest.mark.parametrize("status", ["queued", "running", "done", "failed"])
def test_unfinished_worker_cannot_be_deleted(client, status):
    job = pipeline.Job("busy", pipeline.JobSpec())
    job.state.status = status
    job.save()
    api._jobs["busy"] = job
    api._job_futures["busy"] = Future()
    assert client.delete("/tasks/busy").status_code == 409
    assert (job.dir / "job.json").exists()


@pytest.mark.parametrize("status", ["queued", "running"])
def test_saved_active_task_cannot_be_deleted(client, status):
    job = pipeline.Job("busy", pipeline.JobSpec())
    job.state.status = status
    job.save()
    assert client.delete("/tasks/busy").status_code == 409
    assert job.dir.exists()


def test_auth_and_path_safety(client, tmp_path):
    job = pipeline.Job("safe", pipeline.JobSpec())
    job.state.status = "failed"
    job.save()
    assert client.delete("/tasks/safe", headers={"Authorization": ""}).status_code == 401
    assert job.dir.exists()
    for task_id in ["..", "../safe", str(job.dir), "uploads/safe", "safe\\other"]:
        assert client.get("/query_result", params={"task_id": task_id}).status_code == 404
    (api.settings.WORK / "linked").symlink_to(job.dir, target_is_directory=True)
    assert client.delete("/tasks/linked").status_code == 404
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "safe").mkdir()
    (api.settings.WORK / "uploads").symlink_to(outside, target_is_directory=True)
    assert client.delete("/tasks/safe").status_code == 409
    assert job.dir.exists() and (outside / "safe").exists()
