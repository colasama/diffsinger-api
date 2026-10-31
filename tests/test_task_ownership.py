import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from dsapi import api, pipeline


@pytest.fixture
def client(monkeypatch, tmp_path, voice_files):
    monkeypatch.setattr(api.settings, "WORK", tmp_path)
    monkeypatch.setattr(api, "API_KEY", "legacy-secret")
    monkeypatch.setattr(api, "API_KEYS", {"alice": "alice-secret", "bob": "bob-secret"})
    monkeypatch.setattr(api, "_jobs", {})
    monkeypatch.setattr(api, "_job_futures", {})
    monkeypatch.setattr(api, "_executor", SimpleNamespace(submit=lambda _: None))
    return TestClient(api.app)


def headers(user):
    return {"Authorization": f"Bearer {user}-secret"}


def submit(client, user, upload=False):
    if upload:
        response = client.post("/release_task_upload", headers=headers(user),
                               files={"audio_file": ("clip.wav", b"audio", "audio/wav")},
                               data={"lrc_text": "[00:01]Hello"})
    else:
        response = client.post("/release_task", headers=headers(user), json={"song_id": "1"})
    assert response.status_code == 200
    return response.json()["task_id"]


@pytest.mark.parametrize("upload", [False, True])
@pytest.mark.parametrize("reload", [False, True])
def test_owner_persists_and_all_task_endpoints_are_isolated(client, monkeypatch, upload, reload):
    alice = submit(client, "alice", upload)
    bob = submit(client, "bob")
    job = api._jobs[alice]
    audio = job.dir / "mimic.mp3"
    audio.write_bytes(b"ID3-test")
    (job.dir / "score.json").write_text('{"phrases": []}')
    job.state.status = "done"
    job.state.outputs["mimic"] = str(audio)
    job.save()
    saved = (job.dir / "job.json").read_text()
    assert json.loads(saved)["owner_id"] == "alice"
    assert "alice-secret" not in saved
    if reload:
        api._jobs.clear()

    # Filter before applying the limit, even when another user's task is newer.
    assert [t["task_id"] for t in client.get("/tasks?limit=1", headers=headers("alice")).json()] == [alice]
    assert [t["task_id"] for t in client.get("/tasks", headers=headers("bob")).json()] == [bob]
    endpoints = [("GET", "/query_result"), ("GET", "/v1/audio"),
                 ("GET", "/v1/score"), ("POST", "/evaluate"), ("DELETE", f"/tasks/{alice}")]
    for user in ["bob", "legacy"]:
        for method, path in endpoints:
            response = client.request(method, path, params={"task_id": alice}, headers=headers(user))
            assert response.status_code == 404, (user, path, response.text)
            assert response.json() == {"detail": "unknown task"}
    assert audio.is_file()
    result = client.get("/query_result", params={"task_id": alice}, headers=headers("alice"))
    assert result.status_code == 200
    assert "owner_id" not in result.json()
    assert client.get("/v1/audio", params={"task_id": alice, "key": "alice-secret"}).content == b"ID3-test"
    assert client.get("/v1/score", params={"task_id": alice}, headers=headers("alice")).status_code == 200

    from dsapi import evaluate
    monkeypatch.setattr(evaluate, "evaluate", lambda path: {"evaluated": path.name})
    assert client.post("/evaluate", params={"task_id": alice}, headers=headers("alice")).json() == {"evaluated": alice}
    assert client.delete(f"/tasks/{alice}", headers=headers("alice")).status_code == 200
    assert not job.dir.exists()
    assert not (api.settings.WORK / "uploads" / alice).exists()


def test_legacy_history_is_not_exposed_to_named_users(client, monkeypatch):
    job = pipeline.Job("old", pipeline.JobSpec())
    path = job.dir / "job.json"
    state = json.loads(path.read_text())
    del state["owner_id"]
    path.write_text(json.dumps(state))
    for user in ["alice", "bob"]:
        assert client.get("/tasks", headers=headers(user)).json() == []
        assert client.get("/query_result?task_id=old", headers=headers(user)).status_code == 404
    assert client.get("/query_result?task_id=old", headers=headers("legacy")).status_code == 200
    named = submit(client, "alice")
    monkeypatch.setattr(api, "API_KEY", "")
    monkeypatch.setattr(api, "API_KEYS", {})
    assert client.get("/query_result?task_id=old").status_code == 200
    assert client.get("/query_result", params={"task_id": named}).status_code == 404


def test_key_rotation_and_authentication(client, monkeypatch):
    task = submit(client, "alice")
    monkeypatch.setattr(api, "API_KEY", "")
    monkeypatch.setattr(api, "API_KEYS", {"alice": "rotated-secret", "bob": "bob-secret"})
    for key in [None, "", "invalid", "alice-secret", "legacy-secret"]:
        params = {"key": key} if key is not None else {}
        assert client.get("/tasks", params=params).status_code == 401
        assert client.post("/release_task", json={"song_id": "1"}, params=params).status_code == 401
    assert client.get("/tasks", headers=headers("rotated")).json()[0]["task_id"] == task
    # Headers take precedence; a query parameter cannot switch the caller's identity.
    assert client.get("/query_result", params={"task_id": task, "key": "rotated-secret"},
                      headers=headers("bob")).status_code == 404
    assert client.get("/tasks?key=rotated-secret", headers=headers("invalid")).status_code == 401


@pytest.mark.parametrize("raw", ['bad-json', '[]', 'null', '{"a": ""}', '{"a": 1}',
                                  '{"": "secret"}', '{"__legacy__": "secret"}',
                                  '{"a": "same", "b": "same"}', '{"a": "legacy-secret"}'])
def test_invalid_key_configuration_fails_closed(raw):
    with pytest.raises(ValueError):
        api._load_api_keys(raw, "legacy-secret")


def test_all_users_share_one_fifo_worker(client, monkeypatch):
    started, release = Event(), Event()
    order = []

    def run(job):
        order.append(job.state.task_id)
        if len(order) == 1:
            started.set()
            assert release.wait(5), "test did not release first job"

    monkeypatch.setattr(pipeline.Job, "_run", run)
    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(api, "_executor", executor)
        try:
            first = submit(client, "alice")
            assert started.wait(5)
            second = submit(client, "bob", upload=True)
            third = submit(client, "alice")
            assert order == [first]
            assert api._jobs[second].state.status == api._jobs[third].state.status == "queued"
        finally:
            release.set()
        for future in api._job_futures.values():
            assert future.result(timeout=5).status == "done"
    assert order == [first, second, third]
