"""Exercise real worker pipes without loading neural networks."""

import json
import shutil
from concurrent.futures import ThreadPoolExecutor

import pytest

from dsapi import tools


@pytest.fixture
def worker(tmp_path, monkeypatch):
    tools.close_workers()
    shutil.copy(tools.RUNNERS / "worker.py", tmp_path / "worker.py")
    (tmp_path / "fake_runner.py").write_text('''
import json
import os
import time
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
torch = SimpleNamespace(set_grad_enabled=lambda _: None)
print("noisy import")
@lru_cache(maxsize=None)
def load_models(key):
    return object()
def main(key, output):
    print("python log " + key)
    os.write(1, b"native stdout\\n")
    os.write(2, b"native stderr\\n")
    if key == "crash":
        os._exit(7)
    if key == "hang":
        time.sleep(60)
    if key == "error":
        raise ValueError("bad input")
    load_models(key)
    Path(output).write_text(json.dumps({"pid": os.getpid(), "key": key}))
''')
    monkeypatch.setattr(tools, "RUNNERS", tmp_path)
    monkeypatch.setattr(tools.settings, "VENDOR", tmp_path)

    def run(key, name):
        output, log = tmp_path / f"{name}.json", tmp_path / f"{name}.log"
        tools._run(".", "fake_runner.py", key, str(output), log=log)
        return json.loads(output.read_text()), log.read_text()

    yield run
    tools.close_workers()


def test_reuses_model_and_routes_each_jobs_logs(worker):
    first, log1 = worker("zh", "one")
    second, log2 = worker("zh", "two")
    assert first["pid"] == second["pid"]
    assert tools.worker_status()[0]["cache_loads"] == 1
    assert "noisy import" in log1 and "noisy import" not in log2
    assert "native stdout" in log2 and "native stderr" in log2
    worker("ja", "three")
    worker("zh", "four")
    assert tools.worker_status()[0]["cache_entries"] == 2
    assert tools.worker_status()[0]["cache_loads"] == 2


def test_bad_input_does_not_discard_loaded_models(worker):
    first, _ = worker("zh", "one")
    with pytest.raises(RuntimeError, match="bad input"):
        worker("error", "bad")
    second, _ = worker("zh", "two")
    assert first["pid"] == second["pid"]
    assert tools.worker_status()[0]["cache_loads"] == 1


def test_worker_crash_is_reported_and_next_call_restarts(worker):
    first, _ = worker("zh", "one")
    with pytest.raises(RuntimeError, match="without a reply"):
        worker("crash", "bad")
    second, _ = worker("zh", "two")
    assert first["pid"] != second["pid"]


def test_concurrent_requests_are_serialized_and_shutdown_reaps_child(worker):
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda i: worker("zh", str(i))[0], range(8)))
    assert len({item["pid"] for item in results}) == 1
    assert tools.worker_status()[0]["cache_loads"] == 1
    proc = next(iter(tools._workers.values())).proc
    tools.close_workers()
    assert proc.poll() == 0
    assert tools.worker_status() == []


def test_timeout_discards_worker_and_next_call_recovers(worker, monkeypatch):
    first, _ = worker("zh", "one")
    monkeypatch.setenv("DSAPI_WORKER_TIMEOUT", "0.05")
    with pytest.raises(RuntimeError, match="exceeded"):
        worker("hang", "bad")
    monkeypatch.delenv("DSAPI_WORKER_TIMEOUT")
    second, _ = worker("zh", "two")
    assert second["pid"] != first["pid"]


def test_api_shutdown_closes_workers(worker):
    from fastapi.testclient import TestClient
    from dsapi.api import app

    with TestClient(app):
        worker("zh", "one")
        proc = next(iter(tools._workers.values())).proc
    assert proc.poll() == 0
