import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from dsapi import onnx_backend as backend


@pytest.mark.parametrize("available,failures,expected,attempts", [
    (["CUDAExecutionProvider"], [], "cuda", ["cuda"]),
    (["CUDAExecutionProvider"], ["cuda"], "webgpu", ["cuda", "webgpu"]),
    (["CoreMLExecutionProvider"], [], "webgpu", ["webgpu"]),
    ([], ["webgpu"], "cpu", ["webgpu", "cpu"]),
])
def test_auto_priority_and_observed_status(monkeypatch, available, failures, expected, attempts):
    monkeypatch.delenv("DSAPI_ONNX_BACKEND", raising=False)
    monkeypatch.setattr(backend, "_sessions", {})
    monkeypatch.setattr(backend.ort, "get_available_providers", lambda: available)
    calls = []

    def create(path, candidate):
        calls.append(candidate)
        if candidate in failures:
            raise RuntimeError("unavailable")
        return SimpleNamespace(get_providers=lambda: [candidate])

    monkeypatch.setattr(backend, "_create_session", create)
    assert backend.backend_status()["sessions"] == []
    backend.create_session(Path("model.onnx"))
    assert calls == attempts
    status = backend.backend_status()
    assert status["configured"] == "auto"
    assert status["sessions"][0]["backend"] == expected
    assert len(status["sessions"][0]["fallback_reasons"]) == len(failures)


def test_explicit_unavailable_cuda_fails(monkeypatch):
    monkeypatch.setenv("DSAPI_ONNX_BACKEND", "cuda")
    monkeypatch.setattr(backend.ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    with pytest.raises(RuntimeError, match="unavailable"):
        backend.create_session(Path("model.onnx"))


def test_invalid_backend_fails(monkeypatch):
    monkeypatch.setenv("DSAPI_ONNX_BACKEND", "typo")
    with pytest.raises(ValueError, match="DSAPI_ONNX_BACKEND"):
        backend.create_session(Path("model.onnx"))


def test_webgpu_rejects_silent_cpu_fallback(monkeypatch):
    monkeypatch.setenv("DSAPI_ONNX_BACKEND", "webgpu")
    monkeypatch.setenv("DSAPI_WEBGPU_DEVICE", "0")
    monkeypatch.setattr(backend, "_devices", lambda: [object()])
    monkeypatch.setattr(backend.ort, "SessionOptions", lambda: SimpleNamespace(add_provider_for_devices=lambda *a: None))
    monkeypatch.setattr(backend.ort, "InferenceSession", lambda *a, **kw: SimpleNamespace(
        disable_fallback=lambda: None, get_providers=lambda: ["CPUExecutionProvider"]))
    with pytest.raises(RuntimeError, match="not activated"):
        backend.create_session(Path("model.onnx"))


def test_webgpu_sessions_share_a_run_lock():
    entered, release, attempting, second_entered = (threading.Event() for _ in range(4))

    class First:
        def run(self, *args):
            entered.set()
            assert release.wait(3)
            return "first"

    class Second:
        def run(self, *args):
            second_entered.set()
            return "second"

    first, second = backend._WebGpuSession(First()), backend._WebGpuSession(Second())

    def run_second():
        attempting.set()
        return second.run(None, {})

    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(first.run, None, {})
        assert entered.wait(3)
        b = pool.submit(run_second)
        try:
            assert attempting.wait(3)
            assert not second_entered.wait(0.05)
        finally:
            release.set()
        assert a.result() == "first"
        assert b.result() == "second"
