"""ONNX backends: prefer CUDA, then native WebGPU, then CPU."""

import logging
import os
import threading
from pathlib import Path

import onnxruntime as ort

_log = logging.getLogger(__name__)
_webgpu_lock = threading.RLock()
_webgpu_devices = None
_status_lock = threading.Lock()
_sessions: dict[str, dict] = {}

# ImportedFixed's diffusion loops pass CPU tensors to these Gather nodes. The native
# WebGPU EP 0.4.0 otherwise fails with "All inputs must be tensors on WebGPU
# buffers". Keep just these nodes on CPU, not the surrounding neural network.
_CPU_NODES = "\n".join((
    "/diffusion/velocity_fn/Gather",
    "/pitch_predictor/velocity_fn/Gather",
))


def _devices():
    global _webgpu_devices
    # Caller holds _webgpu_lock: registration must happen once per process.
    if _webgpu_devices is None:
        try:
            import onnxruntime_ep_webgpu as ep
        except ImportError as exc:
            raise RuntimeError("WebGPU plugin is missing; run uv sync") from exc
        ort.register_execution_provider_library("dsapi_webgpu", ep.get_library_path())
        _webgpu_devices = [d for d in ort.get_ep_devices() if d.ep_name == ep.get_ep_name()]
    if not _webgpu_devices:
        raise RuntimeError("WebGPU plugin loaded but no GPU device was found")
    return _webgpu_devices


class _WebGpuSession:
    """Prevent concurrent WebGPU Run calls, including across API requests.

    Native crashes cannot be caught as Python exceptions. Keep this backend
    serialized and fail visibly on runtime errors rather than claiming GPU
    success after silently rebuilding a CPU session.
    """

    def __init__(self, session):
        self._session = session

    def run(self, output_names, input_feed, run_options=None):
        with _webgpu_lock:
            return self._session.run(output_names, input_feed, run_options)

    def get_providers(self):
        return self._session.get_providers()

    def get_outputs(self):
        return self._session.get_outputs()


OnnxSession = ort.InferenceSession | _WebGpuSession


def configured_backend() -> str:
    return os.environ.get("DSAPI_ONNX_BACKEND", "auto").strip().lower()


def backend_status() -> dict:
    """Report observed sessions without loading models or probing the GPU."""
    with _status_lock:
        return {"configured": configured_backend(), "priority": ["cuda", "webgpu", "cpu"],
                "sessions": list(_sessions.values())}


def create_session(path: Path) -> OnnxSession:
    backend = configured_backend()
    if backend not in {"auto", "cpu", "cuda", "webgpu"}:
        raise ValueError("DSAPI_ONNX_BACKEND must be auto, cpu, cuda, or webgpu")
    candidates = [backend]
    if backend == "auto":
        candidates = (["cuda"] if "CUDAExecutionProvider" in ort.get_available_providers() else []) + ["webgpu", "cpu"]
    failures = []
    for candidate in candidates:
        try:
            session = _create_session(path, candidate)
        except Exception as exc:
            if backend != "auto" or candidate == "cpu":
                raise
            reason = f"{candidate}: {exc}"
            failures.append(reason)
            _log.warning("ONNX %s initialization failed, trying next backend: %s", path.name, reason)
            continue
        with _status_lock:
            _sessions[str(path)] = {"model": path.name, "backend": candidate,
                                    "providers": session.get_providers(), "fallback_reasons": failures}
        return session
    raise RuntimeError("No ONNX backend could be initialized")


def _create_session(path: Path, backend: str) -> OnnxSession:
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    if backend == "webgpu":
        if tuple(int(v) for v in ort.__version__.split(".")[:2]) < (1, 31):
            raise RuntimeError("WebGPU requires ONNX Runtime >= 1.31")
        with _webgpu_lock:
            devices = _devices()
            index = int(os.environ.get("DSAPI_WEBGPU_DEVICE", "0"))
            if not 0 <= index < len(devices):
                raise ValueError(f"DSAPI_WEBGPU_DEVICE must be between 0 and {len(devices) - 1}")
            opts.add_provider_for_devices([devices[index]], {"forceCpuNodeNames": _CPU_NODES})
            session = ort.InferenceSession(str(path), sess_options=opts)
            session.disable_fallback()
            if "WebGpuExecutionProvider" not in session.get_providers():
                raise RuntimeError(f"WebGPU was requested but not activated for {path.name}")
            _log.info("ONNX %s: WebGPU device %s (%s), with CPU node fallback",
                      path.name, index, devices[index].device.vendor)
            return _WebGpuSession(session)
    providers = ["CPUExecutionProvider"]
    if backend == "cuda":
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
        elif backend == "cuda":
            raise RuntimeError("CUDA was requested but CUDAExecutionProvider is unavailable")
    session = ort.InferenceSession(str(path), sess_options=opts, providers=providers)
    if backend == "cuda" and "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError(f"CUDA was requested but not activated for {path.name}")
    return session
