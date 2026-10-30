"""Measure resident models on macOS using physical footprint, including GPU memory.

Runs a saved job's local input twice, plus optional multilingual alignment.
Samples the model host and its persistent workers together every 100 ms;
short-lived ffmpeg processes are excluded. No caches are explicitly emptied.
"""

import argparse
import ctypes
import gc
import json
import os
import platform
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path


class RUsage(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64) for name in (
            "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins",
            "wired_size", "resident_size", "phys_footprint", "proc_start_abstime", "proc_exit_abstime",
        )
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("job", type=Path)
    p.add_argument("--multilingual-job", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--backend", choices=("cpu", "webgpu", "auto"), default="auto")
    args = p.parse_args()
    if sys.platform != "darwin":
        p.error("Physical footprint measurement requires macOS")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["DSAPI_WORK"] = str(args.output)
    os.environ["DSAPI_ONNX_BACKEND"] = args.backend

    # Include normal HTTP application imports in the host's baseline.
    from dsapi import api, onnx_backend, settings, tools
    from dsapi.pipeline import Job, JobSpec

    libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    libproc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    libproc.proc_pid_rusage.restype = ctypes.c_int
    samples, marks, runs = [], [], []
    stop = threading.Event()
    stage = "baseline"
    started = time.monotonic()

    def sample():
        processes = [{"vendor": "host", "pid": os.getpid()}, *tools.worker_status()]
        usage = []
        for proc in processes:
            data = RUsage()
            if libproc.proc_pid_rusage(proc["pid"], 0, ctypes.byref(data)) == 0:
                usage.append({"name": proc["vendor"], "pid": proc["pid"],
                              "footprint_bytes": data.phys_footprint, "rss_bytes": data.resident_size})
        return {"seconds": time.monotonic() - started, "stage": stage, "processes": usage,
                "footprint_bytes": sum(x["footprint_bytes"] for x in usage)}

    def monitor():
        while not stop.is_set():
            samples.append(sample())
            stop.wait(0.1)

    def mark(label):
        gc.collect()
        item = {"label": label, **sample(), "workers": tools.worker_status()}
        marks.append(item)
        print("MEMORY", json.dumps(item), flush=True)

    spec = json.loads((args.job / "job.json").read_text())["spec"]
    if not spec.get("audio_path") or not Path(spec["audio_path"]).is_file():
        p.error("Use a saved upload job with its local source audio still available")
    spec["audio_path"] = str(Path(spec["audio_path"]).resolve())
    spec["lrc_path"] = str(Path(spec["lrc_path"]).resolve())
    spec["modes"] = ["mimic", "score"]

    class MeasuredJob(Job):
        @contextmanager
        def stage(self, name):
            nonlocal stage
            stage = self.state.task_id + "/" + name
            with super().stage(name):
                yield
            mark(stage)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        mark("baseline")
        for iteration in range(2):
            job = MeasuredJob(f"run-{iteration}", JobSpec(**spec))
            result = job.run()
            if result.status != "done":
                raise RuntimeError(result.error)
            runs.append({"iteration": iteration, "timings": result.timings, "window": result.window,
                         "outputs": result.outputs, "workers": tools.worker_status()})
            stage = f"idle-{iteration}-mandarin"
            time.sleep(2)
            mark(stage)
            if args.multilingual_job:
                stage = f"multilingual-{iteration}"
                out = args.output / stage
                out.mkdir()
                start = time.perf_counter()
                aligned = tools.align(args.multilingual_job / "segments", out,
                                      settings.SOFA_CKPT_MULTI, args.multilingual_job / "sofa_dict.txt")
                if not aligned or any("error" in x for x in aligned.values()):
                    raise RuntimeError(f"Multilingual alignment failed: {aligned}")
                runs[-1]["multilingual_seconds"] = time.perf_counter() - start
                stage = f"idle-{iteration}-all"
                time.sleep(2)
                mark(stage)
        stage = "idle-final"
        time.sleep(5)
        mark(stage)
    finally:
        stop.set()
        thread.join()
        report = {"platform": platform.platform(), "processor": platform.processor(),
                  "backend": onnx_backend.backend_status(), "sample_interval_seconds": 0.1,
                  "metric": "Sum of macOS physical footprint of host + persistent model workers; excludes ffmpeg",
                  "source_job": str(args.job.resolve()), "runs": runs, "marks": marks,
                  "peak": max(samples, key=lambda s: s["footprint_bytes"]), "samples": samples}
        (args.output / "results.json").write_text(json.dumps(report, indent=2))
        print("PEAK", json.dumps(report["peak"]), flush=True)
        tools.close_workers()


if __name__ == "__main__":
    main()
