"""Persistent workers for note extraction and lyric alignment."""

import atexit
import json
import os
import select
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

from . import settings

RUNNERS = Path(__file__).parent / "runners"
_workers: dict[tuple[str, str], "_Worker"] = {}
_workers_lock = threading.Lock()


class _Worker:
    def __init__(self, vendor: str, script: str):
        self.vendor, self.script = vendor, script
        self.lock = threading.Lock()
        self.proc: subprocess.Popen | None = None
        self.info: dict = {}

    def close(self) -> None:
        with self.lock:
            self._stop()

    def _stop(self) -> None:
        if self.proc is not None:
            proc, self.proc = self.proc, None
            try:
                proc.stdin.close()
            except OSError:
                pass  # A crashed child may have closed its pipe during a write.
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            proc.stdout.close()
            self.info = {}

    def run(self, args: tuple[str, ...], log: Path) -> None:
        log = log.resolve()
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                self._stop()
                with log.open("a") as stderr:
                    self.proc = subprocess.Popen(
                        [sys.executable, "-u", str(RUNNERS / "worker.py"), Path(self.script).stem],
                        cwd=settings.VENDOR / self.vendor, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=stderr, text=True, bufsize=1,
                    )
            try:
                self.proc.stdin.write(json.dumps({"args": args, "log": str(log)}) + "\n")
                self.proc.stdin.flush()
                timeout = float(os.environ.get("DSAPI_WORKER_TIMEOUT", "600"))
                if not select.select([self.proc.stdout], [], [], timeout)[0]:
                    raise TimeoutError(f"worker exceeded {timeout:g}s")
                line = self.proc.stdout.readline()
                if not line:
                    raise RuntimeError("worker exited without a reply")
                result = json.loads(line)
            except (OSError, ValueError, RuntimeError) as exc:
                self._stop()
                raise RuntimeError(f"{self.script}: {exc}\n{_log_tail(log)}") from exc
            if not result["ok"]:
                raise RuntimeError(f"{self.script}: {result['error']}\n{_log_tail(log)}")
            self.info = {k: v for k, v in result.items() if k != "ok"}


def _log_tail(log: Path) -> str:
    return "\n".join(log.read_text(errors="replace").splitlines()[-15:])


def worker_status() -> list[dict]:
    """Snapshot without waiting for an in-flight inference or loading a model."""
    with _workers_lock:
        return [{**w.info, "vendor": w.vendor, "runner": w.script, "pid": proc.pid,
                 "alive": proc.poll() is None}
                for w in _workers.values() for proc in [w.proc] if proc is not None]


def close_workers() -> None:
    with _workers_lock:
        workers = list(_workers.values())
        _workers.clear()
    for worker in workers:
        worker.close()


atexit.register(close_workers)


def _run(vendor: str, script: str, *args: str, log: Path) -> None:
    with _workers_lock:
        worker = _workers.setdefault((vendor, script), _Worker(vendor, script))
    worker.run(args, log)


def extract_some_notes_f0(wav: Path, out_dir: Path) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """SOME notes (absolute seconds) and f0/uv at HOP resolution."""
    out_json, out_f0 = out_dir / "notes.json", out_dir / "f0.npy"
    # SOME's conformer runs fine on MPS, but its STFT path is CPU/CUDA only.
    device = "cuda" if settings.torch_device() == "cuda" else "cpu"
    _run("SOME", "some_runner.py", str(settings.SOME_CKPT.resolve()),
         str(settings.RMVPE_CKPT.resolve()) if settings.RMVPE_CKPT.is_file() else "-", str(wav.resolve()),
         str(out_json.resolve()), str(out_f0.resolve()), device, log=out_dir / "some.log")
    f0, uv = np.load(out_f0)
    return json.loads(out_json.read_text())["notes"], f0, uv.astype(bool)


def note_backend() -> str:
    if settings.NOTE_EXTRACTOR not in ("game-medium", "some"):
        raise ValueError("DSAPI_NOTE_EXTRACTOR must be game-medium or some")
    return settings.NOTE_EXTRACTOR


def extract_f0(wav: Path, out_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    device = "cuda" if settings.torch_device() == "cuda" else "cpu"
    output = out_dir / "f0.npy"
    _run("SOME", "f0_runner.py",
         str(settings.RMVPE_CKPT.resolve()) if settings.RMVPE_CKPT.is_file() else "-",
         str(wav.resolve()), str(output.resolve()), device, log=out_dir / "f0.log")
    f0, uv = np.load(output)
    return f0, uv.astype(bool)


def extract_game_notes(wav: Path, out_dir: Path, phrases: list, lang: str) -> list[dict]:
    regions = [{"start": p.phones[0][1], "end": p.phones[-1][2],
                "boundaries": [t for s in p.syllables for t in (s.start, s.end)],
                **({"language": p.syllables[0].lang} if lang == "mixed" else {})} for p in phrases]
    regions_path, output = out_dir / "game_regions.json", out_dir / "notes.json"
    regions_path.write_text(json.dumps(regions))
    _run(".", "game_runner.py", str(settings.GAME_MODEL.resolve()), str(wav.resolve()),
         str(regions_path.resolve()), str(output.resolve()), lang, str(settings.GAME_STEPS),
         log=out_dir / "game.log")
    return json.loads(output.read_text())["notes"]


def alignment_backend(lang: str = "zh") -> str:
    if settings.ALIGNER not in ("hubertfa", "sofa"):
        raise ValueError("DSAPI_ALIGNER must be hubertfa or sofa")
    if lang == "ja":
        if settings.JA_ALIGNER not in ("hubertfa", "sofa"):
            raise ValueError("DSAPI_JA_ALIGNER must be hubertfa or sofa")
        return settings.JA_ALIGNER
    if lang in ("en", "mixed"):
        return "hubertfa"
    if lang == "zh":
        return settings.ALIGNER
    raise ValueError(f"Unsupported alignment language: {lang}")


def align(folder: Path, out_dir: Path, ckpt: Path | None = None, dictionary: Path | None = None,
          *, backend: str | None = None, lang: str = "zh") -> dict:
    """Align <name>.wav/.lab pairs; retain the legacy artifact name for saved-job tools.

    Explicit checkpoints without a backend retain their original engine, including
    old SOFA evaluation jobs. New requests use the configured language default.
    """
    backend = backend or (("hubertfa" if Path(ckpt).suffix == ".onnx" else "sofa")
                          if ckpt is not None else alignment_backend(lang))
    out_json = out_dir / "sofa.json"
    if backend == "hubertfa":
        ckpt = ckpt or settings.HUBERTFA_MODEL
        dictionaries = {"en": settings.HUBERTFA_MODEL.parent / "ds_cmudict-07b.txt",
                        "ja": settings.HUBERTFA_MODEL.parent / "japanese_dict_full.txt",
                        "zh": settings.HUBERTFA_DICT}
        if dictionary is None and lang == "mixed":
            raise ValueError("Mixed-language alignment requires a phoneme-prefixed dictionary")
        dictionary = dictionary or dictionaries[lang]
        _run("HubertFA", "hubertfa_runner.py", str(Path(ckpt).resolve()), str(folder.resolve()),
             str(Path(dictionary).resolve()), str(out_json.resolve()), lang, log=out_dir / "hubertfa.log")
        return json.loads(out_json.read_text())
    if backend != "sofa":
        raise ValueError(f"Unknown alignment backend: {backend}")
    ckpt = ckpt or (settings.SOFA_CKPT if lang == "zh" else settings.SOFA_CKPT_MULTI)
    dictionary = dictionary or settings.SOFA_DICT
    accel = "gpu" if settings.torch_device() == "cuda" else "cpu"
    # The runner's cwd is vendor/SOFA, so every path must be absolute.
    _run("SOFA", "sofa_runner.py", str(Path(ckpt).resolve()), str(folder.resolve()), str(Path(dictionary).resolve()),
         str(out_json.resolve()), accel, log=out_dir / "sofa.log")
    return json.loads(out_json.read_text())
