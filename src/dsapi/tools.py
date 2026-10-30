"""Subprocess wrappers around vendored SOME and SOFA (their packages clash)."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import settings

RUNNERS = Path(__file__).parent / "runners"


def _run(vendor: str, script: str, *args: str, log: Path) -> None:
    with log.open("a") as f:
        proc = subprocess.run([sys.executable, str(RUNNERS / script), *args], cwd=settings.VENDOR / vendor,
                              stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        tail = log.read_text(errors="replace").splitlines()[-15:]
        raise RuntimeError(f"{script} failed:\n" + "\n".join(tail))


def extract_notes_f0(wav: Path, out_dir: Path) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """SOME notes (absolute seconds) and f0/uv at HOP resolution."""
    out_json, out_f0 = out_dir / "notes.json", out_dir / "f0.npy"
    # SOME's conformer runs fine on MPS, but its STFT path is CPU/CUDA only.
    device = "cuda" if settings.torch_device() == "cuda" else "cpu"
    _run("SOME", "some_runner.py", str(settings.SOME_CKPT),
         str(settings.RMVPE_CKPT) if settings.RMVPE_CKPT.is_file() else "-", str(wav.resolve()),
         str(out_json.resolve()), str(out_f0.resolve()), device, log=out_dir / "some.log")
    f0, uv = np.load(out_f0)
    return json.loads(out_json.read_text())["notes"], f0, uv.astype(bool)


def align(folder: Path, out_dir: Path, ckpt: Path = settings.SOFA_CKPT, dictionary: Path = settings.SOFA_DICT) -> dict:
    """SOFA forced alignment for every <name>.wav/.lab in folder."""
    out_json = out_dir / "sofa.json"
    accel = "gpu" if settings.torch_device() == "cuda" else "cpu"
    # The runner's cwd is vendor/SOFA, so every path must be absolute.
    _run("SOFA", "sofa_runner.py", str(Path(ckpt).resolve()), str(folder.resolve()), str(Path(dictionary).resolve()),
         str(out_json.resolve()), accel, log=out_dir / "sofa.log")
    return json.loads(out_json.read_text())
