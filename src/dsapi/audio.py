"""ffmpeg-backed audio helpers and source separation."""

import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from . import settings
from .vocal_protection import detect_lead_dropouts


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         check=True, capture_output=True, text=True).stdout
    return float(out.strip())


def cut(src: Path, dst: Path, start: float, end: float, mono: bool = False) -> Path:
    args = ["-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(src), "-ar", str(settings.SAMPLE_RATE)]
    if mono:
        args += ["-ac", "1"]
    ffmpeg(*args, str(dst))
    return dst


def load(path: Path) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    assert sr == settings.SAMPLE_RATE, f"{path}: {sr}Hz"
    return wav.mean(axis=1)


def save(path: Path, wav: np.ndarray) -> Path:
    sf.write(path, wav, settings.SAMPLE_RATE)
    return path


def to_mp3(src: Path, dst: Path, loudnorm: bool = True) -> Path:
    args = ["-i", str(src)]
    if loudnorm:
        args += ["-af", "loudnorm=I=-16:TP=-1.5:LRA=11"]
    ffmpeg(*args, "-ar", str(settings.SAMPLE_RATE), "-b:a", "192k", str(dst))
    return dst


_separators: dict[str, object] = {}


def _separate(model: str, src: Path, out_dir: Path) -> dict[str, Path]:
    from audio_separator.separator import Separator

    sep = _separators.get(model)
    if sep is None:
        sep = Separator(model_file_dir=str(settings.SEPARATOR_MODELS), output_dir=str(out_dir), output_format="WAV",
                        sample_rate=settings.SAMPLE_RATE, use_autocast=False, use_native_fp16=False,
                        use_torch_compile=settings.SEPARATOR_COMPILE,
                        # 0.48 uses a window count. One preserves the previous
                        # 8-second stride for our two 8-second RoFormer models.
                        mdxc_params={"segment_size": 256, "override_model_segment_size": False,
                                     "batch_size": 1, "overlap": 1, "pitch_shift": 0})
        sep.load_model(model_filename=model)
        _separators[model] = sep
    sep.output_dir = str(out_dir)
    if getattr(sep, "model_instance", None) is not None:
        sep.model_instance.output_dir = str(out_dir)
    stems = {}
    for name in sep.separate(str(src)):
        path = out_dir / Path(name).name
        stem = path.stem.rsplit("_(", 1)[-1].split(")")[0].lower()
        stems[stem] = path
    return stems


def separate_vocals(src: Path, out_dir: Path) -> tuple[Path, Path]:
    """Return (lead_vocals.wav, instrumental.wav)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stems = _separate(settings.VOCALS_MODEL, src, out_dir)
    vocals, inst = stems["vocals"], stems["instrumental"]
    if settings.KARAOKE_MODEL:
        kara = _separate(settings.KARAOKE_MODEL, vocals, out_dir / "karaoke")
        vocals = protect_lead(vocals, kara["vocals"], out_dir)
    lead = out_dir / "lead.wav"
    ffmpeg("-i", str(vocals), "-ac", "1", "-ar", str(settings.SAMPLE_RATE), str(lead))
    return lead, inst


def protect_lead(original: Path, separated: Path, out_dir: Path) -> Path:
    """Use the entire first-stage vocal if the second stage introduces dropouts."""
    report = {"enabled": settings.KARAOKE_PROTECTION, "mode": "whole_clip",
              "fallback": False, "reason": "disabled", "detected_intervals": [],
              "fallback_seconds": 0.0}
    result = separated
    if settings.KARAOKE_PROTECTION:
        vocals, rate = sf.read(original, dtype="float32", always_2d=True)
        lead, lead_rate = sf.read(separated, dtype="float32", always_2d=True)
        if rate != lead_rate:
            raise ValueError("Vocal protection requires matching sample rates")
        report = detect_lead_dropouts(vocals, lead, rate)
        if report["fallback"]:
            result = original
    report.update(original=str(original), separated=str(separated), output=str(result))
    (out_dir / "lead_protection.json").write_text(json.dumps(report, indent=2))
    return result
