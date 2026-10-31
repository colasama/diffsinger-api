"""ffmpeg-backed audio helpers and source separation."""

import json
import subprocess
import tempfile
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


def save(path: Path, wav: np.ndarray, *, subtype: str = "PCM_16") -> Path:
    sf.write(path, wav, settings.SAMPLE_RATE, subtype=subtype)
    return path


def to_mp3(src: Path, dst: Path, loudnorm: bool = True) -> Path:
    args = ["-i", str(src)]
    if loudnorm:
        args += ["-af", "loudnorm=I=-16:TP=-1.5:LRA=11"]
    ffmpeg(*args, "-ar", str(settings.SAMPLE_RATE), "-b:a", "192k", str(dst))
    return dst


def mix_accompaniment(vocal: Path, instrumental: Path, reference: Path, dst: Path) -> dict:
    """Mix aligned stems in blocks, matching active vocal RMS with bounded gain.

    The float intermediate cannot clip before final loudness/true-peak control.
    Normalize the complete mix once so downloadable parts retain its balance.
    """
    sr = settings.SAMPLE_RATE
    block, frame = sr * 10, max(1, sr // 20)
    info = sf.info(vocal)
    if info.samplerate != sr or info.channels != 1 or not info.frames:
        raise ValueError("Mixed vocals must be nonempty mono audio at the synthesis sample rate")
    n = info.frames
    with tempfile.TemporaryDirectory(prefix="mix-", dir=dst.parent) as temp:
        folder = Path(temp)
        inst_info = sf.info(instrumental)
        if inst_info.channels not in (1, 2):
            raise ValueError("Accompaniment must be mono or stereo")
        if inst_info.samplerate != sr:
            resampled = folder / "instrumental.wav"
            ffmpeg("-i", str(instrumental), "-ar", str(sr), "-c:a", "pcm_f32le", str(resampled))
            instrumental = resampled
        ref_info = sf.info(reference)
        if ref_info.samplerate != sr or ref_info.channels != 1 or ref_info.frames != n:
            raise ValueError("Reference vocals do not share the synthesized vocal timeline")
        delta = sf.info(instrumental).frames - n
        if abs(delta) > round(sr * .05):
            raise ValueError("Accompaniment duration does not match the vocal timeline")

        ref_energy = vocal_energy = 0.
        active_samples = 0
        with sf.SoundFile(vocal) as voice, sf.SoundFile(reference) as ref:
            for start in range(0, n, block):
                count = min(block, n - start)
                v, r = voice.read(count, dtype="float64"), ref.read(count, dtype="float64")
                if not np.isfinite(v).all() or not np.isfinite(r).all():
                    raise ValueError("Nonfinite vocal samples")
                for a in range(0, count, frame):
                    vv, rr = v[a:a+frame], r[a:a+frame]
                    vp, rp = np.mean(vv * vv), np.mean(rr * rr)
                    # Ignore silence and separation noise; never amplify a gap.
                    if rp > 10 ** (-45 / 10) and vp > 10 ** (-60 / 10):
                        vocal_energy += vp * len(vv)
                        ref_energy += rp * len(rr)
                        active_samples += len(vv)
        if not active_samples:
            raise ValueError("No overlapping active vocals available for mix balance")
        gain_db = float(np.clip(10 * np.log10(ref_energy / vocal_energy), -12, 12))
        gain = 10 ** (gain_db / 20)
        raw = folder / "mix.wav"
        with sf.SoundFile(vocal) as voice, sf.SoundFile(instrumental) as inst, \
                sf.SoundFile(raw, "w", sr, 2, subtype="FLOAT") as output:
            for start in range(0, n, block):
                count = min(block, n - start)
                v = voice.read(count, dtype="float32")
                bed = inst.read(count, dtype="float32", always_2d=True)
                if not np.isfinite(bed).all():
                    raise ValueError("Nonfinite accompaniment samples")
                if len(bed) < count:
                    bed = np.pad(bed, ((0, count-len(bed)), (0, 0)))
                if bed.shape[1] == 1:
                    bed = np.repeat(bed, 2, axis=1)
                output.write(bed + (v * gain)[:, None])
        ffmpeg("-i", str(raw), "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
               "-ar", str(sr), "-c:a", "pcm_f32le", str(dst))
    return {"vocal_gain_db": gain_db, "active_samples": active_samples,
            "instrumental_frame_adjustment": -delta, "sample_rate": sr,
            "target_lufs": -16, "true_peak_db": -1.5}


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


def separate_long_vocals(src: Path, out_dir: Path, chunk_seconds: int = 40) -> tuple[Path, Path]:
    """Bound separator activations; stitch overlapping stems on the sample grid."""
    info = sf.info(src)
    if info.duration <= 60:
        return separate_vocals(src, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sr, count = settings.SAMPLE_RATE, info.frames
    step, overlap = chunk_seconds * sr, 2 * sr
    # Accumulators are disk-backed; only a chunk of audio is resident at once.
    accum = np.memmap(out_dir / "stems.tmp", mode="w+", dtype="float32", shape=(count, 3))
    weights = np.memmap(out_dir / "weights.tmp", mode="w+", dtype="float32", shape=(count,))
    accum[:] = 0
    weights[:] = 0
    try:
        with sf.SoundFile(src) as source:
            for index, core in enumerate(range(0, count, step)):
                a, b = max(0, core - overlap), min(count, core + step + overlap)
                folder = out_dir / f"chunk_{index:04}"
                folder.mkdir(exist_ok=True)
                source.seek(a)
                chunk = folder / "input.wav"
                sf.write(chunk, source.read(b - a, dtype="float32", always_2d=True), sr, subtype="FLOAT")
                lead, inst = separate_vocals(chunk, folder / "sep")
                vocal = load(lead)
                instrumental, rate = sf.read(inst, dtype="float32", always_2d=True)
                if rate != sr or min(len(vocal), len(instrumental)) < b - a:
                    raise ValueError("Separator truncated a long-audio chunk")
                instrumental = np.repeat(instrumental, 2, axis=1) if instrumental.shape[1] == 1 else instrumental[:, :2]
                w = np.ones(b - a, np.float32)
                if a > 0:
                    ramp = min(2*overlap, len(w))
                    w[:ramp] = np.linspace(0, 1, ramp)
                if b < count:
                    ramp = min(2*overlap, len(w))
                    w[-ramp:] = np.linspace(1, 0, ramp)
                accum[a:b, 0] += vocal[:b-a] * w
                accum[a:b, 1:] += instrumental[:b-a] * w[:, None]
                weights[a:b] += w
        lead, inst = out_dir / "lead.wav", out_dir / "instrumental.wav"
        with sf.SoundFile(lead, "w", sr, 1, subtype="FLOAT") as lv, sf.SoundFile(inst, "w", sr, 2, subtype="FLOAT") as iv:
            for a in range(0, count, step):
                b = min(count, a + step)
                part = accum[a:b] / np.maximum(weights[a:b, None], 1e-6)
                lv.write(part[:, 0])
                iv.write(part[:, 1:])
        return lead, inst
    finally:
        del accum, weights
        (out_dir / "stems.tmp").unlink(missing_ok=True)
        (out_dir / "weights.tmp").unlink(missing_ok=True)
