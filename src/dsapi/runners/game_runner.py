"""GAME v1.0.3 ONNX inference, with aligned vowel boundaries and cached sessions."""
import json
from functools import lru_cache
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort
import torch


@lru_cache(maxsize=None)
def load_models(root: str):
    root = Path(root)
    config = json.loads((root / "config.json").read_text())
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "CUDAExecutionProvider")
    sessions = {name: ort.InferenceSession(str(root / f"{name}.onnx"), options, providers=providers)
                for name in ("encoder", "segmenter", "estimator", "dur2bd", "bd2dur")}
    return config, sessions


def transcribe(waveform, config, sessions, regions, language, steps):
    if steps < 1:
        raise ValueError("GAME steps must be positive")
    sr = config["samplerate"]
    notes = []
    for region in regions:
        region_language = region.get("language", language)
        if region_language not in config["languages"]:
            raise ValueError(f"Unsupported GAME language: {region_language}")
        a = max(0, round(region["start"] * sr))
        b = min(len(waveform), round(region["end"] * sr))
        if b <= a:
            raise ValueError("GAME region has no audio")
        offset = a / sr
        duration = np.array([(b - a) / sr], np.float32)
        xs, xe, mask = sessions["encoder"].run(None, {
            "waveform": waveform[None, a:b].astype(np.float32), "duration": duration})
        bounds = [0., float(duration[0])]
        bounds.extend(t - offset for t in region["boundaries"] if 0 < t - offset < duration[0])
        durations = np.diff(sorted(set(bounds))).astype(np.float32)[None]
        known = sessions["dur2bd"].run(None, {"durations": durations, "maskT": mask})[0]
        previous = known
        for i in range(steps):
            previous = sessions["segmenter"].run(None, {
                "x_seg": xs, "language": np.array([config["languages"][region_language]], np.int64),
                "known_boundaries": known, "prev_boundaries": previous,
                "t": np.array([i / steps], np.float32), "maskT": mask,
                "threshold": np.array(.2, np.float32), "radius": np.array(2, np.int64)})[0]
        ds, mn = sessions["bd2dur"].run(None, {"boundaries": previous, "maskT": mask})
        presence, scores = sessions["estimator"].run(None, {
            "x_est": xe, "boundaries": previous, "maskT": mask, "maskN": mn,
            "threshold": np.array(.2, np.float32)})
        t = offset
        for dur, midi, voiced, valid in zip(ds[0], scores[0], presence[0], mn[0]):
            if not valid:
                continue
            if not np.isfinite(dur) or not np.isfinite(midi) or dur < 0:
                raise ValueError("GAME produced invalid note values")
            if dur > 0:
                notes.append({"start": t, "dur": float(dur), "midi": float(midi), "rest": not bool(voiced)})
            t += float(dur)
    return notes


def main(root: str, wav: str, regions_json: str, output: str, language: str, steps: str):
    config, sessions = load_models(root)
    waveform, _ = librosa.load(wav, sr=config["samplerate"], mono=True)
    regions = json.loads(Path(regions_json).read_text())
    notes = transcribe(waveform, config, sessions, regions, language, int(steps))
    if not notes:
        raise ValueError("GAME produced no note regions")
    Path(output).write_text(json.dumps({
        "backend": "game-medium", "version": "1.0.3", "steps": int(steps),
        "language": language, "word_boundaries": True,
        "languages": sorted({region.get("language", language) for region in regions}),
        "sample_rate": config["samplerate"], "timestep": config["timestep"],
        "providers": sessions["encoder"].get_providers(), "notes": notes}))
