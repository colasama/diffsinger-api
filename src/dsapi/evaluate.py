"""Objective checks for a finished task, since nobody listens in CI.

- mimic pitch: rendered f0 vs the source f0 it was given (cents)
- mimic intelligibility: re-align the render with the task's original aligner
  and compare word onsets with the source alignment
- score pitch: rendered f0 per note vs the saved synthesis score's midi
- alignment: share of lyric units aligned shorter than 50 ms
"""

import itertools
import json
import shutil
from pathlib import Path

import librosa
import numpy as np

from . import audio, settings, tools


def _f0(path: Path) -> tuple[np.ndarray, np.ndarray]:
    y = audio.load(path)
    f0, voiced, _ = librosa.pyin(y, fmin=80, fmax=1000, sr=settings.SAMPLE_RATE,
                                 frame_length=2048, hop_length=settings.HOP)
    return f0, voiced


def _cents(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 1200 * np.abs(np.log2(a / b))


def evaluate(task_dir: Path) -> dict:
    out: dict = {}
    aligned = json.loads((task_dir / "sofa.json").read_text())
    segments = (json.loads((task_dir / "segments.json").read_text())
                if (task_dir / "segments.json").is_file() else [{"name": "clip", "offset": 0}])
    # Lyric units the aligner squeezed to slivers; a regression signal without labels.
    durations = [w[2] - w[1] for segment in segments if "words" in aligned.get(segment["name"], {})
                 for w, line in zip([w for w in aligned[segment["name"]]["words"] if w[0] not in ("SP", "AP")],
                                    segment.get("line_of") or itertools.repeat(0)) if line >= 0]
    out["alignment"] = {"units": len(durations),
                        "below_50ms": round(float(np.mean(np.array(durations) < 0.05)), 3) if durations else None}

    if (task_dir / "mimic.wav").is_file():
        src_f0, src_uv = np.load(task_dir / "f0.npy")
        f0, voiced = _f0(task_dir / "mimic.wav")
        n = min(len(f0), len(src_f0))
        m = voiced[:n] & (src_uv[:n] < 0.5) & np.isfinite(f0[:n])
        c = _cents(f0[:n][m], src_f0[:n][m])
        out["mimic_pitch"] = {"median_cents": round(float(np.median(c)), 1),
                              "within_50c": round(float(np.mean(c < 50)), 3), "frames": int(m.sum())}

        realign = task_dir / "eval_align"
        shutil.rmtree(realign, ignore_errors=True)
        realign.mkdir()
        waveform = audio.load(task_dir / "mimic.wav")
        for segment in segments:
            name, offset = segment["name"], segment["offset"]
            if len(segments) == 1 and offset == 0:
                shutil.copy(task_dir / "mimic.wav", realign / f"{name}.wav")
            else:
                start = round(offset * settings.SAMPLE_RATE)
                stop = start + round(aligned[name]["length"] * settings.SAMPLE_RATE)
                audio.save(realign / f"{name}.wav", waveform[start:stop])
            shutil.copy(task_dir / "segments" / f"{name}.lab", realign / f"{name}.lab")
        meta = json.loads((task_dir / "align_meta.json").read_text()) if (task_dir / "align_meta.json").is_file() else {}
        results = tools.align(realign, realign, Path(meta.get("ckpt", settings.SOFA_CKPT)),
                              Path(meta.get("dictionary", settings.SOFA_DICT)),
                              backend=meta.get("backend", "sofa"), lang=meta.get("lang", "zh"))
        differences, source_confidence, render_confidence = [], [], []
        for segment in segments:
            name, offset = segment["name"], segment["offset"]
            sofa, res = aligned[name], results[name]
            src_words = [w for w in sofa["words"] if w[0] not in ("SP", "AP")]
            out_words = [w for w in res["words"] if w[0] not in ("SP", "AP")]
            if [w[0] for w in src_words] != [w[0] for w in out_words]:
                raise ValueError("Evaluation word sequence mismatch")
            for i, word in enumerate(src_words):
                if segment.get("line_of", [0]*len(src_words))[i] < 0:
                    continue
                start, stop = (round((t+offset)*settings.SAMPLE_RATE) for t in word[1:])
                if np.any(np.abs(waveform[start:stop]) > 1e-3):
                    differences.append(abs(out_words[i][1]-word[1])*1000)
            if sofa.get("confidence") is not None:
                source_confidence.append(sofa["confidence"])
            if res.get("confidence") is not None:
                render_confidence.append(res["confidence"])
        diffs = np.array(differences)
        out["mimic_intelligibility"] = {
            "realign_confidence": round(float(np.mean(render_confidence)), 3) if render_confidence else None,
            "source_confidence": round(float(np.mean(source_confidence)), 3) if source_confidence else None,
            "onset_median_ms": round(float(np.median(diffs)), 1) if len(diffs) else None,
            "onset_within_50ms": round(float(np.mean(diffs < 50)), 3) if len(diffs) else None,
            "words": len(diffs)}

    if (task_dir / "score.wav").is_file():
        score = json.loads((task_dir / "score.json").read_text())
        f0, voiced = _f0(task_dir / "score.wav")
        fps = settings.SAMPLE_RATE / settings.HOP
        hits, total = 0, 0
        for phrase in score["phrases"]:
            for syl in phrase:
                for note in syl["notes"]:
                    # Skip the attack and release; judge the sustained middle.
                    a = int((note["start"] + 0.25 * note["dur"]) * fps)
                    b = int((note["start"] + 0.75 * note["dur"]) * fps)
                    seg_f0 = f0[a:b][voiced[a:b]]
                    if len(seg_f0) < 2:
                        continue
                    total += 1
                    hits += int(abs(np.median(librosa.hz_to_midi(seg_f0)) - note["midi"]) < 0.5)
        out["score_pitch"] = {"notes_on_pitch": round(hits / total, 3) if total else None, "notes": total}

    (task_dir / "eval.json").write_text(json.dumps(out, indent=2))
    return out
