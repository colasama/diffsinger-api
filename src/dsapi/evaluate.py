"""Objective checks for a finished task, since nobody listens in CI.

- mimic pitch: rendered f0 vs the source f0 it was given (cents)
- mimic intelligibility: re-align the render against the same lyrics with SOFA
  and compare word onsets with the source alignment
- score pitch: rendered f0 per note vs the quantized score's midi
"""

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
    sofa = json.loads((task_dir / "sofa.json").read_text())["clip"]
    seg = (task_dir / "segments" / "clip.lab").read_text().split()

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
        shutil.copy(task_dir / "mimic.wav", realign / "clip.wav")
        (realign / "clip.lab").write_text(" ".join(seg), encoding="utf8")
        meta = json.loads((task_dir / "align_meta.json").read_text()) if (task_dir / "align_meta.json").is_file() else {}
        res = tools.align(realign, realign, Path(meta.get("ckpt", settings.SOFA_CKPT)),
                          Path(meta.get("dictionary", settings.SOFA_DICT)))["clip"]
        src_words = [w for w in sofa["words"] if w[0] not in ("SP", "AP")]
        out_words = [w for w in res["words"] if w[0] not in ("SP", "AP")]
        # Only words that were actually rendered (context lines are silent in the render).
        rendered = [i for i, w in enumerate(src_words)
                    if np.any(np.abs(audio.load(task_dir / "mimic.wav")[int(w[1] * settings.SAMPLE_RATE):
                                                                       int(w[2] * settings.SAMPLE_RATE)]) > 1e-3)]
        diffs = np.array([abs(out_words[i][1] - src_words[i][1]) for i in rendered]) * 1000
        out["mimic_intelligibility"] = {"realign_confidence": round(res["confidence"], 3),
                                        "source_confidence": round(sofa["confidence"], 3),
                                        "onset_median_ms": round(float(np.median(diffs)), 1),
                                        "onset_within_50ms": round(float(np.mean(diffs < 50)), 3),
                                        "words": len(rendered)}

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
