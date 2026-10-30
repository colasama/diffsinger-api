"""Compare CPU and native WebGPU on an existing job, without rerunning extraction.

Run each backend in its own process. Model loading is included in iteration 0;
subsequent iterations are warm. Outputs and timings go to a NEW output folder.
"""

import argparse
import json
import os
import time
from functools import partial
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path, help="completed job directory containing score/sofa/notes/f0")
    parser.add_argument("--backend", choices=("cpu", "webgpu"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phrase", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=2, help="warm repetitions after the first run")
    parser.add_argument("--acoustic-steps", type=int, default=20)
    parser.add_argument("--pitch-steps", type=int, default=10)
    parser.add_argument("--variance-steps", type=int, default=20)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if min(args.acoustic_steps, args.pitch_steps, args.variance_steps) < 1:
        parser.error("sampling steps must be positive")
    os.environ["DSAPI_ONNX_BACKEND"] = args.backend
    from dsapi import extract, voicebank
    from dsapi.score import Note, Phrase, Syllable, render_mimic, render_score

    args.output.mkdir(parents=True, exist_ok=False)
    vb = voicebank.Voicebank()
    # Pin these independently of changes to application defaults between runs.
    vb.render = partial(vb.render, steps=args.acoustic_steps)
    vb.predict_pitch = partial(vb.predict_pitch, steps=args.pitch_steps)
    vb.predict_variance = partial(vb.predict_variance, steps=args.variance_steps)
    raw = json.loads((args.job / "score.json").read_text())["phrases"][args.phrase]
    score = Phrase([Syllable(s["text"], s["pinyin"], [Note(**n) for n in s["notes"]], s.get("lang", "zh"))
                    for s in raw])
    # Reuse the saved alignment and original F0. Match the selected score
    # phrase's words; the source timing is deliberately not score-quantized.
    sofa = json.loads((args.job / "sofa.json").read_text())
    words = [w for w, _, _ in sofa["clip"]["words"] if w not in ("SP", "AP")]
    wanted = [s["pinyin"] for s in raw]
    matches = [i for i in range(len(words)) if words[i:i + len(wanted)] == wanted]
    if len(matches) != 1:
        raise ValueError("Phrase must match exactly one contiguous sequence of aligned words")
    first = matches[0]
    seg = {"name": "clip", "offset": 0.0, "lang": raw[0].get("lang", "zh"),
           "pinyin": words, "chars": words,
           "line_of": [0 if first <= i < first + len(wanted) else -1 for i in range(len(words))],
           "texts": ["benchmark source phrase"]}
    phrases, _ = extract.build_phrases(
        vb, [seg], sofa, json.loads((args.job / "notes.json").read_text())["notes"],
        np.load(args.job / "f0.npy")[0],
    )
    if len(phrases) != 1:
        raise ValueError("Select a score phrase corresponding to exactly one source phrase")

    records, sessions = [], []
    original_session = voicebank._session

    class MeasuredSession:
        def __init__(self, path):
            self.path = path
            t = time.perf_counter()
            self.session = original_session(path)
            sessions.append({"model": path.name, "load_seconds": time.perf_counter() - t,
                             "providers": self.session.get_providers()})

        def run(self, names, feed):
            t = time.perf_counter()
            result = self.session.run(names, feed)
            records.append({"model": self.path.name, "seconds": time.perf_counter() - t})
            if any(np.issubdtype(x.dtype, np.floating) and not np.isfinite(x).all() for x in result):
                raise RuntimeError(f"Nonfinite output from {self.path.name}")
            return result

    voicebank._session = MeasuredSession
    summary = {"backend": args.backend, "ort": ort.__version__, "job": str(args.job.resolve()),
               "phrase": args.phrase,
               "steps": {"acoustic": args.acoustic_steps, "pitch": args.pitch_steps,
                         "variance": args.variance_steps},
               "sessions": sessions, "runs": []}
    try:
        for iteration in range(args.repeats + 1):
            for mode, phrase, render in (("mimic", phrases[0], render_mimic), ("score", score, render_score)):
                ort.set_seed(123)
                records.clear()
                t = time.perf_counter()
                wav = render(vb, phrase)
                elapsed = time.perf_counter() - t
                sf.write(args.output / f"{mode}-{iteration}.wav", wav, 44100, subtype="FLOAT")
                item = {"mode": mode, "iteration": iteration, "seconds": elapsed,
                        "duration": len(wav) / 44100, "rms": float(np.sqrt(np.mean(wav ** 2))),
                        "peak": float(np.max(abs(wav))), "stages": records.copy()}
                summary["runs"].append(item)
                (args.output / "results.json").write_text(json.dumps(summary, indent=2))
                print(json.dumps(item), flush=True)
    finally:
        voicebank._session = original_session


if __name__ == "__main__":
    main()
