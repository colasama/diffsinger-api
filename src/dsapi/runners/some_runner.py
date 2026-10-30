"""Run SOME note extraction and f0 (RMVPE if a compatible checkpoint is given,
else Praat via parselmouth) on one wav. Executed as a subprocess
with cwd=vendor/SOME, because SOME's top-level packages clash with SOFA's.

Usage: python some_runner.py <ckpt> <rmvpe.pt|-> <wav> <out.json> <out_f0.npy> <device>
"""

import json
import pathlib
import sys

sys.path.insert(0, ".")

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

import inference  # noqa: E402
import modules.rmvpe  # noqa: E402
from utils.binarizer_utils import get_pitch_parselmouth  # noqa: E402
from utils.slicer2 import Slicer  # noqa: E402


def main(ckpt: str, rmvpe_path: str, wav: str, out_json: str, out_f0: str, device: str):
    ckpt_path = pathlib.Path(ckpt)
    config = yaml.safe_load(ckpt_path.with_name("config.yaml").read_text(encoding="utf8"))
    sr, hop = config["audio_sample_rate"], config["hop_size"]
    waveform, _ = librosa.load(wav, sr=sr, mono=True)

    infer_ins = inference.MIDIExtractionInference(config=config, model_path=ckpt_path, device=device)
    slicer = Slicer(sr=sr, max_sil_kept=1000)
    chunks = slicer.slice(waveform)
    results = infer_ins.infer([c["waveform"] for c in chunks])

    notes = []
    for chunk, res in zip(chunks, results):
        t = float(chunk["offset"])
        for midi, dur, rest in zip(res["note_midi"], res["note_dur"], res["note_rest"]):
            notes.append({"start": t, "dur": float(dur), "midi": float(midi), "rest": bool(rest)})
            t += float(dur)

    length = (len(waveform) + hop - 1) // hop
    if rmvpe_path != "-":
        rmvpe = modules.rmvpe.RMVPE(rmvpe_path, device=device)
        f0, uv = rmvpe.get_pitch(waveform, sample_rate=sr, hop_size=hop, length=length, interp_uv=True)
    else:
        f0, uv = get_pitch_parselmouth(waveform, sample_rate=sr, hop_size=hop, length=length, interp_uv=True)
    np.save(out_f0, np.stack([f0.astype(np.float32), uv.astype(np.float32)]))

    pathlib.Path(out_json).write_text(json.dumps({"sample_rate": sr, "hop_size": hop, "notes": notes}))


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main(*sys.argv[1:7])
