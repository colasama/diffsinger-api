"""Run SOME note extraction and f0 (RMVPE if a compatible checkpoint is given,
else Praat via parselmouth) on one wav. Executed as a subprocess
with cwd=vendor/SOME, because SOME's top-level packages clash with SOFA's.

Usage: python some_runner.py <ckpt> <rmvpe.pt|-> <wav> <out.json> <out_f0.npy> <device>
"""

import json
import pathlib
import sys
from functools import lru_cache

sys.path.insert(0, ".")

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

import inference  # noqa: E402
import modules.rmvpe  # noqa: E402
from utils.binarizer_utils import get_pitch_parselmouth  # noqa: E402
from utils.slicer2 import Slicer  # noqa: E402


@lru_cache(maxsize=None)
def load_models(ckpt: str, rmvpe_path: str, device: str):
    ckpt_path = pathlib.Path(ckpt)
    config = yaml.safe_load(ckpt_path.with_name("config.yaml").read_text(encoding="utf8"))
    infer_ins = inference.MIDIExtractionInference(config=config, model_path=ckpt_path, device=device)
    rmvpe = modules.rmvpe.RMVPE(rmvpe_path, device=device) if rmvpe_path != "-" else None
    return config, infer_ins, rmvpe


def main(ckpt: str, rmvpe_path: str, wav: str, out_json: str, out_f0: str, device: str):
    config, infer_ins, rmvpe = load_models(ckpt, rmvpe_path, device)
    sr, hop = config["audio_sample_rate"], config["hop_size"]
    waveform, _ = librosa.load(wav, sr=sr, mono=True)

    slicer = Slicer(sr=sr, max_sil_kept=1000)
    chunks = []
    for original in slicer.slice(waveform):
        samples = original["waveform"]
        for start in range(0, len(samples), sr * 40):
            chunks.append({"waveform": samples[start:start + sr * 40],
                           "offset": float(original["offset"]) + start / sr})
    # Avoid retaining activations for an entire long excerpt in one batch.
    results = (infer_ins.infer([c["waveform"]])[0] for c in chunks)

    notes = []
    for chunk, res in zip(chunks, results):
        t = float(chunk["offset"])
        for midi, dur, rest in zip(res["note_midi"], res["note_dur"], res["note_rest"]):
            notes.append({"start": t, "dur": float(dur), "midi": float(midi), "rest": bool(rest)})
            t += float(dur)

    length = (len(waveform) + hop - 1) // hop
    f0, uv = np.zeros(length, np.float32), np.ones(length, bool)
    step, context = max(1, round(sr * 40 / hop)), max(1, round(sr * 2 / hop))
    for start in range(0, length, step):
        stop = min(length, start + step)
        lo, hi = max(0, start - context), min(length, stop + context)
        samples = waveform[lo * hop:min(len(waveform), hi * hop)]
        kwargs = dict(sample_rate=sr, hop_size=hop, length=hi-lo, interp_uv=True)
        cf, cu = (rmvpe.get_pitch(samples, **kwargs) if rmvpe is not None
                  else get_pitch_parselmouth(samples, **kwargs))
        f0[start:stop], uv[start:stop] = cf[start-lo:stop-lo], cu[start-lo:stop-lo]
    np.save(out_f0, np.stack([f0.astype(np.float32), uv.astype(np.float32)]))

    pathlib.Path(out_json).write_text(json.dumps({"sample_rate": sr, "hop_size": hop, "notes": notes}))


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main(*sys.argv[1:7])
