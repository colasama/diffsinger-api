"""Extract F0 without loading SOME's note network (cwd=vendor/SOME)."""
import sys
from functools import lru_cache

sys.path.insert(0, ".")

import librosa
import numpy as np
import torch
from modules.rmvpe import RMVPE
from utils.binarizer_utils import get_pitch_parselmouth


@lru_cache(maxsize=None)
def load_models(checkpoint: str, device: str):
    return RMVPE(checkpoint, device=device) if checkpoint != "-" else None


def main(checkpoint: str, wav: str, output: str, device: str):
    model = load_models(checkpoint, device)
    waveform, _ = librosa.load(wav, sr=44100, mono=True)
    length = (len(waveform) + 511) // 512
    f0, uv = np.zeros(length, np.float32), np.ones(length, bool)
    step, context = 3445, 173  # about 40 seconds plus 2 seconds context
    for start in range(0, length, step):
        stop = min(length, start + step)
        lo, hi = max(0, start - context), min(length, stop + context)
        chunk = waveform[lo * 512:min(len(waveform), hi * 512)]
        kwargs = dict(sample_rate=44100, hop_size=512, length=hi-lo, interp_uv=True)
        cf, cu = (model.get_pitch(chunk, **kwargs) if model is not None
                  else get_pitch_parselmouth(chunk, **kwargs))
        f0[start:stop], uv[start:stop] = cf[start-lo:stop-lo], cu[start-lo:stop-lo]
    np.save(output, np.stack([f0, uv]).astype(np.float32))
