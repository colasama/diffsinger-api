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
    kwargs = dict(sample_rate=44100, hop_size=512,
                  length=(len(waveform) + 511) // 512, interp_uv=True)
    f0, uv = (model.get_pitch(waveform, **kwargs) if model is not None
              else get_pitch_parselmouth(waveform, **kwargs))
    np.save(output, np.stack([f0, uv]).astype(np.float32))
