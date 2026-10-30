import os
from pathlib import Path

import torch

ROOT = Path(os.environ.get("DSAPI_ROOT", Path(__file__).resolve().parents[2]))

# Local, uncommitted overrides (KEY=VALUE per line); real env vars win.
_env = ROOT / ".env"
if _env.is_file():
    for _line in _env.read_text().splitlines():
        _k, _sep, _v = _line.strip().partition("=")
        if _sep and not _k.startswith("#"):
            os.environ.setdefault(_k.strip(), _v.strip())
MODELS = ROOT / "models"
VENDOR = ROOT / "vendor"
WORK = Path(os.environ.get("DSAPI_WORK", ROOT / "work"))

VOICEBANK = MODELS / "imported-fixed" / "voicebank"
VOCODER = MODELS / "vocoder"
SOME_CKPT = MODELS / "some" / "0119_continuous256_5spk" / "model_ckpt_steps_100000_simplified.ckpt"
NOTE_EXTRACTOR = os.environ.get("DSAPI_NOTE_EXTRACTOR", "game-medium").strip().lower()
GAME_MODEL = MODELS / "game" / "GAME-1.0.3-medium-onnx"
GAME_STEPS = 8
# Must be the yxlllc/RMVPE checkpoint SOME's model expects (the RVC rmvpe.pt has a
# different architecture). Absent -> Praat/parselmouth f0.
RMVPE_CKPT = Path(os.environ.get("DSAPI_RMVPE", MODELS / "rmvpe" / "model.pt"))
SOFA_CKPT = MODELS / "sofa_mandarin" / "pretrained_mandarin_singing" / "v1.0.0_mandarin_singing.ckpt"
SOFA_DICT = VENDOR / "SOFA" / "dictionary" / "opencpop-extension.txt"
ALIGNER = os.environ.get("DSAPI_ALIGNER", "hubertfa").strip().lower()
JA_ALIGNER = os.environ.get("DSAPI_JA_ALIGNER", "sofa").strip().lower()
HUBERTFA_MODEL = MODELS / "hubertfa" / "1218_hfa_model_new_dict" / "model.onnx"
HUBERTFA_DICT = HUBERTFA_MODEL.parent / "ds-zh-pinyin-lite.txt"
# Japanese aligns with SOFA's multilingual model (opencpop-style phoneme set incl. "cl").
SOFA_CKPT_MULTI = MODELS / "sofa_multilingual" / "pretrained_multilingual_singing" / "v1.0.0_multilingual_singing.ckpt"

SEPARATOR_MODELS = MODELS / "separator"
VOCALS_MODEL = os.environ.get("DSAPI_VOCALS_MODEL", "model_bs_roformer_ep_317_sdr_12.9755.ckpt")
# Strips backing vocals from the separated vocal stem; empty disables it.
KARAOKE_MODEL = os.environ.get("DSAPI_KARAOKE_MODEL", "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt")
# Keep FP32; compile repeated transformer blocks once per cached model.
SEPARATOR_COMPILE = os.environ.get("DSAPI_SEPARATOR_COMPILE", "1").lower() in ("1", "true", "yes")

WEBUI = os.environ.get("DSAPI_WEBUI", "").lower() in ("1", "true", "yes")

SAMPLE_RATE = 44100
HOP = 512


def torch_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
