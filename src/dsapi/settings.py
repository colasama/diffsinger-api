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
CACHE_ENABLED = os.environ.get("DSAPI_CACHE", "1").lower() in ("1", "true", "yes")
CACHE_TTL = max(0, float(os.environ.get("DSAPI_CACHE_TTL", "86400")))
CACHE_MAX_BYTES = max(0, int(os.environ.get("DSAPI_CACHE_MAX_BYTES", str(5 * 1024**3))))
PREFETCH_DOWNLOADS = os.environ.get("DSAPI_PREFETCH", "1").lower() in ("1", "true", "yes")


def _positive_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


ACTIVE_JOBS = _positive_int("DSAPI_ACTIVE_JOBS", 3)
# Share a GPU budget across MPS separation, child-process alignment and synthesis.
# Raising this is experimental; the native WebGPU Run lock remains in force.
GPU_CONCURRENCY = _positive_int("DSAPI_GPU_CONCURRENCY", 1)

VOICEBANK = Path(os.environ.get("DSAPI_VOICEBANK", MODELS / "voicebank")).expanduser()
VOCODER = Path(os.environ.get("DSAPI_VOCODER", MODELS / "vocoder")).expanduser()
VOICES_FILE = Path(os.environ["DSAPI_VOICES_FILE"]).expanduser().resolve() if os.environ.get("DSAPI_VOICES_FILE") else None
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
JA_ALIGNER = os.environ.get("DSAPI_JA_ALIGNER", "hubertfa").strip().lower()
HUBERTFA_MODEL = MODELS / "hubertfa" / "1218_hfa_model_new_dict" / "model.onnx"
HUBERTFA_DICT = HUBERTFA_MODEL.parent / "ds-zh-pinyin-lite.txt"
# Optional legacy Japanese alignment model (opencpop-style phoneme set incl. "cl").
SOFA_CKPT_MULTI = MODELS / "sofa_multilingual" / "pretrained_multilingual_singing" / "v1.0.0_multilingual_singing.ckpt"

SEPARATOR_MODELS = MODELS / "separator"
VOCALS_MODEL = os.environ.get("DSAPI_VOCALS_MODEL", "model_bs_roformer_ep_368_sdr_12.9628.ckpt")
# Strips backing vocals from the separated vocal stem; empty disables it.
KARAOKE_MODEL = os.environ.get("DSAPI_KARAOKE_MODEL", "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt")
# Fall back to first-stage vocals for the whole clip on sustained second-stage dropouts.
KARAOKE_PROTECTION = os.environ.get("DSAPI_KARAOKE_PROTECTION", "1").lower() in ("1", "true", "yes")
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
