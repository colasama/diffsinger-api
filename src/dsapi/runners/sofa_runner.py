"""Run SOFA forced alignment over a folder of <name>.wav + <name>.lab (space
separated pinyin). Executed as a subprocess with cwd=vendor/SOFA.

Usage: python sofa_runner.py <ckpt> <folder> <dictionary> <out.json> <accelerator>
"""

import json
import pathlib
import sys
from functools import lru_cache

sys.path.insert(0, ".")

import lightning as pl  # noqa: E402
import torch  # noqa: E402

import modules.AP_detector  # noqa: E402
import modules.g2p  # noqa: E402
import modules.utils.load_wav  # noqa: E402
from modules.utils.post_processing import post_processing  # noqa: E402
from train import LitForcedAlignmentTask  # noqa: E402

# torchaudio.load now needs torchcodec; SOFA's own librosa branch reads wavs fine.
modules.utils.load_wav.installed_torchaudio = False


@lru_cache(maxsize=None)
def load_models(ckpt: str, accelerator: str):
    model = LitForcedAlignmentTask.load_from_checkpoint(ckpt, map_location="cpu")
    model.set_inference_mode("force")
    return model


def main(ckpt: str, folder: str, dictionary: str, out_json: str, accelerator: str):
    g2p = modules.g2p.DictionaryG2P(dictionary=dictionary)
    g2p.set_in_format("lab")
    dataset = g2p.get_dataset(sorted(pathlib.Path(folder).glob("*.wav")))

    model = load_models(ckpt, accelerator)
    trainer = pl.Trainer(logger=False, accelerator=accelerator, devices=1, enable_progress_bar=False)
    try:
        predictions = trainer.predict(model, dataloaders=dataset, return_predictions=True)
    finally:
        # Cache the network, not the previous request's trainer/dataset/results.
        model.trainer = None
    predictions = modules.AP_detector.LoudnessSpectralcentroidAPDetector().process(predictions)
    predictions, errors = post_processing(predictions)

    out = {}
    for wav_path, wav_length, confidence, ph_seq, ph_iv, word_seq, word_iv in predictions:
        out[pathlib.Path(wav_path).stem] = {
            "length": float(wav_length),
            "confidence": float(confidence),
            "phones": [[p, float(s), float(e)] for p, (s, e) in zip(ph_seq, ph_iv)],
            "words": [[w, float(s), float(e)] for w, (s, e) in zip(word_seq, word_iv)],
        }
    for wav_path, err in errors:
        out[pathlib.Path(wav_path).stem] = {"error": str(err)}
    pathlib.Path(out_json).write_text(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main(*sys.argv[1:6])
