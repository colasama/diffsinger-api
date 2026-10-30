"""Official HubertFA v0.0.7 in a persistent worker, with the shared ONNX backend."""
import json
import pathlib
import sys
from functools import lru_cache

sys.path.insert(0, ".")

import torch  # worker host disables gradients for all runners
from onnx_infer import InferenceOnnx
from tools.g2p import DictionaryG2P

from dsapi.onnx_backend import backend_status, create_session


class Inference(InferenceOnnx):
    create_session = staticmethod(create_session)


@lru_cache(maxsize=None)
def load_models(ckpt: str):
    inference = Inference(pathlib.Path(ckpt))
    inference.load_config()
    inference.init_decoder()
    inference.load_model()
    return inference


def runtime_info():
    return backend_status()


def main(ckpt: str, folder: str, dictionary: str, out_json: str, language: str):
    inference = load_models(ckpt)
    g2p = DictionaryG2P(language if inference.vocab['language_prefix'] and language != 'mixed' else None,
                       dictionary)
    inference.dataset, inference.predictions = [], []
    try:
        for wav in sorted(pathlib.Path(folder).glob('*.wav')):
            units = wav.with_suffix('.lab').read_text(encoding='utf8').split()
            unknown = set(units) - g2p.dictionary.keys()
            if not units or unknown:
                raise ValueError(f"{wav.name}: empty lyrics or unknown units: {sorted(unknown)}")
            ph_seq, word_seq, mapping = g2p(' '.join(units))
            # This checkpoint merges Japanese closure with language-neutral silence.
            # Keep it inside its mora so its boundary still maps back to ja/cl.
            if language in ('ja', 'mixed'):
                ph_seq = ['cl' if p == 'ja/cl' else p for p in ph_seq]
            inference.dataset.append((wav, ph_seq, word_seq, mapping))
        if not inference.dataset:
            raise ValueError(f"No WAV/label pairs in {folder}")
        inference.infer(non_lexical_phonemes='AP', pad_times=1)
        out = {}
        for (wav, length, words), (_, _, expected, _) in zip(inference.predictions, inference.dataset):
            if [w.text for w in words if w.text not in ('SP', 'AP')] != expected:
                raise ValueError(f"{wav.name}: aligned word sequence mismatch")
            out[wav.stem] = dict(length=float(length), confidence=None, backend='hubertfa',
                providers=inference.model.get_providers(),
                words=[[w.text, float(w.start), float(w.end)] for w in words],
                phones=[[p.text, float(p.start), float(p.end)] for w in words for p in w.phonemes])
        pathlib.Path(out_json).write_text(json.dumps(out, ensure_ascii=False))
        print(json.dumps(runtime_info()), flush=True)
    finally:
        # Keep the network/decoders resident, but never reuse the preceding job's inputs.
        inference.dataset, inference.predictions = [], []
