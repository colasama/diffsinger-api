"""Local model smoke test; no song downloads or external messages.

DSAPI_ONNX_BACKEND=cpu uv run python scripts/smoke_voices.py
"""
import json
from pathlib import Path

import numpy as np

from dsapi import audio, voices
from dsapi.pipeline import voicebank
from dsapi.score import Note, Phrase, Syllable, render_mimic, render_score


def main():
    output = Path('work/voice-smoke')
    output.mkdir(parents=True, exist_ok=True)
    report = []
    for voice in voices.VOICES:
        voice.validate()
        bank = voicebank(voice.id, 42)
        for mode in ('score', 'mimic'):
            phrase = Phrase([Syllable('啦', 'la', [Note(.5, .6, 60), Note(1.1, .6, 64)])])
            if mode == 'mimic':
                phones = bank.g2p('la')
                edges = np.linspace(.4, 1.7, len(phones) + 1)
                phrase.phones = [(p, float(a), float(b)) for p, a, b in zip(phones, edges[:-1], edges[1:])]
                phrase.f0_midi = np.full(200, 60, np.float32)
            wav = (render_score if mode == 'score' else render_mimic)(bank, phrase)
            peak = float(np.max(np.abs(wav)))
            if not np.isfinite(wav).all() or not 1e-5 < peak < 10:
                raise RuntimeError(f'{voice.id}/{mode}: invalid waveform')
            path = output / f'{voice.id}-{mode}.wav'
            audio.save(path, wav)
            item = {'voice':voice.id, 'mode':mode, 'seconds':len(wav)/44100, 'peak':peak}
            report.append(item)
            print(json.dumps(item), flush=True)
    (output/'report.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
