"""Preserve performance timing through model inputs and score re-rendering."""
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from dsapi import api, pipeline, settings
from dsapi.jianpu import Grid
from dsapi.score import Note, Phrase, Syllable, score_inputs


def test_off_grid_short_notes_and_rests_reach_pitch_model():
    bank = SimpleNamespace(g2p=lambda *a: ['a'], is_vowel=lambda p: True)
    phrase = Phrase([
        Syllable('啊', 'a', [Note(.61, .06, 60), Note(.72, .09, 62)]),
        Syllable('啊', 'a', [Note(.88, .21, 64)]),
    ])
    _, _, word_dur, _, notes = score_inputs(bank, phrase)
    assert notes.rest.tolist() == [True, False, True, False, True, False, True]
    assert notes.midi[~notes.rest].tolist() == [60, 62, 64]
    fps = settings.SAMPLE_RATE / settings.HOP
    # Absolute boundaries are rounded to synthesis frames, never musical beats.
    bounds = np.array([phrase.start, .61, .67, .72, .81, .88, 1.09, phrase.end])
    np.testing.assert_array_equal(notes.dur, np.diff(np.round(bounds * fps)).astype(int))
    assert notes.dur.sum() == word_dur.sum()


@pytest.mark.parametrize('start', [.11, .61])
def test_exported_score_rerenders_at_same_clip_time(voice_files, tmp_path, monkeypatch, start):
    monkeypatch.setattr(settings, 'WORK', tmp_path / 'work')
    monkeypatch.setattr(api, 'API_KEY', '')
    monkeypatch.setattr(api, 'API_KEYS', {})
    monkeypatch.setattr(api, 'voicebank', lambda *a: object())
    phrase = Phrase([Syllable('啊', 'a', [Note(start, .08, 60), Note(start + .08, .12, 62)])])
    body = pipeline._score_json([phrase], Grid(120, 0))
    seen = []
    sr = settings.SAMPLE_RATE

    def render(bank, p):
        seen.append(pipeline._score_json([p], Grid(120, 0)))
        wav = np.zeros(round((p.end - p.start) * sr), np.float32)
        offset = round(p.start * sr)
        wav[round(p.syllables[0].start * sr) - offset:
            round(p.syllables[-1].end * sr) - offset] = .25
        return wav

    def encode(src, dst, **kwargs):
        dst.write_bytes(b'ID3')
        return dst

    monkeypatch.setattr(api, 'render_score', render)
    monkeypatch.setattr(api.audio, 'to_mp3', encode)
    monkeypatch.setattr(pipeline.Job, '_export_audio', lambda *a: None)
    job = pipeline.Job('timing', pipeline.JobSpec(modes=['score']))
    job._mix('score', [(phrase, render(None, phrase))], 2 * sr)
    # The exported payload, including its descriptive BPM, is accepted unchanged.
    response = TestClient(api.app).post('/render_score', json=body)
    assert response.status_code == 200, response.text
    assert seen == [body, body]
    original, _ = sf.read(job.dir / 'score.wav')
    rerendered, _ = sf.read(next((settings.WORK / 'render').glob('*/out.wav')))
    for wav in (original, rerendered):
        voiced = np.flatnonzero(wav)
        assert voiced[0] == round(start * sr)
        assert voiced[-1] + 1 == round((start + .2) * sr)
