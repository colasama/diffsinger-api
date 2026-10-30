"""Exercise GAME timeline conversion without downloading inference weights."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from dsapi import settings, tools
from dsapi.runners.game_runner import transcribe


def test_note_backend_rejects_typos_and_supports_legacy(monkeypatch):
    for name in ("game-medium", "some"):
        monkeypatch.setattr(settings, "NOTE_EXTRACTOR", name)
        assert tools.note_backend() == name
    monkeypatch.setattr(settings, "NOTE_EXTRACTOR", "gam")
    with pytest.raises(ValueError, match="DSAPI_NOTE_EXTRACTOR"):
        tools.note_backend()


def test_regions_preserve_vowel_boundaries_and_mimic_context(tmp_path, monkeypatch):
    phrase = SimpleNamespace(phones=[("SP", 1., 1.3), ("a", 1.3, 2.)],
                             syllables=[SimpleNamespace(start=1.4, end=1.9)])
    def run(vendor, script, *args, log):
        from pathlib import Path
        assert script == "game_runner.py"
        assert json.loads(Path(args[2]).read_text()) == [
            {"start": 1., "end": 2., "boundaries": [1.4, 1.9]}]
        assert args[-2:] == ("ja", "8")
        Path(args[3]).write_text(json.dumps({"notes": [{"midi": 60}]}))
    monkeypatch.setattr(tools, "_run", run)
    assert tools.extract_game_notes(tmp_path/"input.wav", tmp_path, [phrase], "ja") == [{"midi": 60}]


def test_absolute_offsets_rests_padding_and_sampling_schedule():
    calls = []
    mask = np.ones((1, 100), bool)
    known = np.zeros((1, 100), bool)
    known[0, 25] = True
    def encode(inputs):
        assert inputs["waveform"].shape == (1, 100)
        np.testing.assert_allclose(inputs["duration"], [1.])
        return [np.zeros((1, 100, 2)), np.zeros((1, 100, 2)), mask]
    def dur2bd(inputs):
        np.testing.assert_allclose(inputs["durations"], [[.25, .5, .25]])
        return [known]
    def segment(inputs):
        calls.append(float(inputs["t"][0]))
        assert inputs["language"].tolist() == [4]
        np.testing.assert_array_equal(inputs["known_boundaries"], known)
        return [known]
    def bd2dur(inputs):
        return [np.array([[.25,.75,0.]],np.float32), np.array([[True,True,False]])]
    def estimate(inputs):
        return [np.array([[False,True,True]]), np.array([[60.,72.,127.]])]
    funcs = dict(encoder=encode,dur2bd=dur2bd,segmenter=segment,bd2dur=bd2dur,estimator=estimate)
    sessions = {name: SimpleNamespace(run=lambda _, inputs, fn=fn: fn(inputs)) for name,fn in funcs.items()}
    notes = transcribe(np.zeros(400), {"samplerate":100,"languages":{"zh":4}}, sessions,
                       [{"start":2.,"end":3.,"boundaries":[1.,2.25,2.25,2.75,4.]}], "zh", 8)
    assert notes == [{"start":2.,"dur":.25,"midi":60.,"rest":True},
                     {"start":2.25,"dur":.75,"midi":72.,"rest":False}]
    assert calls == [i/8 for i in range(8)]
    with pytest.raises(ValueError, match="positive"):
        transcribe(np.zeros(1), {}, {}, [], "zh", 0)
