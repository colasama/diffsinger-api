"""Keep the upgraded separator's precision, chunk schedule and cache stable."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
import soundfile as sf

from dsapi import audio


@pytest.mark.parametrize("compile_enabled", [True, False])
def test_separator_config_and_cached_output_directories(monkeypatch, tmp_path, compile_enabled):
    created = []

    class Separator:
        def __init__(self, **options):
            self.options = options
            self.model_instance = SimpleNamespace(output_dir=options["output_dir"])
            self.calls = []
            created.append(self)

        def load_model(self, model_filename):
            self.filename = model_filename

        def separate(self, source):
            self.calls.append((source, self.output_dir, self.model_instance.output_dir))
            return ["clip_(Vocals)_model.wav", "clip_(Instrumental)_model.wav"]

    monkeypatch.setitem(sys.modules, "audio_separator.separator", SimpleNamespace(Separator=Separator))
    monkeypatch.setattr(audio, "_separators", {})
    monkeypatch.setattr(audio.settings, "SEPARATOR_COMPILE", compile_enabled)
    first, second = tmp_path / "first", tmp_path / "second"
    audio._separate("model.ckpt", Path("first.wav"), first)
    stems = audio._separate("model.ckpt", Path("second.wav"), second)
    assert len(created) == 1
    sep = created[0]
    assert sep.options["use_autocast"] is False
    assert sep.options["use_native_fp16"] is False
    assert sep.options["use_torch_compile"] is compile_enabled
    assert sep.options["mdxc_params"]["overlap"] == 1
    assert sep.options["mdxc_params"]["override_model_segment_size"] is False
    assert sep.calls == [("first.wav", str(first), str(first)), ("second.wav", str(second), str(second))]
    assert stems == {"vocals": second / "clip_(Vocals)_model.wav",
                     "instrumental": second / "clip_(Instrumental)_model.wav"}


@pytest.mark.parametrize('karaoke,protection,dropout', [(True, True, True), (True, False, True),
                                                     (True, True, False), (False, True, True)])
def test_pipeline_uses_protected_lead_and_keeps_raw_stems(monkeypatch, tmp_path, karaoke, protection, dropout):
    rate = audio.settings.SAMPLE_RATE
    t = np.arange(rate * 2) / rate
    original = np.tile((0.2 * np.sin(2 * np.pi * 220 * t))[:, None], (1, 2)).astype(np.float32)
    raw = original * 0.5
    if dropout:
        raw[rate // 2:rate + rate // 2] = 0
    source, vocal, lead, inst = [tmp_path / n for n in ['clip.wav', 'vocals.wav', 'raw_lead.wav', 'inst.wav']]
    sf.write(vocal, original, rate, subtype='FLOAT')
    sf.write(lead, raw, rate, subtype='FLOAT')
    before = lead.read_bytes()
    monkeypatch.setattr(audio.settings, 'KARAOKE_MODEL', 'karaoke.ckpt' if karaoke else '')
    monkeypatch.setattr(audio.settings, 'KARAOKE_PROTECTION', protection)
    monkeypatch.setattr(audio, '_separate', lambda model, src, out: (
        {'vocals': lead, 'instrumental': inst} if model == 'karaoke.ckpt'
        else {'vocals': vocal, 'instrumental': inst}))
    out = tmp_path / 'sep'
    result, instrumental = audio.separate_vocals(source, out)
    mono, actual_rate = sf.read(result)
    assert actual_rate == rate and mono.shape == (len(original),)
    expected = original if not karaoke or (protection and dropout) else raw
    np.testing.assert_allclose(mono, expected.mean(axis=1), atol=4e-5)
    assert instrumental == inst
    assert lead.read_bytes() == before
    if karaoke:
        report = json.loads((out / 'lead_protection.json').read_text())
        assert report['enabled'] is protection
        if protection:
            assert report['fallback'] is dropout
            assert report['output'] == str(vocal if dropout else lead)
            assert not (out / 'lead_protected.wav').exists()
    else:
        assert not (out / 'lead_protection.json').exists()
