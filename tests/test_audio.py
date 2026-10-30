"""Keep the upgraded separator's precision, chunk schedule and cache stable."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

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
