"""API tests validate a tiny on-disk catalog without downloading model weights."""
import json

import numpy as np
import pytest
import yaml


@pytest.fixture
def voice_files(tmp_path, monkeypatch):
    from dsapi import settings, voices

    for setting in ("VOICEBANK", "MULTI_VOICE"):
        root = tmp_path / setting.lower()
        monkeypatch.setattr(settings, setting, root)
        for stage in ("", "dsdur", "dspitch", "dsvariance"):
            folder = root / stage
            folder.mkdir(parents=True, exist_ok=True)
            cfg = {"phonemes": "phones.json", "languages": "languages.json", "hidden_size": 384,
                   "vocoder": "test", **{key:"model.onnx" for key in ("acoustic","linguistic","dur","pitch","variance")}}
            (folder / "phones.json").write_text(json.dumps({"SP": 0}))
            (folder / "languages.json").write_text('{}')
            (folder / "model.onnx").touch()
            if setting == "MULTI_VOICE":
                cfg["speakers"] = ["stage." + v.speaker for v in voices.VOICES if v.speaker]
                for speaker in cfg["speakers"]:
                    np.ones(384, dtype='<f4').tofile(folder / (speaker + '.emb'))
            (folder / "dsconfig.yaml").write_text(yaml.safe_dump(cfg))
        for lang in ("zh", "ja", "en"):
            (root / "dsdur" / f"dsdict-{lang}.yaml").touch()
    vocoder = tmp_path / 'vocoder'
    vocoder.mkdir()
    (vocoder / 'vocoder.yaml').write_text('name: test\nmodel: vocoder.onnx\n')
    (vocoder / 'vocoder.onnx').touch()
    monkeypatch.setattr(settings, 'VOCODER', vocoder)
