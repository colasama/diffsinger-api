"""API tests validate a tiny on-disk catalog without downloading model weights."""
import json

import numpy as np
import pytest
import yaml


@pytest.fixture(autouse=True)
def no_background_downloads(monkeypatch):
    # Route tests stub inference; they must not launch real network prefetches.
    from dsapi import settings
    monkeypatch.setattr(settings, "PREFETCH_DOWNLOADS", False)


@pytest.fixture
def voice_files(tmp_path, monkeypatch):
    from dsapi import settings, voices

    imported = (
        voices.Voice("default", "测试音源", "A", ("测试",), tmp_path / "bank"),
        voices.Voice("imported-b", "测试音源 B", "B", ("乙",), tmp_path / "bank"),
        voices.Voice("imported-fixed", "固定音源", None, (), tmp_path / "fixed"),
    )
    monkeypatch.setattr(voices, "VOICES", imported)
    for root in {v.root for v in imported}:
        for stage in ("", "dsdur", "dspitch", "dsvariance"):
            folder = root / stage
            folder.mkdir(parents=True, exist_ok=True)
            cfg = {"phonemes": "phones.json", "languages": "languages.json", "hidden_size": 384,
                   "vocoder": "test", **{key:"model.onnx" for key in ("acoustic","linguistic","dur","pitch","variance")}}
            (folder / "phones.json").write_text(json.dumps({"SP": 0}))
            (folder / "languages.json").write_text('{}')
            (folder / "model.onnx").touch()
            cfg["speakers"] = ["stage." + v.speaker for v in imported if v.root == root and v.speaker]
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
