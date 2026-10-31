from types import SimpleNamespace

from dsapi import settings
from dsapi.voicebank import Voicebank, _read_dictionary


def bank(root):
    b = Voicebank.__new__(Voicebank)
    b.acoustic = b.dur = b.pitch = b.variance = SimpleNamespace(root=root, phonemes={'en/ax': 1})
    return b


def test_shared_lazy_dictionary_preserves_both_yaml_section_orders(tmp_path, monkeypatch):
    _read_dictionary.cache_clear()
    monkeypatch.setattr(settings, 'HUBERTFA_MODEL', tmp_path / 'model.onnx')
    (tmp_path / 'dsdict-zh.yaml').write_text('entries:\n- grapheme: ni\n  phonemes:\n  - i\nsymbols:\n- symbol: i\n  type: vowel\n')
    (tmp_path / 'dsdict-ja.yaml').write_text('symbols:\n- symbol: ja/a\n  type: vowel\nentries:\n- grapheme: a\n  phonemes:\n  - ja/a\n')
    # Missing CMU lexicon must not be read by a Chinese-only request.
    (tmp_path / 'dsdict-en.yaml').write_text('symbols:\n- symbol: en/ax\n  type: vowel\n')
    a, b = bank(tmp_path), bank(tmp_path)
    assert a.g2p('ni') == ['i']
    assert a.is_vowel('i') and a.is_vowel('ja/a') and a.is_vowel('en/ax')
    assert a.dictionaries['zh'] is b.dictionaries['zh']
    assert _read_dictionary.cache_info().hits > 0
    phones = a.g2p('ni'); phones.append('corruption')
    assert b.g2p('ni') == ['i']
    (tmp_path / 'dsdict-zh.yaml').write_text('symbols:\n- symbol: a\n  type: vowel\nentries:\n- grapheme: ni\n  phonemes:\n  - a\n')
    assert bank(tmp_path).g2p('ni') == ['a']
