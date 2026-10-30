"""English lyrics must survive parsing and preserve each sung vowel."""
import json

import numpy as np
import pytest

from dsapi import english, extract, lrc, settings, tools
from dsapi.score import score_inputs
from dsapi.voicebank import Voicebank


def test_english_detection_contractions_and_distinct_chorus_lines(monkeypatch):
    monkeypatch.delenv('DSAPI_EXPERIMENTAL_JA', raising=False)
    lines = lrc.parse("[00:00]作词: Someone\n[00:01]Hello, don't go\n"
                      "[00:05]You’re still here\n[00:09]空へ\n[00:12]Hello, don't go", 18)
    assert len(lines) == 3
    assert all(ln.lang == 'en' for ln in lines)
    assert lines[0].units == [('Hello', 'hello'), ("don't", "don't"), ('go', 'go')]
    assert lines[1].units[0] == ("You're", "you're")
    assert lrc._norm(lines[0].text) != lrc._norm(lines[1].text)
    assert lrc.parse('[00:01]迎着风\n[00:05]向前走', 10)[0].lang == 'zh'


@pytest.fixture
def vb():
    v = Voicebank.__new__(Voicebank)
    v.__dict__['dictionaries'] = {'en': {'hello': ['en/hh', 'en/ax', 'en/l', 'en/ow']}}
    v.__dict__['vowels'] = {'SP', 'AP', 'en/ax', 'en/ow'}
    return v


def test_english_dictionary_matches_voicebank_and_rejects_missing_words(tmp_path, monkeypatch, vb):
    monkeypatch.setattr(settings, 'ALIGNER', 'sofa')
    monkeypatch.setattr(settings, 'HUBERTFA_MODEL', tmp_path/'model.onnx')
    (tmp_path/'vocab.json').write_text(json.dumps({'vocab': dict.fromkeys(vb.g2p('hello', 'en'), 1)}))
    assert tools.alignment_backend('en') == 'hubertfa'
    segs = extract.write_segments([lrc.Line(0, 1, 'Hello', lang='en')],
                                  np.zeros(44100), tmp_path/'segments', vb)
    assert segs[0]['backend'] == 'hubertfa'
    assert (tmp_path/'hubertfa_dict.txt').read_text() == 'hello\thh ax l ow\n'
    assert json.loads((tmp_path/'align_meta.json').read_text())['lang'] == 'en'
    with pytest.raises(ValueError, match='missing.*unknown'):
        extract.write_segments([lrc.Line(0, 1, 'unknown', lang='en')],
                               np.zeros(44100), tmp_path/'segments', vb)


def test_multisyllable_word_preserves_vowels_phones_and_score_roundtrip(vb):
    seg = dict(name='clip', offset=0, chars=['Hello'], pinyin=['hello'],
               line_of=[0], texts=['Hello'], lang='en', backend='hubertfa')
    phones = [['en/hh', .3, .4], ['en/ax', .4, .7], ['en/l', .7, .8], ['en/ow', .8, 1.3]]
    aligned = {'clip': dict(length=1.6, words=[['hello', .3, 1.3]], phones=phones)}
    phrases, report = extract.build_phrases(vb, [seg], aligned, [], np.full(150, 261.6))
    syls = phrases[0].syllables
    assert [s.start for s in syls] == [.4, .8]
    assert [s.pinyin for s in syls] == ['hello~0', 'hello~1']
    assert sum((vb.g2p(s.pinyin, 'en') for s in syls), []) == vb.g2p('hello', 'en')
    assert report[0]['syllables'] == 2
    encoded, _, word_dur, _, notes = score_inputs(vb, phrases[0])
    assert [p for p in encoded if p not in ('SP', 'AP')] == vb.g2p('hello', 'en')
    assert word_dur.sum() == notes.dur.sum()
    assert all(d > 0 for d in word_dur)
    with pytest.raises(KeyError):
        vb.g2p('hello~4', 'en')
