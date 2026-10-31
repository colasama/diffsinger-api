"""Alignment selection and preservation of singing vowel boundaries."""
import json
from pathlib import Path

import numpy as np
import pytest

from dsapi import extract, lrc, settings, tools


def test_long_continuous_phrase_splits_into_balanced_groups_not_single_syllables():
    # Includes tiny floating-point gaps like those emitted by the aligner.
    spans = [(i * .25 + (1e-8 if i % 3 else 0), (i + 1) * .25) for i in range(120)]
    groups = extract._split(spans)
    assert [i for a, b in groups for i in range(a, b)] == list(range(120))
    assert len(groups) == 4
    assert all(6 <= spans[b - 1][1] - spans[a][0] <= extract.MAX_PHRASE for a, b in groups)


def test_long_phrase_does_not_peel_off_short_leading_syllable():
    spans = [(0, .013)] + [(.013 + i * .25, .013 + (i + 1) * .25) for i in range(80)]
    groups = extract._split(spans)
    assert groups[0][0] == 0 and groups[0][1] > 1
    assert all(b - a > 1 for a, b in groups)


def test_phrase_splitting_preserves_breaths_and_language_boundaries():
    spans = [(0, 1), (1, 1.013), (1.013, 2), (2.5, 3)]
    assert extract._split(spans, ['ja', 'en', 'ja', 'ja']) == [(0, 1), (1, 2), (2, 3), (3, 4)]


@pytest.fixture
def routing(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, 'ALIGNER', 'hubertfa')
    calls = []

    def run(vendor, script, *args, log):
        calls.append((vendor, script, args))
        Path(args[3]).write_text(json.dumps({'clip': {'backend': vendor}}))

    monkeypatch.setattr(tools, '_run', run)
    return tmp_path, calls


def test_default_hubertfa_and_explicit_legacy_sofa(routing):
    d, calls = routing
    assert tools.align(d, d)['clip']['backend'] == 'HubertFA'
    assert calls[-1][2][0] == str(settings.HUBERTFA_MODEL.resolve())
    # Existing checkpoint-based callers must retain SOFA even after default changes.
    tools.align(d, d, settings.SOFA_CKPT_MULTI, d/'ja_dict.txt')
    assert calls[-1][0] == 'SOFA'


def test_sofa_override_and_japanese_route(routing, monkeypatch):
    d, calls = routing
    monkeypatch.setattr(settings, 'JA_ALIGNER', 'sofa')
    assert tools.alignment_backend('ja') == 'sofa'
    monkeypatch.setattr(settings, 'ALIGNER', 'sofa')
    tools.align(d, d)
    assert calls[-1][0] == 'SOFA'
    monkeypatch.setattr(settings, 'ALIGNER', 'typo')
    with pytest.raises(ValueError, match='DSAPI_ALIGNER'):
        tools.align(d, d)


class Voicebank:
    def knows(self, unit, lang):
        return unit in ('guo', 'ge', 'mo_cl')

    def g2p(self, unit, lang):
        return {'guo': ['zh/g', 'zh/uo'], 'ge': ['zh/g', 'zh/e'],
                'mo_cl': ['ja/m', 'ja/o', 'ja/cl']}[unit]

    def is_vowel(self, phone):
        return phone in ('zh/uo', 'zh/e', 'ja/o')


def test_segments_use_selected_dictionary_and_persist_engine(tmp_path, monkeypatch):
    dictionary = tmp_path/'hfa.txt'
    dictionary.write_text('guo\tg uo\nge\tg e\n')
    monkeypatch.setattr(settings, 'ALIGNER', 'hubertfa')
    monkeypatch.setattr(settings, 'HUBERTFA_DICT', dictionary)
    # Default Chinese path must not depend on having the SOFA dictionary installed.
    monkeypatch.setattr(settings, 'SOFA_DICT', tmp_path/'missing.txt')
    segs = extract.write_segments([lrc.Line(0, 2, '过歌')], np.zeros(88200), tmp_path/'segments', Voicebank())
    meta = json.loads((tmp_path/'align_meta.json').read_text())
    assert meta['backend'] == 'hubertfa'
    assert meta['dictionary'] == str(dictionary)
    assert segs[0]['pinyin'] == ['guo', 'ge']


def test_hubertfa_vowels_keep_boundaries_and_confidence_is_unavailable():
    seg = dict(name='clip', offset=0, chars=['过','歌'], pinyin=['guo','ge'],
               line_of=[0,0], texts=['过歌'], lang='zh', backend='hubertfa')
    aligned = {'clip': dict(length=2, confidence=None,
        words=[['guo',0,.8],['ge',1,1.9]],
        phones=[['g',0,.1],['uo',.1,.8],['SP',.8,1],['g',1,1.1],['e',1.1,1.9]])}
    notes = [dict(start=0,dur=2,midi=60,rest=False)]
    phrases, report = extract.build_phrases(Voicebank(),[seg],aligned,notes,np.full(200,261.6))
    assert [s.notes[0].start for p in phrases for s in p.syllables] == [.1,1.1]
    assert [x for p in phrases for x in p.phones if x[0]=='zh/uo'] == [('zh/uo',.1,.8)]
    assert all(r['confidence'] is None and r['backend']=='hubertfa' for r in report)


def test_japanese_keeps_custom_sofa_dictionary(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, 'ALIGNER', 'hubertfa')
    monkeypatch.setattr(settings, 'JA_ALIGNER', 'sofa')
    line = lrc.Line(0, 1, 'モッ', lang='ja', _units=[('モッ', 'mo_cl')])
    segs = extract.write_segments([line], np.zeros(44100), tmp_path/'segments', Voicebank())
    assert segs[0]['backend'] == 'sofa'
    assert segs[0]['ckpt'] == str(settings.SOFA_CKPT_MULTI)
    assert Path(segs[0]['dictionary']).read_text() == 'mo_cl\tm o cl\n'


def test_japanese_hubertfa_preserves_native_phones_and_closure(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, 'JA_ALIGNER', 'hubertfa')
    monkeypatch.setattr(settings, 'HUBERTFA_MODEL', tmp_path/'model.onnx')
    (tmp_path/'vocab.json').write_text(json.dumps({'vocab': {'ja/m': 1, 'ja/o': 2, 'cl': 0}}))
    line = lrc.Line(0, 1, 'モッ', lang='ja', _units=[('モッ', 'mo_cl')])
    segs = extract.write_segments([line], np.zeros(44100), tmp_path/'segments', Voicebank())
    assert segs[0]['backend'] == 'hubertfa'
    assert segs[0]['ckpt'] == str(settings.HUBERTFA_MODEL)
    assert Path(segs[0]['dictionary']).read_text() == 'mo_cl\tm o cl\n'
    (tmp_path/'vocab.json').write_text(json.dumps({'vocab': {'ja/m': 1, 'cl': 0}}))
    with pytest.raises(ValueError, match='missing.*ja/o'):
        extract.write_segments([line], np.zeros(44100), tmp_path/'segments', Voicebank())
    monkeypatch.setattr(settings, 'JA_ALIGNER', 'typo')
    with pytest.raises(ValueError, match='DSAPI_JA_ALIGNER'):
        tools.alignment_backend('ja')


def test_japanese_hubertfa_routes_native_dictionary(routing, monkeypatch):
    d, calls = routing
    monkeypatch.setattr(settings, 'JA_ALIGNER', 'hubertfa')
    tools.align(d, d, lang='ja')
    assert calls[-1][0] == 'HubertFA'
    assert calls[-1][2][2].endswith('japanese_dict_full.txt')
    assert calls[-1][2][-1] == 'ja'


@pytest.mark.parametrize('backend,confidence', [('hubertfa',None), ('sofa',.8)])
def test_evaluation_reuses_original_engine_and_accepts_missing_confidence(tmp_path, monkeypatch, backend, confidence):
    from dsapi import evaluate
    (tmp_path/'segments').mkdir()
    (tmp_path/'segments/clip.lab').write_text('guo')
    result = dict(confidence=confidence, words=[['guo',0,.8]])
    (tmp_path/'sofa.json').write_text(json.dumps({'clip':result}))
    # Legacy SOFA metadata has no backend field.
    meta = dict(ckpt='model.onnx' if backend=='hubertfa' else 'model.ckpt', dictionary='dict.txt')
    if backend=='hubertfa': meta['backend']=backend
    (tmp_path/'align_meta.json').write_text(json.dumps(meta))
    (tmp_path/'mimic.wav').write_bytes(b'placeholder')
    np.save(tmp_path/'f0.npy',np.array([[261.6]*20,[0]*20]))
    monkeypatch.setattr(evaluate,'_f0',lambda _: (np.full(20,261.6),np.ones(20,dtype=bool)))
    monkeypatch.setattr(evaluate.audio,'load',lambda _: np.ones(44100))
    calls=[]
    def align(*args,**kwargs):
        calls.append(kwargs)
        return {'clip':result}
    monkeypatch.setattr(evaluate.tools,'align',align)
    out=evaluate.evaluate(tmp_path)['mimic_intelligibility']
    assert calls[0]['backend']==backend
    assert out['source_confidence']==confidence
    assert out['realign_confidence']==confidence
    assert out['onset_median_ms']==0


def test_long_alignment_chunks_keep_all_lines_and_offsets(tmp_path, monkeypatch):
    dictionary = tmp_path/'hfa.txt'
    dictionary.write_text('ge\tg e\n')
    monkeypatch.setattr(settings, 'ALIGNER', 'hubertfa')
    monkeypatch.setattr(settings, 'HUBERTFA_DICT', dictionary)
    lines = [lrc.Line(float(i), float(i+4), '歌') for i in range(0, 100, 10)]
    segs = extract.write_segments(lines, np.zeros(100*44100, np.float32), tmp_path/'segments', Voicebank())
    assert len(segs) == 3
    assert [s['offset'] for s in segs] == [0, 38, 78]
    assert sum(sum(k>=0 for k in s['line_of']) for s in segs) == 10
    assert all((tmp_path/'segments'/f"{s['name']}.wav").is_file() for s in segs)
    assert len({s['dictionary'] for s in segs}) == 1


def test_evaluate_chunked_alignment_preserves_offsets_and_ignores_context(tmp_path, monkeypatch):
    from dsapi import evaluate
    (tmp_path/'segments').mkdir()
    segments=[{'name':'a','offset':0,'line_of':[0]}, {'name':'b','offset':1,'line_of':[-1,0]}]
    aligned={'a':{'length':1,'words':[['ge',0,.8]]},
             'b':{'length':1,'words':[['ge',0,.2],['ge',.3,.8]]}}
    (tmp_path/'segments.json').write_text(json.dumps(segments))
    (tmp_path/'sofa.json').write_text(json.dumps(aligned))
    for seg in segments:
        (tmp_path/'segments'/f"{seg['name']}.lab").write_text('ge')
    (tmp_path/'mimic.wav').touch()
    np.save(tmp_path/'f0.npy',np.array([[261.6]*20,[0]*20]))
    monkeypatch.setattr(evaluate,'_f0',lambda _: (np.full(20,261.6),np.ones(20,dtype=bool)))
    monkeypatch.setattr(evaluate.audio,'load',lambda _: np.ones(88200))
    monkeypatch.setattr(evaluate.tools,'align',lambda *args,**kwargs: aligned)
    out=evaluate.evaluate(tmp_path)['mimic_intelligibility']
    assert out['words']==2 and out['onset_median_ms']==0
