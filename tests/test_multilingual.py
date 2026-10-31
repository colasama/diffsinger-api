"""Mixed lyrics retain all units, language conditioning and a shared timeline."""
import json

import numpy as np

from dsapi import extract, lrc, settings, tools


def test_japanese_enabled_by_default_and_english_inside_a_line(monkeypatch):
    monkeypatch.delenv('DSAPI_EXPERIMENTAL_JA', raising=False)
    lines = lrc.parse('[00:01]We sing\n[00:05]君と hello 世界へ\n[00:10]We go', 14)
    assert [ln.lang for ln in lines] == ['en', 'ja', 'en']
    units = lines[1].routed_units
    assert any(g == 'hello' and lang == 'en' for _, g, lang in units)
    assert any(lang == 'ja' for _, _, lang in units)
    assert ''.join(d for d, _, lang in units if lang == 'ja') == 'キミトセカイエ'
    chinese = lrc.parse('[00:01]迎着风 hello 向前走', 10)[0]
    assert [lang for _, _, lang in chinese.routed_units] == ['zh'] * 3 + ['en'] + ['zh'] * 3


class Bank:
    vowels = {'zh/e', 'ja/o', 'en/ow'}
    entries = {('ge', 'zh'): ['zh/g', 'zh/e'], ('go', 'en'): ['en/g', 'en/ow'],
               ('go', 'ja'): ['ja/g', 'ja/o']}

    def knows(self, unit, lang):
        return (unit, lang) in self.entries

    def g2p(self, unit, lang):
        return self.entries[(unit, lang)]

    def is_vowel(self, phone):
        return phone in self.vowels


def test_one_alignment_keeps_language_namespaces_and_separates_game_phrases(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, 'HUBERTFA_MODEL', tmp_path/'model.onnx')
    monkeypatch.setattr(settings, 'HUBERTFA_DICT', tmp_path/'zh.txt')
    (tmp_path/'zh.txt').write_text('ge\tg e\n')
    (tmp_path/'vocab.json').write_text(json.dumps({'vocab': dict.fromkeys(
        [p for phones in Bank.entries.values() for p in phones], 1)}))
    lines = [lrc.Line(0, 1, 'go', lang='en', _units=[('go', 'go')]),
             lrc.Line(1, 2, 'ゴ', lang='ja', _units=[('ゴ', 'go')])]
    context = lrc.Line(-1, 0, '歌', _units=[('歌', 'ge')])
    segs = extract.write_segments(lines, np.zeros(88200), tmp_path/'segments', Bank(), before=[context], clip_start=-1)
    seg = segs[0]
    assert seg['lang'] == 'mixed' and seg['backend'] == 'hubertfa'
    assert seg['pinyin'] == ['zh:ge', 'en:go', 'ja:go']
    assert seg['word_langs'] == ['zh', 'en', 'ja']
    dictionary = (tmp_path/'hubertfa_dict.txt').read_text()
    assert 'en:go\ten/g en/ow\n' in dictionary
    assert 'ja:go\tja/g ja/o\n' in dictionary
    assert json.loads((tmp_path/'segments.json').read_text())[0]['word_units'] == ['ge', 'go', 'go']
    aligned = {'clip': {'length': 3, 'words': [['zh:ge', 0, 1], ['en:go', 1, 2], ['ja:go', 2, 3]],
                       'phones': [['g', 0, .1], ['e', .1, 1], ['g', 1, 1.1], ['ow', 1.1, 2],
                                  ['g', 2, 2.1], ['o', 2.1, 3]]}}
    phrases, report = extract.build_phrases(Bank(), segs, aligned, [], np.full(300, 261.6))
    assert [p.syllables[0].lang for p in phrases] == ['en', 'ja']
    assert [p.syllables[0].start for p in phrases] == [1.1, 2.1]
    assert all(len({s.lang for s in p.syllables}) == 1 for p in phrases)
    assert [r['lang'] for r in report] == ['en', 'ja']
    assert all(s.lang != 'zh' for p in phrases for s in p.syllables)  # Context is not rendered.
    def run(vendor, script, *args, log):
        from pathlib import Path
        regions = json.loads(Path(args[2]).read_text())
        assert [r['language'] for r in regions] == ['en', 'ja']
        Path(args[3]).write_text(json.dumps({'notes': [{'midi': 60}]}))
    monkeypatch.setattr(tools, '_run', run)
    tools.extract_game_notes(tmp_path/'lead.wav', tmp_path, phrases, 'mixed')


def test_english_and_japanese_default_routing(monkeypatch):
    monkeypatch.setattr(settings, 'ALIGNER', 'sofa')
    monkeypatch.setattr(settings, 'JA_ALIGNER', 'hubertfa')
    assert tools.alignment_backend('zh') == 'sofa'
    assert all(tools.alignment_backend(lang) == 'hubertfa' for lang in ('en', 'ja', 'mixed'))
