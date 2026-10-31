import json

import numpy as np
import pytest

from dsapi import extract, lrc, settings
from dsapi.errors import TaskError


class Bank:
    entries = {
        'yu': ['zh/H', 'zh/v'], 'yuan': ['zh/H', 'zh/van'],
        'yue': ['zh/H', 'zh/ve'], 'yun': ['zh/H', 'zh/vn'],
        'yong': ['zh/H', 'zh/iong'], 'ywu': ['zh/H', 'zh/u'],
        'yo': ['zh/y', 'zh/o'], 'ge': ['zh/g', 'zh/e'],
        'go': ['en/g', 'en/ow'], 'moc': ['ja/m', 'ja/o', 'ja/cl'],
    }
    vowels = {'zh/v', 'zh/van', 'zh/ve', 'zh/vn', 'zh/iong', 'zh/u', 'zh/o', 'zh/e', 'en/ow', 'ja/o'}

    def g2p(self, word, language='zh'): return list(self.entries[word])
    def knows(self, word, language): return word in self.entries
    def is_vowel(self, phone): return phone in self.vowels


@pytest.fixture
def dictionaries(tmp_path, monkeypatch):
    chinese = {'yu': ['y0', 'v'], 'yuan': ['y0', 'van'], 'yue': ['y0', 've'],
               'yun': ['y0', 'vn'], 'yong': ['y', 'iong'], 'ywu': ['y', 'u'],
               'yo': ['y', 'io'], 'ge': ['g', 'e']}
    dictionary = tmp_path/'zh.txt'
    dictionary.write_text(''.join(f"{word}\t{' '.join(phones)}\n" for word, phones in chinese.items()))
    monkeypatch.setattr(settings, 'HUBERTFA_DICT', dictionary)
    monkeypatch.setattr(settings, 'HUBERTFA_MODEL', tmp_path/'model.onnx')
    monkeypatch.setattr(settings, 'ALIGNER', 'hubertfa')
    vocab = {f'zh/{p}' for phones in chinese.values() for p in phones}
    vocab.update(['en/g', 'en/ow', 'ja/m', 'ja/o', 'cl'])
    (tmp_path/'vocab.json').write_text(json.dumps({'vocab':dict.fromkeys(vocab, 1)}))
    return chinese


def lines(word='yu'):
    return [lrc.Line(0, 1, '欲', _units=[('欲', word)]),
            lrc.Line(1, 2, 'go', lang='en', _units=[('go', 'go')])]


@pytest.mark.parametrize('word', ['yu', 'yuan', 'yue', 'yun', 'yong', 'ywu', 'yo', 'ge'])
def test_mixed_uses_hubertfa_chinese_and_preserves_synthesis(word, dictionaries, tmp_path):
    bank = Bank()
    segs = extract.write_segments(lines(word), np.zeros(88200), tmp_path/'segments', bank)
    expected = ['zh/'+p for p in dictionaries[word]]
    mapping = segs[0]['phone_mappings']['zh:'+word]
    assert mapping == {'alignment': expected, 'synthesis':bank.entries[word]}
    assert f"zh:{word}\t{' '.join(expected)}\n" in (tmp_path/'hubertfa_dict.txt').read_text()
    assert 'zh/H' not in (tmp_path/'hubertfa_dict.txt').read_text()
    persisted = json.loads((tmp_path/'segments.json').read_text())
    aligned = {'clip': {'length':2, 'words':[['zh:'+word, 0, 1], ['en:go', 1, 2]],
                       'phones':[[expected[0],0,.13], [expected[1],.13,1], ['en/g',1,1.12], ['en/ow',1.12,2]]}}
    phrases, report = extract.build_phrases(bank, persisted, aligned, [], np.full(200,261.6))
    assert phrases[0].phones == [(bank.entries[word][0],0,.13), (bank.entries[word][1],.13,1)]
    assert phrases[0].syllables[0].start == .13
    assert phrases[1].phones == [('en/g',1,1.12), ('en/ow',1.12,2)]
    assert not any('error' in r for r in report)


def test_pure_chinese_uses_same_alignment_pronunciation(dictionaries, tmp_path):
    segs = extract.write_segments(lines()[:1],np.zeros(44100),tmp_path/'segments',Bank())
    assert segs[0]['dictionary'] == str(settings.HUBERTFA_DICT)
    assert dictionaries['yu'] == ['y0','v']
    # The existing monolingual mapping is unchanged.
    assert extract._map_phones(Bank(),'yu',[('y0',0,.2),('v',.2,1)]) == [('zh/H',0,.2),('zh/v',.2,1)]


@pytest.mark.parametrize('target', [['zh/H'], ['zh/h','zh/v'], ['zh/H','zh/e'], ['zh/H','zh/v','zh/n']])
def test_unverified_correspondence_is_rejected(target, dictionaries):
    with pytest.raises(TaskError, match='Unsupported Chinese phoneme mapping'):
        extract._chinese_phone_mapping('yu',target,{'yu':['zh/y0','zh/v']})
    with pytest.raises(TaskError):
        extract._chinese_phone_mapping('ge',['zh/H','zh/e'],{'ge':['zh/y','zh/e']})


@pytest.mark.parametrize('actual', [[], [('zh/v',0,1)], [('zh/h',0,.2),('zh/v',.2,1)],
                                    [('zh/y0',0,.1),('zh/v',.1,.8),('zh/n',.8,1)]])
def test_aligner_output_must_match_saved_mapping(actual):
    mapping = {'alignment':['zh/y0','zh/v'], 'synthesis':['zh/H','zh/v']}
    with pytest.raises(TaskError, match='validated mapping'):
        extract._map_phones(Bank(),'yu',actual,'zh',mapping)


def test_missing_dictionary_entry_and_unsupported_aligner_phone(dictionaries,tmp_path):
    settings.HUBERTFA_DICT.write_text('ge\tg e\n')
    with pytest.raises(TaskError, match='missing from HubertFA dictionary: yu'):
        extract.write_segments(lines(),np.zeros(88200),tmp_path/'segments',Bank())
    settings.HUBERTFA_DICT.write_text('yu\ty0 v\n')
    (tmp_path/'vocab.json').write_text(json.dumps({'vocab':{'zh/v':1,'en/g':2,'en/ow':3}}))
    with pytest.raises(TaskError, match='missing.*zh/y0'):
        extract.write_segments(lines(),np.zeros(88200),tmp_path/'segments',Bank())


def test_chunked_mixed_keeps_mappings_and_context(dictionaries,tmp_path):
    text = [lrc.Line(t,t+2,'欲 go') for t in (0,30,65)]
    segs = extract.write_segments(text,np.zeros(70*44100,np.float32),tmp_path/'segments',Bank())
    assert len(segs) == 2
    assert all(s['phone_mappings']['zh:yu']['alignment'] == ['zh/y0','zh/v'] for s in segs)
    assert sum(sum(k>=0 for k in s['line_of']) for s in segs) == 6
    assert len({s['dictionary'] for s in segs}) == 1
    assert 'zh:yu\tzh/y0 zh/v' in open(segs[0]['dictionary']).read()


def test_japanese_closure_mapping_is_unchanged(dictionaries,tmp_path):
    text = lines()+[lrc.Line(2,3,'モッ',lang='ja',_units=[('モッ','moc')])]
    extract.write_segments(text,np.zeros(132300),tmp_path/'segments',Bank())
    assert 'ja:moc\tja/m ja/o cl\n' in (tmp_path/'hubertfa_dict.txt').read_text()
