"""Japanese long vowels: sung pronunciation and morae the aligner left no time."""
import numpy as np

from dsapi import extract, japanese


def test_long_o_is_sung_as_held_vowel_but_displayed_as_written():
    assert japanese.lyric_morae('ありがとう') == [('ア', 'a'), ('リ', 'ri'), ('ガ', 'ga'), ('ト', 'to'), ('ウ', 'o')]
    assert japanese.lyric_morae('そうだね')[:2] == [('ソ', 'so'), ('ウ', 'o')]
    # A verb ending is a real "u"; エイ keeps its written vowel.
    assert japanese.lyric_morae('思う')[-1] == ('ウ', 'u')
    assert japanese.lyric_morae('先生')[-1] == ('イ', 'i')
    assert japanese.lyric_morae('僕は')[-1] == ('ワ', 'wa')


class Voicebank:
    def g2p(self, unit, lang):
        return {'to': ['ja/t', 'ja/o'], 'o': ['ja/o'], 'u': ['ja/u'], 'ka': ['ja/k', 'ja/a'],
                'i': ['ja/i'], 'n': ['ja/N']}[unit]

    def is_vowel(self, phone):
        return phone.split('/')[1] in 'aiueo'


def _build(units, words, phones, line_of=None):
    seg = dict(name='clip', offset=0, chars=[u.upper() for u in units], pinyin=units, word_units=units,
               word_langs=['ja'] * len(units), line_of=line_of or [0] * len(units), texts=['x', 'y'],
               lang='ja', backend='hubertfa')
    aligned = {'clip': dict(length=3, words=words, phones=phones)}
    notes = [dict(start=0, dur=3, midi=60, rest=False)]
    return extract.build_phrases(Voicebank(), [seg], aligned, notes, np.full(300, 261.6))


def test_sliver_long_vowel_joins_previous_note_and_drops_unsung_phone():
    phrases, report = _build(
        ['to', 'u', 'ka'],
        [['to', 0, 1.0], ['u', 1.0, 1.01], ['ka', 1.01, 2]],
        [['t', 0, .1], ['o', .1, 1.0], ['u', 1.0, 1.01], ['k', 1.01, 1.1], ['a', 1.1, 2]])
    syls = phrases[0].syllables
    assert [(s.text, s.pinyin) for s in syls] == [('TOU', 'to'), ('KA', 'ka')]
    assert ('ja/o', .1, 1.01) in phrases[0].phones
    assert not any(p == 'ja/u' for p, _, _ in phrases[0].phones)
    assert report[0]['held_morae'] == 1


def test_sung_vowels_diphthongs_moraic_n_and_line_breaks_are_kept():
    # Long enough to have been sung.
    phrases, report = _build(['to', 'o'], [['to', 0, 1], ['o', 1, 1.2]],
                             [['t', 0, .1], ['o', .1, 1], ['o', 1, 1.2]])
    assert len(phrases[0].syllables) == 2 and report[0]['held_morae'] == 0
    # a+i is a diphthong and ん is a consonant: never absorbed.
    for unit, phone in (('i', 'i'), ('n', 'N')):
        phrases, _ = _build(['ka', unit], [['ka', 0, 1], [unit, 1, 1.01]],
                            [['k', 0, .1], ['a', .1, 1], [phone, 1, 1.01]])
        assert [s.pinyin for s in phrases[0].syllables] == ['ka', unit]
    # Different lyric lines stay separate.
    phrases, _ = _build(['to', 'o'], [['to', 0, 1], ['o', 1, 1.01]],
                        [['t', 0, .1], ['o', .1, 1], ['o', 1, 1.01]], line_of=[0, 1])
    assert sum(len(p.syllables) for p in phrases) == 2


def test_chained_held_morae_join_the_same_note():
    phrases, report = _build(['to', 'o', 'o'], [['to', 0, 1], ['o', 1, 1.01], ['o', 1.01, 1.02]],
                             [['t', 0, .1], ['o', .1, 1], ['o', 1, 1.01], ['o', 1.01, 1.02]])
    assert [(s.text, s.start, s.end) for s in phrases[0].syllables] == [('TOOO', .1, 1.02)]
    assert report[0]['held_morae'] == 2
