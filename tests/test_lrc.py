"""Lyric-only selection should prefer sustained singing over sparse repeats."""

import pytest

from dsapi import lrc


TEXTS = ["迎着风向前走", "让阳光照进来", "把梦想放在心", "我们一起歌唱", "看星光落下来", "等明天的花开"]


def block(start, step=5.0):
    return [lrc.Line(start + i * step, start + (i + 1) * step, text)
            for i, text in enumerate(TEXTS)]


def test_no_repeats_searches_past_sparse_opening():
    opening = [lrc.Line(i * 20, i * 20 + 2, text)
               for i, text in enumerate(["天空", "大地", "远方", "此刻"])]
    lines = opening + block(100)
    i, j = lrc.pick_chorus(lines)
    assert (i, j) == (4, 9)
    assert lines[j - 1].end - lines[i].start == 25


def test_evaluates_endings_instead_of_always_filling_max_length():
    assert lrc.pick_chorus(block(0)) == (0, 5)


def test_dense_verse_beats_highly_repeated_sparse_lines():
    lines = [lrc.Line(i * 8, i * 8 + 3, "啦啦啦") for i in range(12)] + block(110)
    i, j = lrc.pick_chorus(lines)
    assert i == 12
    assert j == 17


def test_repetition_still_prefers_chorus_when_coverage_is_equal():
    verse = [lrc.Line(i * 5, (i + 1) * 5, text + "啊") for i, text in enumerate(TEXTS)]
    lines = verse + block(50) + block(100)
    assert lrc.pick_chorus(lines) == (6, 11)


def test_never_bridges_long_instrumental_gap_even_for_target_duration():
    lines = [lrc.Line(0, 8, "迎着风向前走"), lrc.Line(17, 25, "我们一起歌唱")]
    assert lrc.pick_chorus(lines) == (0, 1)


def test_overlapping_intervals_do_not_inflate_coverage():
    lines = [lrc.Line(0, 10, text) for text in TEXTS] + block(50)
    i, j = lrc.pick_chorus(lines)
    assert i >= 6
    assert lines[j - 1].end - lines[i].start == 25


@pytest.mark.parametrize("lines", [[], [lrc.Line(0, 0, "风")],
                                    [lrc.Line(0, 40, "风")],
                                    [lrc.Line(0, float("nan"), "风")]])
def test_no_usable_window_returns_empty_range(lines):
    assert lrc.pick_chorus(lines) == (0, 0)


def test_short_song_and_custom_maximum():
    assert lrc.pick_chorus([lrc.Line(1, 4, "迎着风")]) == (0, 1)
    assert lrc.pick_chorus(block(0), max_len=10) == (0, 2)


@pytest.mark.parametrize("kwargs", [{"target": 0}, {"max_len": -1}, {"target": float("inf")}])
def test_invalid_window_parameters(kwargs):
    with pytest.raises(ValueError, match="finite and positive"):
        lrc.pick_chorus(block(0), **kwargs)


def test_parse_removes_instrumental_labels_but_preserves_their_boundaries():
    lines = lrc.parse("[00:00]作词：某人\n[00:05]【前奏】\n"
                      "[00:20]迎着风向前走\n[00:22]（间奏）\n"
                      "[00:40]我们一起歌唱\n[00:43]\n"
                      "[00:50]纯音乐，请欣赏\n[00:55][Outro]", 60)
    assert [(line.start, line.end) for line in lines] == [(20, 22), (40, 43)]


def test_parse_duplicate_timestamps_keep_one_positive_length_lyric():
    lines = lrc.parse("[00:01]迎着风\n[00:01]迎着风\n[00:01]\n"
                      "[00:01]We face the wind\n[00:05]向前走", 10)
    assert [line.text for line in lines] == ["迎着风", "向前走"]
    assert all(line.end > line.start for line in lines)


def test_parse_multiple_timestamps_and_audio_bounds():
    lines = lrc.parse("[00:01][00:07]迎着风向前走\n"
                      "[00:60]不合法时间\n[00:10]超出歌曲\n[00:20]尾声", 10)
    assert [line.start for line in lines] == [1, 7]
    assert lines[-1].end == 10


def test_english_instrumental_labels_are_not_lyrics():
    lines = lrc.parse("[00:00](Intro)\n[00:10]Here comes the sun\n"
                      "[00:15]We walk along the road\n[00:20][Instrumental]", 50)
    assert len(lines) == 2
    assert all(line.lang == "en" for line in lines)
    assert lrc.pick_chorus(lines) == (0, 2)
