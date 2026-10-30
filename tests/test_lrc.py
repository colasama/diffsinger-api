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


def test_parse_colon_fractions_mixed_with_standard_timestamps():
    lines = lrc.parse("[00:00.00]作词：某人\n"
                      "[00:12:95][01:12:950]空を見上げて\n"
                      "[00:15.520]風と歌おう\n"
                      "[01:15:52]風と歌おう\n"
                      "[01:60:10]不合法时间\n[02:00:00]超出歌曲", 120)
    assert [line.start for line in lines] == pytest.approx([12.95, 15.52, 72.95, 75.52])
    assert lines[0].end == pytest.approx(15.52)
    assert all(line.lang == "ja" and line.end > line.start for line in lines)
    assert all("[" not in line.text and ":" not in line.text for line in lines)
    i, j = lrc.pick_random(lines, seed=7)
    assert j > i


def test_equivalent_colon_and_dot_timestamps_are_deduplicated():
    lines = lrc.parse("[00:12:95]空を見上げて\n"
                      "[00:12.950]抬头看天空\n[00:15:52]風と歌おう", 20)
    assert [line.text for line in lines] == ["空を見上げて", "風と歌おう"]
    assert lines[0].end == pytest.approx(15.52)


@pytest.mark.parametrize("stamp, expected", [
    ("[0:1]", 1), ("[00:01.5]", 1.5), ("[00:01,125]", 1.125),
    ("[00:01:125]", 1.125), ("[ 00 : 01 . 125 ]", 1.125),
    ("［００：０１．１２５］", 1.125), ("【00：01，125】", 1.125),
    ("[01:02:03.125]", 3723.125), ("[01:02:03,125]", 3723.125),
    ("[123:01.125]", 7381.125),
])
def test_timestamp_variants_preserve_lyric_text(stamp, expected):
    lines = lrc.parse("\ufeff" + stamp + "風とＡＢＣを歌おう\r\n", 8000)
    assert len(lines) == 1
    assert lines[0].start == pytest.approx(expected)
    assert lines[0].text == "風とＡＢＣを歌おう"


@pytest.mark.parametrize("tag, expected", [
    ("[offset:+500]", .5), ("[offset:-500]", 1.5),
    ("［ offset ： +500 ］", .5), ("[offset:bad]", 1),
    ("[offset:100][offset:-250]", 1.25),
])
def test_offset_is_global_and_last_valid_value_wins(tag, expected):
    lines = lrc.parse("[00:01]迎着风\n" + tag, 10)
    assert lines[0].start == pytest.approx(expected)


def test_offset_filters_bounds_after_adjustment_and_shifts_empty_boundaries():
    lines = lrc.parse("[00:00.1]天空\n[00:01]迎着风\n[00:02]\n"
                      "[00:10.1]歌唱\n[offset:500]", 10)
    assert [line.start for line in lines] == pytest.approx([.5, 9.6])
    assert [line.end for line in lines] == pytest.approx([1.5, 10])


def test_enhanced_lyrics_keep_words_and_repeat_line_without_word_duplicates():
    lines = lrc.parse("[00:01][00:11]<00:01>Hello <00:02,5>world<00:03>\n"
                      "[00:21]迎<00:21:50>着风<00:23>", 30)
    assert [(line.start, line.end, line.text) for line in lines] == [
        (1, 3, "Hello world"), (11, 13, "Hello world"), (21, 23, "迎着风")]


def test_word_only_and_inline_square_timestamps_do_not_repeat_whole_lyrics():
    lines = lrc.parse("＜00：01＞空を＜00：02＞見て＜00：03＞\n"
                      "[00:11]Hello [00:12]world", 20)
    assert [line.start for line in lines] == [1, 11]
    assert [line.text for line in lines] == ["空を見て", "Hello world"]
    assert lines[0].end == 3


def test_metadata_and_colons_in_actual_lyrics():
    lines = lrc.parse("[ti:A song][ar:An artist]\n[00:01][lang:en]I say: hello\n"
                      "[00:05]君は言う：歌おう\n[00:10]作词：某人\n"
                      "[00:11]作詞：誰か\n[00:12]Vocals: Someone\n"
                      "[00:15]Music is in the air", 20)
    assert [line.text for line in lines] == ["I say: hello", "君は言う：歌おう", "Music is in the air"]


@pytest.mark.parametrize("stamp", ["[00:60.1]", "[01:60:03.5]", "[00:-1]", "[00:NaN]",
                                     "[00:01.2.3]", "[00:01:95.1]", "[00:01", "[offset:100]"])
def test_malformed_timestamps_do_not_create_lyrics(stamp):
    assert lrc.parse(stamp + "迎着风", 100) == []
