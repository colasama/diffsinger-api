import pytest
from dsapi import lrc, lyric_match, selection, resolver, sources


def lines(step=20):
    return [lrc.Line(i*step, i*step+3, text) for i, text in enumerate(
        ['迎着风向前走', '抬头仰望那天空', '我们一起去远方', '那里有新的希望', '最后唱给你听', '未指定的尾声'])]


@pytest.mark.parametrize('separator', ['\n', '。', ''])
def test_full_range_over_four_lines_and_instrumental_gaps(separator):
    song = lines()
    query = separator.join(ln.text for ln in song[:5])
    result = selection.select(song, 200, lyric_query=query)
    assert result.strategy == 'lyric_exact'
    assert result.lines == song[:5]
    assert result.start == 0 and result.end == 84
    assert result.match['line_count'] == 5


def test_three_short_lines_not_padded_and_two_lines_keep_window():
    song = lines(3)
    result = selection.select(song, 100, lyric_query='\n'.join(l.text for l in song[:3]))
    assert result.lines == song[:3] and result.end == 10
    short = selection.select(song, 100, lyric_query='\n'.join(l.text for l in song[:2]))
    assert short.strategy == 'lyric' and len(short.lines) > 2


def test_incomplete_explicit_lyrics_fail_without_random_fallback():
    song = lines()
    query = '\n'.join(l.text for l in song[:2]) + '\n不存在的第三句'
    with pytest.raises(ValueError, match='完整歌词未匹配'):
        selection.select(song, 200, lyric_query=query)
    with pytest.raises(ValueError, match='完整歌词未匹配'):
        selection.select(song[:2], 200, lyric_query='\n'.join(l.text for l in song[:3]))
    # Splitting a single line into three pieces cannot unlock long selection.
    assert lyric_match.full_match(song, '迎着\n风向\n前走') is None


def test_extent_overrides_and_manual_precedence():
    song = lines()
    query = '\n'.join(l.text for l in song[:3])
    manual = selection.select(song, 200, start=0, end=4, lyric_query=query)
    assert manual.strategy == 'manual' and manual.lines == song[:1]
    window = selection.select(song, 200, lyric_query=query, lyric_extent='window', seed=1)
    assert window.strategy == 'random' and window.end-window.start <= 40
    exact = selection.select(song, 200, lyric_query=song[0].text, lyric_extent='exact')
    assert exact.strategy == 'lyric_exact' and exact.lines == song[:1]


def test_long_resolver_searches_excerpt_but_verifies_every_line():
    song = lines()
    query = '\n'.join(l.text for l in song[:5])
    class Source(sources.Source):
        def search(self, q, limit=10):
            assert q != query
            return [sources.Song('partial', 'wrong', []), sources.Song('full', 'right', [])]
        def lyric(self, song_id):
            chosen = song[:2] if song_id == 'partial' else song
            return '\n'.join(f'[{int(l.start)//60:02}:{int(l.start)%60:02}]{l.text}' for l in chosen)
    result = resolver.resolve(Source(), query)
    assert result['song']['id'] == 'full' and result['lyric_query'] == query
    assert result['match']['line_count'] == 5


def test_partial_edge_lines_still_cover_requested_three_sentences():
    song = lines()
    query = song[0].text[2:]+'\n'+song[1].text+'\n'+song[2].text[:-2]
    result = selection.select(song, 200, lyric_query=query)
    assert result.strategy == 'lyric_exact' and result.lines == song[:3]


def test_explicit_parser_keeps_long_held_line_end():
    text = '[00:00]啦\n[00:20]啦\n[00:40]啦\n[01:10]下一句'
    assert lrc.parse(text, 80)[2].end < 50
    assert lrc.parse(text, 80, trim_gaps=False)[2].end == 70


def test_manual_long_range_keeps_held_lines_and_audio_boundary():
    text = '[00:00]啦\n[00:20]啦\n[00:40]啦\n[01:10]下一句'
    song = lrc.parse(text, 80, trim_gaps=False)
    result = selection.select(song, 80, start=0, end=120)
    assert result.strategy == 'manual' and result.lines == song
    assert result.start == 0 and result.end == 80
