import threading

import pytest

from dsapi import resolver, sources


class Source(sources.Source):
    def __init__(self, songs, lyrics=None):
        self.songs = songs
        self.lyrics = lyrics or {}

    def search(self, query, limit=10):
        return self.songs[:limit]

    def lyric(self, song_id):
        return self.lyrics[song_id]


@pytest.fixture(autouse=True)
def clear_cache():
    with resolver._lock:
        resolver._cache.clear()


def song(id, name='雨爱', artists=None):
    return sources.Song(id, name, artists or ['杨丞琳'], 120000)


@pytest.mark.parametrize('query,name,artist', [
    ('雨爱 杨丞琳', '雨爱', '杨丞琳'),
    ('雨爱   杨丞琳', '雨爱', '杨丞琳'),
    ('"Love Story" "Taylor Swift"', 'Love Story', 'Taylor Swift'),
    ('love story taylor swift', 'Love Story', 'Taylor Swift'),
])
def test_title_artist_selects_correct_performer_without_lyric_fetch(query, name, artist):
    source = Source([song('cover', name, ['其他歌手']), song('original', name, [artist])])
    resolved = resolver.resolve(source, query)
    assert resolved['song']['id'] == 'original'
    assert resolved['match_type'] == 'title_artist' and resolved['lyric_query'] is None


def test_title_wins_over_lyric_and_multi_artist_matches():
    source = Source([song('1'), song('2', '迎着风向前走', ['甲', '乙'])],
                    {'1': '[00:10]迎着风向前走'})
    assert resolver.resolve(source, '迎着风向前走')['match_type'] == 'title'
    assert resolver.resolve(source, '迎着风向前走 乙')['match_type'] == 'title_artist'
    assert resolver.resolve(source, '迎着风向前走 甲 乙')['match_type'] == 'title_artist'


def test_lyrics_rerank_candidates_skip_errors_and_prefer_exact():
    source = Source([song('broken'), song('partial'), song('exact')], {
        'partial': '[00:10]迎着风向前走吧', 'exact': '[00:20]迎着风向前走',
    })
    result = resolver.resolve(source, '迎着风向前走')
    assert result['song']['id'] == 'exact'
    assert result['match_type'] == 'lyric'
    assert result['lyric_query'] == '迎着风向前走' and result['match']['start'] == 20


def test_no_match_fallback_empty_and_invalid_queries():
    source = Source([song(''), song('1')])
    result = resolver.resolve(source, '不存在的一句歌词')
    assert result['song']['id'] == '1' and result['match_type'] == 'fallback'
    assert result['lyric_query'] is None and result['fallback_reason']
    assert resolver.resolve(Source([]), '雨爱') is None
    with pytest.raises(ValueError):
        resolver.resolve(source, ' ！！！ ')


def test_cache_avoids_repeated_lyric_requests():
    calls = []

    class Cached(Source):
        def lyric(self, song_id):
            calls.append(song_id)
            return '[00:10]迎着风向前走'

    for _ in range(2):
        assert resolver.resolve(Cached([song('1')]), '迎着风向前走')['match_type'] == 'lyric'
    assert calls == ['1']


def test_slow_provider_does_not_hold_resolution_past_deadline(monkeypatch):
    release = threading.Event()
    finished = threading.Event()

    class Slow(Source):
        def lyric(self, song_id):
            try:
                release.wait(2)
                return '[00:10]迎着风向前走'
            finally:
                finished.set()

    monkeypatch.setattr(resolver, 'LYRIC_TIMEOUT', .01)
    try:
        result = resolver.resolve(Slow([song('1')]), '迎着风向前走')
        assert not finished.is_set()
        assert result['match_type'] == 'fallback' and result['fallback_reason']
    finally:
        release.set()
        finished.wait(2)
