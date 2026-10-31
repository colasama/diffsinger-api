"""Resolve a title, title + artist, or lyric without downloading audio."""
import logging
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import asdict
from itertools import zip_longest
from threading import BoundedSemaphore, Lock

from . import lrc, lyric_match, sources

_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="lyric-resolve")
_slots = BoundedSemaphore(10)  # includes queued work, even after request timeout
_cache: OrderedDict[tuple, tuple[float, str]] = OrderedDict()
_lock = Lock()
LYRIC_TIMEOUT = 8.
_log = logging.getLogger(__name__)


def _lyric(source: sources.Source, song_id: str) -> str:
    key = (type(source), repr(sorted(vars(source).items())), song_id)
    with _lock:
        cached = _cache.get(key)
        if cached and time.monotonic() - cached[0] < 300:
            _cache.move_to_end(key)
            return cached[1]
    text = source.lyric(song_id)
    with _lock:
        _cache[key] = (time.monotonic(), text)
        _cache.move_to_end(key)
        while len(_cache) > 128:
            _cache.popitem(last=False)
    return text


def resolve(source: sources.Source, query: str, lyric_extent: str = "auto") -> dict | None:
    query = query.strip()
    needle = lyric_match.tokens(query)
    if not needle:
        raise ValueError("query must contain a song title or lyrics")
    songs = []
    seen = set()
    parts = lyric_match.sentences(query)
    queries = [query]
    if len(parts) >= 3 or len(query) > 512:
        # Search excerpts, but only accept candidates verified against ALL lyrics.
        queries = list(dict.fromkeys([max(parts, key=len)[:200], parts[0][:200], parts[-1][:200]]))
    candidates = [source.search(q, limit=10)[:10] for q in queries]
    for row in zip_longest(*candidates):
        for song in row:
            if song is not None and song.id.strip() and song.id not in seen and len(songs) < 10:
                seen.add(song.id)
                songs.append(song)
    if not songs:
        return None

    def result(song, kind, **extra):
        return {"song": asdict(song), "match_type": kind, "lyric_query": None, **extra}

    for song in songs:
        if needle == lyric_match.tokens(song.name):
            return result(song, "title")
    for song in songs:
        # Compare full metadata, rather than splitting multiword titles/artists.
        artists = [*song.artists, " ".join(song.artists)]
        if any(artist.strip() and needle == lyric_match.tokens(song.name + " " + artist)
               for artist in artists):
            return result(song, "title_artist")

    futures = {}
    for index, song in enumerate(songs):
        if not _slots.acquire(blocking=False):
            break
        try:
            future = _pool.submit(_lyric, source, song.id)
        except Exception:
            _slots.release()
            raise
        future.add_done_callback(lambda _: _slots.release())
        futures[future] = (index, song)
    done, pending = wait(futures, timeout=LYRIC_TIMEOUT)
    for future in pending:
        future.cancel()
    hits = []
    for future in done:
        index, song = futures[future]
        try:
            # Unknown duration is provisional; the job rechecks against audio.
            duration = song.duration_ms / 1000 if song.duration_ms and song.duration_ms > 0 else 86400.
            lines = lrc.parse(future.result(), duration)
            full = lyric_match.full_match(lines, query, lyric_extent)
            match = None if full else lyric_match.pick_window(lines, query)
            if full:
                if song.duration_ms and song.duration_ms > 0:
                    lines = lrc.parse(future.result(), duration, trim_gaps=False)
                hits.append((full.rank, -index, song, full.metadata(lines)))
                continue
            if lyric_match.requires_full_match(query, lyric_extent):
                continue
        except Exception:
            _log.debug("Candidate lyrics unavailable: song=%s", song.id, exc_info=True)
            continue
        if match:
            i, j, anchor = match
            hits.append((anchor.rank, -index, song, anchor.metadata(lines)))
    if hits:
        _, _, song, match = max(hits, key=lambda hit: hit[:2])
        return result(song, "lyric", lyric_query=query, match=match)
    if lyric_match.requires_full_match(query, lyric_extent):
        raise ValueError("完整歌词未匹配，请检查歌词内容与歌曲版本")
    reason = "lyrics_unavailable_or_no_match" if len(done) < len(songs) else "no_reliable_lyric_match"
    return result(songs[0], "fallback", fallback_reason=reason)
