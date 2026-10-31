"""One lyric selection shared by cropping, alignment context and job reporting."""
import math
import secrets
from dataclasses import dataclass

from . import lrc, lyric_match


@dataclass
class Selection:
    lines: list[lrc.Line]
    before: list[lrc.Line]
    after: list[lrc.Line]
    start: float
    end: float
    strategy: str
    seed: int | None
    lyric_query: str | None = None
    match: dict | None = None
    fallback_reason: str | None = None

    def metadata(self) -> dict:
        return {"strategy": self.strategy, "seed": self.seed,
                "lyric_window": [self.lines[0].start, self.lines[-1].end],
                "audio_window": [self.start, self.end],
                "languages": sorted({lang for line in self.lines for _, _, lang in line.routed_units}),
                "lyric_query": self.lyric_query, "match": self.match, "fallback_reason": self.fallback_reason}


def select(lines: list[lrc.Line], duration: float, *, start: float | None = None,
           end: float | None = None, strategy: str = "random", seed: int | None = None,
           max_clip: float = 40., lyric_query: str | None = None, lyric_extent: str = "auto") -> Selection:
    if lyric_extent not in ("auto", "window", "exact"):
        raise ValueError("lyric_extent must be auto, window, or exact")
    if lyric_extent == "exact" and not lyric_query and start is None:
        raise ValueError("exact selection requires lyrics")
    if strategy not in ("random", "chorus"):
        raise ValueError("selection must be random or chorus")
    if (start is None) != (end is None):
        raise ValueError("start and end must be supplied together")
    match, fallback_reason = None, None
    if start is not None:
        if not all(math.isfinite(t) for t in (start, end)) or not 0 <= start < end:
            raise ValueError("start/end must be finite and 0 <= start < end")
        chosen = [ln for ln in lines if ln.start >= start - .5 and ln.end <= end + 1.]
        strategy, seed = "manual", None
    else:
        max_len = min(32., max_clip - 2.5)
        full = lyric_match.full_match(lines, lyric_query, lyric_extent) if lyric_query else None
        if full:
            i, j = full.start, full.stop
            match = full.metadata(lines)
            strategy, seed = "lyric_exact", None
        elif lyric_query and lyric_match.requires_full_match(lyric_query, lyric_extent):
            raise ValueError("完整歌词未匹配，请检查歌词内容与歌曲版本；不会改唱随机片段")
        anchor = lyric_match.pick_window(lines, lyric_query, max_len) if lyric_query and not full else None
        if full:
            pass
        elif anchor:
            i, j, hit = anchor
            match = hit.metadata(lines)
            strategy, seed = "lyric", None
        elif strategy == "random":
            seed = seed if seed is not None else secrets.randbits(32)
            i, j = lrc.pick_random(lines, max_len=min(32., max_clip - 2.5), seed=seed)
        else:
            i, j = lrc.pick_chorus(lines, max_len=min(32., max_clip - 2.5))
        if lyric_query and anchor is None and not full:
            fallback_reason = "no_usable_lyric_match"
        chosen = lines[i:j]
    if not chosen:
        if not lines:
            raise ValueError("no singable lyric lines parsed from LRC within the audio duration; check lyric timestamps and content")
        if strategy == "manual":
            raise ValueError("no complete lyric lines in the requested time window")
        raise ValueError("no continuous singable lyric window fits the clip limit")
    lo = max(0., chosen[0].start - 1.5)
    # Keep whole lyric lines; never feed a full line over a truncated waveform.
    limit = duration if strategy in ("lyric_exact", "manual") else min(duration, lo + max_clip)
    if strategy == "lyric_exact" and any(ln.end > duration for ln in chosen):
        raise ValueError("Audio does not cover the complete requested lyrics")
    chosen = [ln for ln in chosen if ln.end <= limit]
    if not chosen:
        raise ValueError("no complete lyric line fits the clip limit")
    hi = min(limit, chosen[-1].end + 1.)
    k0, k1 = lines.index(chosen[0]), lines.index(chosen[-1])
    before = [ln for ln in lines[max(0, k0 - 1):k0] if ln.end > lo - .5]
    after = [ln for ln in lines[k1 + 1:k1 + 2] if ln.start < hi + .5]
    return Selection(chosen, before, after, lo, hi, strategy, seed, lyric_query, match, fallback_reason)
