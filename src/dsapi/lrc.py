"""LRC parsing and chorus window selection."""

import math
import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from itertools import groupby

from pypinyin import Style, lazy_pinyin
from pypinyin.constants import PHRASES_DICT

# Words where 着 really is zháo when it starts them; pypinyin's dictionary also
# has rare readings like 着风 (catch a chill) that hijack 迎着风.
_ZHAO_HEAD = {"着急", "着迷", "着火", "着凉", "着陆", "着落", "着想", "着手", "着实", "着魔", "着慌", "着数"}

# Three colon-only fields mean mm:ss:fraction; hours require decimal seconds
# to avoid interpreting the provider's [00:12:95] as hours/minutes/seconds.
_CLOCK = r"(?:\d{1,6}:\d{1,2}:\d{1,2}[.,]\d{1,9}|\d{1,6}:\d{1,2}(?:[.,:]\d{1,9})?)"
_TIME = re.compile(r"\[(" + _CLOCK + r")\]")
_WORD_TIME = re.compile(r"<(" + _CLOCK + r")>")
_TAG = re.compile(r"\[[^\]\r\n]*\]|［[^］\r\n]*］|【[^】\r\n]*】|<[^>\r\n]*>|＜[^＞\r\n]*＞")
_OFFSET = re.compile(r"\[offset:([+-]?\d{1,12})\]", re.I)
_HEADER = re.compile(r"\[(?:ar|ti|al|by|re|ve|length|offset|lang|au|id):[^\]]*\]", re.I)
_META = re.compile(
    r"^\s*(?:作词|作詞|作曲|编曲|編曲|混音|母带|调校|调教|曲绘|PV|视频|和声|吉他|贝斯|鼓|弦乐|制作|出品|监制|策划|演唱|填词|原曲)"
    r"|^\s*(?:歌手|歌唱|唄|A&R|Lyrics?|Music|Arrange(?:ment)?|Composer|Singer|Vocals?|Producer)\s*[:：\-]", re.I
)
_HAN = re.compile(r"[一-鿿]")
_KANA = re.compile(r"[\u3040-\u30ff]")
_LATIN = re.compile(r"[A-Za-z]")
_INSTRUMENTAL = re.compile(
    r"^(前奏|间奏|尾奏|纯音乐|纯伴奏|无人声|intro|interlude|outro|instrumental)"
    r"(?:[，,\s]*请欣赏)?$", re.I
)


@dataclass
class Line:
    start: float
    end: float
    text: str
    lang: str = "zh"
    _units: list[tuple[str, str]] | None = field(default=None, repr=False, compare=False)
    _unit_langs: list[str] | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def units(self) -> list[tuple[str, str]]:
        """[(display, grapheme)]: Han/pinyin, kana/romaji, or English words."""
        if self._units is None:
            self._units = self._compute_units()
        return self._units

    def _compute_units(self) -> list[tuple[str, str]]:
        from . import english, japanese

        units, langs = [], []
        # Keep Latin words even inside Japanese/Chinese lines, in sung order.
        for chunk in re.split(r"([A-Za-z]+(?:['’][A-Za-z]+)*)", self.text):
            if _LATIN.search(chunk):
                part, lang = english.words(chunk), "en"
            elif _HAN.search(chunk) or _KANA.search(chunk):
                lang = "ja" if self.lang == "ja" or _KANA.search(chunk) else "zh"
                han = _HAN.findall(chunk)
                part = (japanese.morae(japanese.reading(chunk)) if lang == "ja"
                        else list(zip(han, to_pinyin(han))))
            else:
                continue
            units.extend(part)
            langs.extend([lang] * len(part))
        self._unit_langs = langs
        return units

    @property
    def routed_units(self) -> list[tuple[str, str, str]]:
        units = self.units
        return [(d, g, lang) for (d, g), lang in zip(units, self._unit_langs or [self.lang] * len(units))]

    @property
    def chars(self) -> list[str]:
        return [d for d, _ in self.units]

    @property
    def pinyin(self) -> list[str]:
        return [g for _, g in self.units]


def detect_lang(lines: list["Line"]) -> str:
    """CJK context for ambiguous Han-only lines; never filters other languages."""
    cjk = [ln for ln in lines if _HAN.search(ln.text) or _KANA.search(ln.text)]
    if not cjk:
        return "en"
    return "ja" if sum(bool(_KANA.search(ln.text)) for ln in cjk) / len(cjk) >= 0.3 else "zh"


def to_pinyin(chars: list[str]) -> list[str]:
    out = lazy_pinyin("".join(chars), style=Style.NORMAL, errors="ignore")
    for i, c in enumerate(chars):
        # 着 as an aspect particle (迎着, 朝着) is "zhe"; pypinyin only gets that
        # right inside phrases it knows and otherwise falls back to "zhao".
        if c == "着" and out[i] == "zhao":
            prev = "".join(chars[i - 1:i + 1]) if i else ""
            nxt = "".join(chars[i:i + 2])
            if prev not in PHRASES_DICT and nxt not in _ZHAO_HEAD:
                out[i] = "zhe"
    return out


def _normalize_tag(match: re.Match) -> str:
    original = match[0]
    tag = unicodedata.normalize("NFKC", original).replace("【", "[").replace("】", "]")
    compact = re.sub(r"\s+", "", tag)
    # Normalize syntax only: full-width characters in the lyrics stay intact.
    if _TIME.fullmatch(compact) or _WORD_TIME.fullmatch(compact) or _HEADER.fullmatch(compact):
        return compact
    return original


def _seconds(stamp: str) -> float | None:
    parts = stamp.replace(",", ".").split(":")
    if len(parts) == 3 and "." in parts[2]:
        hours, minutes, seconds = int(parts[0]), int(parts[1]), float(parts[2])
        if minutes >= 60:
            return None
    else:
        hours, minutes = 0, int(parts[0])
        seconds = float(".".join(parts[1:]))
    return hours * 3600 + minutes * 60 + seconds if seconds < 60 else None


def parse(text: str, song_end: float) -> list[Line]:
    text = _TAG.sub(_normalize_tag, text.replace("\ufeff", ""))
    offsets = _OFFSET.findall(text)
    # LRC player correction in milliseconds: positive displays lyrics earlier.
    offset = -int(offsets[-1]) / 1000 if offsets else 0.
    stamped: list[tuple[float, str]] = []
    for raw in text.splitlines():
        raw = _HEADER.sub("", raw).strip()
        times = []
        # Only leading square stamps repeat a line. Inline stamps time words.
        while match := _TIME.match(raw):
            times.append(_seconds(match[1]))
            raw = raw[match.end():].lstrip()
        words = list(_WORD_TIME.finditer(raw))
        if not times and words and words[0].start() == 0:
            times = [_seconds(words[0][1])]
        body = _WORD_TIME.sub("", _TIME.sub("", raw)).strip()
        # Enhanced LRC may end with an explicit empty word marking line end.
        tail = _seconds(words[-1][1]) if words and words[-1].end() == len(raw) else None
        for t in times:
            if t is not None and 0 <= t + offset < song_end:
                stamped.append((t + offset, body))
                if tail is not None and times[0] is not None:
                    end = tail + t - times[0] + offset
                    if t + offset < end < song_end:
                        stamped.append((end, ""))
    stamped.sort(key=lambda x: x[0])
    # Duplicate stamps (including bilingual LRC) share the next *distinct*
    # timestamp; otherwise the first copy gets a zero-length interval.
    groups = [(t, [body for _, body in group]) for t, group in groupby(stamped, key=lambda x: x[0])]
    lines = []
    for i, (t, bodies) in enumerate(groups):
        end = groups[i + 1][0] if i + 1 < len(groups) else song_end
        for body in dict.fromkeys(bodies):
            marker = body.strip(" \t[]【】()（）<>〈〉「」~～-—….")
            if (body and not _META.match(body)
                    and not _INSTRUMENTAL.fullmatch(marker)):
                lines.append(Line(t, end, body))
    context_lang = detect_lang(lines)
    # An instrumental gap would otherwise be glued onto the line before it.
    chosen = []
    seen = set()
    for ln in lines:
        ln.lang = ("ja" if _KANA.search(ln.text) else context_lang if _HAN.search(ln.text) else "en")
        sung_length = sum({"ja": .45, "en": 1.2, "zh": .6}[lang] for _, _, lang in ln.routed_units)
        ln.end = min(ln.end, ln.start + max(2.0, sung_length + 1.5))
        if ln.chars and ln.end > ln.start and ln.start not in seen:
            chosen.append(ln)
            seen.add(ln.start)
    return chosen


def _norm(text: str) -> str:
    from .english import words

    return "".join(_HAN.findall(text) + _KANA.findall(text)) or " ".join(w for _, w in words(text))


def _windows(lines: list[Line], target: float = 25.0, max_len: float = 32.0):
    """Yield (start index, exclusive end index, score) for lyric windows.

    Score every whole-line window, including songs without repeated lyrics.
    Estimated singing coverage and duration dominate; repetition is a bounded
    bonus. Never bridge a gap longer than five seconds. These are LRC-derived
    estimates, not evidence that the audio actually contains vocals.
    """
    if not all(math.isfinite(x) and x > 0 for x in (target, max_len)):
        raise ValueError("target and max_len must be finite and positive")
    target = min(target, max_len)
    valid = [math.isfinite(ln.start) and math.isfinite(ln.end)
             and 0 <= ln.start < ln.end and bool(ln.chars) for ln in lines]
    norms = [_norm(ln.text) if ok else "" for ln, ok in zip(lines, valid)]
    counts = Counter(norm for norm in norms if norm)
    for i, ln in enumerate(lines):
        if not valid[i]:
            continue
        covered = repetition = longest_gap = 0.0
        right = ln.start
        for j in range(i, len(lines)):
            line = lines[j]
            if not valid[j] or (j > i and line.start < lines[j - 1].start):
                break
            gap = max(0.0, line.start - right)
            end = max(right, line.end)
            span = end - ln.start
            if gap > 5.0 or span > max_len:
                break
            # Union of intervals: overlapping lyrics must not inflate coverage.
            covered += max(0.0, line.end - max(right, line.start))
            right = end
            longest_gap = max(longest_gap, gap)
            repetition += min(max(counts[norms[j]] - 1, 0), 2) / 2
            score = (3.0 * covered / span
                     + 2.0 * min(covered / target, 1.0)
                     - abs(span - target) / target
                     - longest_gap / target
                     + 0.5 * repetition / (j - i + 1))
            yield i, j + 1, score


def pick_chorus(lines: list[Line], target: float = 25.0, max_len: float = 32.0) -> tuple[int, int]:
    """Highest-scoring continuous lyric window, retaining the existing selector."""
    i, j, _ = max(_windows(lines, target, max_len), key=lambda x: x[2], default=(0, 0, 0))
    return i, j


def pick_random(lines: list[Line], target: float = 25.0, max_len: float = 32.0,
                seed: int | None = None) -> tuple[int, int]:
    """Random start among good lyric windows, with a reproducible optional seed."""
    starts = {}
    for i, j, score in _windows(lines, target, max_len):
        if i not in starts or score > starts[i][2]:
            starts[i] = (i, j, score)
    if not starts:
        return 0, 0
    best = max(v[2] for v in starts.values())
    candidates = [v for v in starts.values() if v[2] >= best - .5]
    i, j, _ = random.Random(seed).choice(candidates)
    return i, j
