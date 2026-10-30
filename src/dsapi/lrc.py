"""LRC parsing and chorus window selection."""

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from itertools import groupby

from pypinyin import Style, lazy_pinyin
from pypinyin.constants import PHRASES_DICT

# Words where 着 really is zháo when it starts them; pypinyin's dictionary also
# has rare readings like 着风 (catch a chill) that hijack 迎着风.
_ZHAO_HEAD = {"着急", "着迷", "着火", "着凉", "着陆", "着落", "着想", "着手", "着实", "着魔", "着慌", "着数"}

_TIME = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\]")
_META = re.compile(r"^\s*(作词|作曲|编曲|混音|母带|调校|调教|曲绘|PV|视频|和声|吉他|贝斯|鼓|弦乐|制作|出品|监制|策划|演唱|填词|原曲|A&R|Lyric|Music|Arrange)", re.I)
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

    @property
    def units(self) -> list[tuple[str, str]]:
        """[(display, grapheme)]: Han/pinyin, kana/romaji, or English words."""
        if self._units is None:
            self._units = self._compute_units()
        return self._units

    def _compute_units(self) -> list[tuple[str, str]]:
        han = _HAN.findall(self.text)
        if self.lang == "en":
            from .english import words

            return [] if han or _KANA.search(self.text) else words(self.text)
        if self.lang == "ja":
            from . import japanese

            kana = len(_KANA.findall(self.text))
            if len(_LATIN.findall(self.text)) > len(han) + kana:
                return []  # romaji/scat lines
            return japanese.morae(japanese.reading(self.text))
        # A line with kana is Japanese: its kanji are not Mandarin, so drop it whole.
        if _KANA.search(self.text) or len(_LATIN.findall(self.text)) > len(han):
            return []  # Japanese, or romaji/scat ("dala-da-da") dressed around a few kanji
        # pypinyin writes ü as "v", which matches both dictionaries (lv, nve, ...)
        return list(zip(han, to_pinyin(han)))

    @property
    def chars(self) -> list[str]:
        return [d for d, _ in self.units]

    @property
    def pinyin(self) -> list[str]:
        return [g for _, g in self.units]


def detect_lang(lines: list["Line"]) -> str:
    """Song-level: English majority, otherwise optionally Japanese or Mandarin.

    Japanese is experimental (alignment is not validated yet) and off unless
    DSAPI_EXPERIMENTAL_JA=1; otherwise kana lines are simply dropped as before.
    """
    import os

    latin = sum(bool(_LATIN.search(ln.text)) and not (_HAN.search(ln.text) or _KANA.search(ln.text))
                for ln in lines)
    if latin > len(lines) / 2:
        return "en"
    if os.environ.get("DSAPI_EXPERIMENTAL_JA", "") != "1":
        return "zh"
    cjk = [ln for ln in lines if _HAN.search(ln.text) or _KANA.search(ln.text)]
    if not cjk:
        return "zh"
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


def parse(text: str, song_end: float) -> list[Line]:
    stamped: list[tuple[float, str]] = []
    for raw in text.splitlines():
        times = _TIME.findall(raw)
        body = _TIME.sub("", raw).strip()
        for m, s in times:
            t = int(m) * 60 + float(s)
            if float(s) < 60 and 0 <= t < song_end:
                stamped.append((t, body))
    stamped.sort(key=lambda x: x[0])
    # Duplicate stamps (including bilingual LRC) share the next *distinct*
    # timestamp; otherwise the first copy gets a zero-length interval.
    groups = [(t, [body for _, body in group]) for t, group in groupby(stamped, key=lambda x: x[0])]
    lines = []
    for i, (t, bodies) in enumerate(groups):
        end = groups[i + 1][0] if i + 1 < len(groups) else song_end
        for body in dict.fromkeys(bodies):
            marker = body.strip(" \t[]【】()（）<>〈〉「」~～-—….")
            if (body and not _META.match(body) and ":" not in body and "：" not in body
                    and not _INSTRUMENTAL.fullmatch(marker)):
                lines.append(Line(t, end, body))
    lang = detect_lang(lines)
    # An instrumental gap would otherwise be glued onto the line before it.
    per_unit = {"ja": 0.45, "en": 1.2}.get(lang, 0.6)
    chosen = []
    seen = set()
    for ln in lines:
        ln.lang = lang
        ln.end = min(ln.end, ln.start + max(2.0, per_unit * len(ln.chars) + 1.5))
        if ln.chars and ln.end > ln.start and ln.start not in seen:
            chosen.append(ln)
            seen.add(ln.start)
    return chosen


def _norm(text: str) -> str:
    from .english import words

    return "".join(_HAN.findall(text) + _KANA.findall(text)) or " ".join(w for _, w in words(text))


def pick_chorus(lines: list[Line], target: float = 25.0, max_len: float = 32.0) -> tuple[int, int]:
    """Return [i, j) of a continuous, lyric-filled window (ties: earliest).

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
    best, best_score = (0, 0), -math.inf
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
            if score > best_score:
                best, best_score = (i, j + 1), score
    return best
