"""LRC parsing and chorus window selection."""

import re
from collections import Counter
from dataclasses import dataclass, field

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
            stamped.append((int(m) * 60 + float(s), body))
    stamped.sort(key=lambda x: x[0])
    lines = []
    for i, (t, body) in enumerate(stamped):
        end = stamped[i + 1][0] if i + 1 < len(stamped) else song_end
        if body and not _META.match(body) and ":" not in body and "：" not in body:
            lines.append(Line(t, end, body))
    lang = detect_lang(lines)
    # An instrumental gap would otherwise be glued onto the line before it.
    per_unit = {"ja": 0.45, "en": 1.2}.get(lang, 0.6)
    for ln in lines:
        ln.lang = lang
        ln.end = min(ln.end, ln.start + max(2.0, per_unit * len(ln.chars) + 1.5))
    return [ln for ln in lines if ln.chars]


def _norm(text: str) -> str:
    from .english import words

    return "".join(_HAN.findall(text) + _KANA.findall(text)) or " ".join(w for _, w in words(text))


def pick_chorus(lines: list[Line], target: float = 25.0, max_len: float = 32.0) -> tuple[int, int]:
    """Return [i, j) of lines forming the most repeated contiguous block.

    Choruses repeat, so each line scores by how often its text appears in the
    song. The window starts on a repeated line, grows while it fits max_len,
    and the best mean repetition (ties: earliest) wins.
    """
    counts = Counter(_norm(ln.text) for ln in lines)
    best, best_score = (0, min(len(lines), 4)), -1.0
    for i, ln in enumerate(lines):
        if counts[_norm(ln.text)] < 2:
            continue
        j = i
        while j < len(lines) and lines[j].end - ln.start <= max_len:
            j += 1
        if j == i:
            continue
        # Prefer windows around target length, not just the longest.
        span = lines[j - 1].end - ln.start
        score = sum(counts[_norm(x.text)] for x in lines[i:j]) / (j - i)
        score -= abs(span - target) / target
        # Unsingable lines were dropped in parse(), leaving holes; avoid windows full of them.
        coverage = sum(x.end - x.start for x in lines[i:j]) / span
        score -= 1.5 * (1 - min(coverage, 1.0))
        if score > best_score:
            best, best_score = (i, j), score
    return best
