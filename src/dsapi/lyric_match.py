"""Conservative lyric anchors shared by song resolution and clip selection."""
import re
import unicodedata
from dataclasses import dataclass

from . import lrc


def tokens(text: str) -> tuple[str, ...]:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"(?<=[a-z])['’](?=[a-z])", "", text)
    # Latin words remain atomic; CJK characters can match across punctuation.
    return tuple(re.findall(r"[a-z0-9]+|[^\W_]", text))


@dataclass(frozen=True)
class Match:
    start: int
    stop: int  # exclusive; a cross-line match must fit in the selected window
    kind: str
    rank: int

    def metadata(self, lines: list[lrc.Line]) -> dict:
        return {"kind": self.kind, "text": " / ".join(ln.text for ln in lines[self.start:self.stop]),
                "start": lines[self.start].start, "end": lines[self.stop - 1].end}


def find_matches(lines: list[lrc.Line], query: str) -> list[Match]:
    needle = tokens(query)
    if not needle:
        return []
    # Short/common fragments may only match a complete line.
    short = len(needle) < 4 and sum(bool(re.fullmatch(r"[a-z0-9]+", t)) for t in needle) < 2
    texts = [tokens(ln.text) for ln in lines]
    matches = []
    for i, text in enumerate(texts):
        if not text:
            continue
        if text == needle:
            matches.append(Match(i, i + 1, "exact", 3))
            continue
        if short:
            continue
        joined = text
        # At most four adjacent lines; do not jump over empty lines or gaps.
        for j in range(i, min(i + 4, len(lines))):
            if j > i:
                if not texts[j] or not 0 <= lines[j].start - lines[j - 1].end <= 5:
                    break
                joined += texts[j]
            found = False
            for offset in range(len(text)):
                if joined[offset:offset + len(needle)] == needle:
                    matches.append(Match(i, j + 1, "prefix" if offset == 0 else "contains",
                                         2 if offset == 0 else 1))
                    found = True
                    break
            if found:
                break
    return matches


def pick_window(lines: list[lrc.Line], query: str, max_len: float = 32.) -> tuple[int, int, Match] | None:
    matches = {match.start: match for match in find_matches(lines, query)}
    best = None
    best_key = None
    for i, j, score in lrc._windows(lines, max_len=max_len):
        match = matches.get(i)
        if match is None or j < match.stop:
            continue
        key = (match.rank, score, -lines[i].start)
        if best_key is None or key > best_key:
            best, best_key = (i, j, match), key
    return best
