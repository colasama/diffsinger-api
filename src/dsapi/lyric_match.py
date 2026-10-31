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
    complete_lines: int = 0

    def metadata(self, lines: list[lrc.Line]) -> dict:
        return {"kind": self.kind, "text": " / ".join(ln.text for ln in lines[self.start:self.stop]),
                "start": lines[self.start].start, "end": lines[self.stop - 1].end,
                "line_count": self.stop - self.start}


def sentences(query: str) -> list[str]:
    return [part.strip() for part in re.split(r"[\r\n。！？!?；;]+|(?<!\d)\.(?!\d)", query)
            if tokens(part)]


def requires_full_match(query: str, extent: str = "auto") -> bool:
    return extent == "exact" or (extent == "auto" and len(sentences(query)) >= 3)


def find_matches(lines: list[lrc.Line], query: str, *, unrestricted: bool = False) -> list[Match]:
    """KMP on atomic lyric tokens; long queries cost O(song + query), not O(n²)."""
    needle = tokens(query)
    if not needle:
        return []
    short = len(needle) < 4 and sum(bool(re.fullmatch(r"[a-z0-9]+", t)) for t in needle) < 2
    texts = [tokens(ln.text) for ln in lines]
    flat, owners, starts, ends = [], [], [], []
    for i, text in enumerate(texts):
        starts.append(len(flat))
        flat.extend(text)
        owners.extend([i] * len(text))
        ends.append(len(flat))
    prefix = [0] * len(needle)
    j = 0
    for i in range(1, len(needle)):
        while j and needle[i] != needle[j]:
            j = prefix[j - 1]
        if needle[i] == needle[j]:
            j += 1
        prefix[i] = j
    matches, j = [], 0
    for pos, token in enumerate(flat):
        while j and token != needle[j]:
            j = prefix[j - 1]
        if token == needle[j]:
            j += 1
        if j != len(needle):
            continue
        begin, end = pos + 1 - j, pos + 1
        first, last = owners[begin], owners[pos]
        full = begin == starts[first] and end == ends[last]
        contiguous = all(texts[k] and lines[k].start >= lines[k - 1].start
                         and (unrestricted or 0 <= lines[k].start - lines[k - 1].end <= 5)
                         for k in range(first + 1, last + 1))
        if contiguous and (unrestricted or last - first < 4) and (not short or (full and first == last)):
            kind = "exact" if full else "prefix" if begin == starts[first] else "contains"
            complete = sum(begin <= starts[k] and ends[k] <= end for k in range(first, last + 1))
            matches.append(Match(first, last + 1, kind, {"exact": 3, "prefix": 2, "contains": 1}[kind], complete))
        j = prefix[j - 1]
    return matches


def full_match(lines: list[lrc.Line], query: str, extent: str = "auto") -> Match | None:
    if extent == "window":
        return None
    matches = find_matches(lines, query, unrestricted=True)
    def explicit_sentences(m: Match) -> bool:
        supplied = [tokens(t) for t in sentences(query)]
        original = [tokens(t) for t in sentences("\n".join(ln.text for ln in lines[m.start:m.stop]))]
        return (len(supplied) >= 3 and len(supplied) == len(original)
                and original[0][-len(supplied[0]):] == supplied[0]
                and original[-1][:len(supplied[-1])] == supplied[-1]
                and supplied[1:-1] == original[1:-1])
    eligible = [m for m in matches if extent == "exact" or m.complete_lines >= 3 or explicit_sentences(m)]
    return max(eligible, key=lambda m: (m.rank, -lines[m.start].start), default=None)


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
