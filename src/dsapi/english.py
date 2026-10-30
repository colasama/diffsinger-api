"""English word tokenization and vowel-centred singing units."""

import re

_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*")


def words(text: str) -> list[tuple[str, str]]:
    return [(w, w.lower()) for w in _WORD.findall(text.replace("’", "'"))]


def phone_groups(phones: list[str], vowels: set[str]) -> list[tuple[int, int]]:
    """Split multi-vowel words, assigning one intervening consonant to the onset.

    A deterministic approximation, not a pronunciation dictionary syllabifier.
    It preserves every phone and lets each vowel carry its own score notes.
    """
    nuclei = [i for i, p in enumerate(phones) if p in vowels]
    cuts = [0] + [max(a + 1, b - 1) for a, b in zip(nuclei, nuclei[1:])] + [len(phones)]
    return list(zip(cuts, cuts[1:]))
