"""Japanese lyrics -> sung morae -> voicebank romaji and SOFA phonemes.

One mora is one sung unit (like one Han character in Mandarin). っ joins the
mora before it ("ka_cl"), ー repeats the previous vowel, ん is its own mora "n".
"""

import re
from functools import lru_cache

_KANA_RE = re.compile(r"[ぁ-ゖァ-ヺー]")
_SMALL = set("ャュョァィゥェォヮ")
# Sung readings that differ from the dictionary's orthographic kana.
_PARTICLES = {"ハ": "ワ", "ヘ": "エ", "ヲ": "オ"}
_OVERRIDES = {"私": "ワタシ"}  # unidic prefers the formal ワタクシ; lyrics almost never mean it

_BASE = {
    "ア": "a", "イ": "i", "ウ": "u", "エ": "e", "オ": "o",
    "カ": "ka", "キ": "ki", "ク": "ku", "ケ": "ke", "コ": "ko",
    "ガ": "ga", "ギ": "gi", "グ": "gu", "ゲ": "ge", "ゴ": "go",
    "サ": "sa", "シ": "shi", "ス": "su", "セ": "se", "ソ": "so",
    "ザ": "za", "ジ": "ji", "ズ": "zu", "ゼ": "ze", "ゾ": "zo",
    "タ": "ta", "チ": "chi", "ツ": "tsu", "テ": "te", "ト": "to",
    "ダ": "da", "ヂ": "ji", "ヅ": "zu", "デ": "de", "ド": "do",
    "ナ": "na", "ニ": "ni", "ヌ": "nu", "ネ": "ne", "ノ": "no",
    "ハ": "ha", "ヒ": "hi", "フ": "fu", "ヘ": "he", "ホ": "ho",
    "バ": "ba", "ビ": "bi", "ブ": "bu", "ベ": "be", "ボ": "bo",
    "パ": "pa", "ピ": "pi", "プ": "pu", "ペ": "pe", "ポ": "po",
    "マ": "ma", "ミ": "mi", "ム": "mu", "メ": "me", "モ": "mo",
    "ヤ": "ya", "ユ": "yu", "ヨ": "yo",
    "ラ": "ra", "リ": "ri", "ル": "ru", "レ": "re", "ロ": "ro",
    "ワ": "wa", "ヰ": "i", "ヱ": "e", "ヲ": "o", "ン": "n", "ヴ": "vu",  # The phoneme mapping represents ん as "n"
    # small vowels standing alone are sung as plain vowels
    "ァ": "a", "ィ": "i", "ゥ": "u", "ェ": "e", "ォ": "o", "ャ": "ya", "ュ": "yu", "ョ": "yo", "ヮ": "wa",
}
_YOON = {"ャ": "a", "ュ": "u", "ョ": "o"}
_SPECIAL = {
    "シャ": "sha", "シュ": "shu", "ショ": "sho", "シェ": "she",
    "ジャ": "ja", "ジュ": "ju", "ジョ": "jo", "ジェ": "je",
    "チャ": "cha", "チュ": "chu", "チョ": "cho", "チェ": "che",
    "ヂャ": "ja", "ヂュ": "ju", "ヂョ": "jo",
    "ティ": "ti", "ディ": "di", "トゥ": "tu", "ドゥ": "du", "テュ": "tyu", "デュ": "dyu",
    "ファ": "fa", "フィ": "fi", "フェ": "fe", "フォ": "fo", "フュ": "fyu",
    "ウィ": "wi", "ウェ": "we", "ウォ": "wo", "ヴァ": "va", "ヴィ": "vi", "ヴェ": "ve", "ヴォ": "vo",
    "ツァ": "tsa", "ツィ": "tsi", "ツェ": "tse", "ツォ": "tso", "イェ": "ye",
    "クァ": "kwa", "グァ": "gwa", "クヮ": "kwa", "グヮ": "gwa",
}
_VOWEL = re.compile(r"[aiueo]$")

# Voicebank phoneme -> SOFA multilingual (opencpop-style) phoneme, for alignment only.
_SOFA = {"a": "a", "i": "i", "u": "u", "e": "E", "o": "o", "N": "n", "cl": "cl",
         "sh": "x", "ch": "q", "ts": "c", "j": "j", "r": "l", "v": "w", "kw": "k", "gw": "g", "dy": "d", "ty": "t",
         "AP": "AP", "SP": "SP"}


def has_kana(text: str) -> bool:
    return bool(_KANA_RE.search(text))


@lru_cache(maxsize=1)
def _tagger():
    import fugashi

    return fugashi.Tagger()


def reading(text: str, *, strict: bool = False, spoken: bool = False) -> str:
    """Katakana reading of a lyric line; non-Japanese characters are dropped.

    spoken=True uses unidic's pronunciation (ソウ -> ソー) where it has one.
    """
    out = []
    for w in _tagger()(text):
        surface = w.surface
        if surface in _OVERRIDES:
            out.append(_OVERRIDES[surface])
            continue
        kana = getattr(w.feature, "kana", None)
        pron = getattr(w.feature, "pron", None) if spoken else None
        if pron and pron != "*" and kana and kana != "*":
            kana = pron
        if not kana or kana == "*":
            if strict and not has_kana(surface) and re.search(r"[\u3400-\u9fff々]", surface):
                raise ValueError(f"无法识别 {surface} 的日语读音，请改填假名")
            kana = surface if has_kana(surface) else ""
        if w.feature.pos1 == "助詞" and kana in _PARTICLES:
            kana = _PARTICLES[kana]
        out.append(kana)
    kata = "".join(out).translate({c: c + 0x60 for c in range(0x3041, 0x3097)})
    return "".join(_KANA_RE.findall(kata))


def morae(kata: str) -> list[tuple[str, str]]:
    """[(display kana, romaji unit)], e.g. "モット" -> [("モッ", "mo_cl"), ("ト", "to")]."""
    units: list[list[str]] = []  # [display, romaji]
    i = 0
    while i < len(kata):
        c = kata[i]
        pair = kata[i:i + 2]
        if c == "ッ":
            if units:
                units[-1][0] += c
                units[-1][1] += "_cl"
            i += 1
            continue
        if c == "ー":
            if units:
                prev = units[-1][1].split("_")[0]
                m = _VOWEL.search(prev)
                units.append([c, m.group(0) if m else prev])
            i += 1
            continue
        if pair in _SPECIAL:
            units.append([pair, _SPECIAL[pair]])
            i += 2
            continue
        if len(pair) == 2 and pair[1] in _YOON and c in _BASE and c not in "ャュョ":
            onset = _BASE[c][:-1]  # キ -> k, ニ -> n
            units.append([pair, onset + "y" + _YOON[pair[1]]])
            i += 2
            continue
        if c in _BASE:
            units.append([c, _BASE[c]])
        i += 1
    return [(d, r) for d, r in units]


def lyric_morae(text: str, *, strict: bool = False) -> list[tuple[str, str]]:
    """Morae of a lyric line, displayed as written but sung as pronounced.

    An orthographic ウ that unidic pronounces as a long vowel (ソウ = ソー,
    トウキョウ = トーキョー) is sung as the held vowel, not a separate "u".
    Other spellings (エイ, verb-final ウ as in オモウ) keep their written mora.
    """
    written = morae(reading(text, strict=strict))
    spoken = morae(reading(text, spoken=True))
    if len(spoken) != len(written):
        return written
    return [(d, s if d == "ウ" and sd == "ー" else r) for (d, r), (sd, s) in zip(written, spoken)]


def unit_phonemes(unit: str, dictionary: dict[str, list[str]]) -> list[str] | None:
    """Voicebank phonemes for a romaji unit ("mo_cl" -> mo's phonemes + ja/cl)."""
    phones: list[str] = []
    for part in unit.split("_"):
        if part not in dictionary:
            return None
        phones.extend(dictionary[part])
    return phones


def sofa_phoneme(vb_phone: str) -> str:
    p = vb_phone.split("/", 1)[-1]
    if p in _SOFA:
        return _SOFA[p]
    # Palatalised onsets (ky, ny, ...) align as their plain consonant.
    return p[:-1] if len(p) == 2 and p.endswith("y") else p
