"""Turn phoneme alignment, transcribed notes and F0 into phrases."""

from pathlib import Path

import json
import shutil

import numpy as np

from . import audio, japanese, settings, tools
from .lrc import Line
from .score import Note, Phrase, Syllable
from .voicebank import Voicebank, hz_to_midi
from .errors import TaskError

MIN_NOTE = 0.06
CONSONANT_GAP = 0.25  # gaps shorter than this between syllables are consonants, not rests
PHRASE_GAP = 0.3  # a silence/breath at least this long starts a new phrase
MAX_PHRASE = 12.0  # seconds; longer runs are split at their widest internal gap
HEAD = 0.4  # max SP context kept around a phrase when rendering
HELD_MORA = 0.05  # a Japanese vowel mora aligned shorter than this was sung as its predecessor held
# (previous vowel, mora) pairs commonly sung as one long vowel: ソウ = ソー, セイ = セー.
_HELD_PAIRS = {("a", "a"), ("i", "i"), ("u", "u"), ("e", "e"), ("o", "o"), ("o", "u"), ("e", "i")}


def sofa_words() -> set[str]:
    return {ln.split("\t", 1)[0] for ln in settings.SOFA_DICT.read_text(encoding="utf8").splitlines() if ln}


def _in_clip(ln: Line, lo: float, hi: float) -> list[tuple[str, str, str]]:
    """Units of a context line whose (evenly spread) time estimate falls in [lo, hi].

    Only the tail of the previous line / head of the next one is actually in the
    clip; aligning a whole long line would cram it into a second of audio.
    """
    n = len(ln.units)
    step = (ln.end - ln.start) / max(n, 1)
    return [u for k, u in enumerate(ln.routed_units) if lo <= ln.start + (k + 0.5) * step <= hi]


# Explicit, position-preserving differences between the Mandarin dictionaries.
# The glide depends on the syllable: H is not a global alias for y or y0.
_ZH_PHONE_ALIASES = {
    *{(word, 0, "zh/y0", "zh/H") for word in ("yu", "yuan", "yue", "yun")},
    *{(word, 0, "zh/y", "zh/H") for word in ("yong", "ywu")},
    ("yo", 1, "zh/io", "zh/o"),
}


def _chinese_phone_mapping(unit: str, synthesis: list[str], dictionary: dict[str, list[str]]) -> dict:
    alignment = dictionary.get(unit)
    if not alignment:
        raise TaskError(f"Chinese word missing from HubertFA dictionary: {unit}",
                        "LYRIC_WORD_NOT_FOUND", word=unit, language="zh", source="aligner_dictionary")
    if len(alignment) != len(synthesis) or any(
        a != b and (unit, i, a, b) not in _ZH_PHONE_ALIASES
        for i, (a, b) in enumerate(zip(alignment, synthesis))
    ):
        raise TaskError(f"Unsupported Chinese phoneme mapping for {unit}: "
                        f"HubertFA {alignment} -> synthesis {synthesis}",
                        "PHONEME_UNSUPPORTED", word=unit, language="zh",
                        alignment=alignment, synthesis=synthesis)
    return {"alignment": alignment, "synthesis": synthesis}


def write_segments(lines: list[Line], lead: np.ndarray, seg_dir: Path, vb: Voicebank,
                   before: list[Line] = (), after: list[Line] = (), clip_start: float = 0.0,
                   _language: str | None = None, _chunked: bool = False) -> list[dict]:
    """The whole clip as one alignment input carrying every line's syllables.

    LRC timestamps are often off by most of a second, so cutting per line
    slices words in half; a single global forced alignment avoids trusting them.
    Only syllables both dictionaries know are kept. `before`/`after` lines are
    aligned too (their audio is in the clip) but tagged line -1 and dropped later.
    """
    if not _chunked and len(lead) > 60 * settings.SAMPLE_RATE and len(lines) > 1:
        languages = {lang for ln in [*before, *lines, *after] for _, _, lang in ln.routed_units}
        language = next(iter(languages)) if len(languages) == 1 else "mixed"
        segments, dictionaries = [], set()
        begin = 0
        while begin < len(lines):
            end = begin + 1
            while end < len(lines) and lines[end].end - lines[begin].start <= 40:
                end += 1
            lo = max(0, round((lines[begin].start - clip_start - 2) * settings.SAMPLE_RATE))
            hi = min(len(lead), round((lines[end - 1].end - clip_start + 2) * settings.SAMPLE_RATE))
            folder = seg_dir / f"chunk_{begin:04}" / "segments"
            items = write_segments(lines[begin:end], lead[lo:hi], folder, vb,
                before=lines[begin-1:begin] if begin else before,
                after=lines[end:end+1] if end < len(lines) else after,
                clip_start=clip_start + lo / settings.SAMPLE_RATE, _language=language, _chunked=True)
            if not items:
                raise ValueError("A requested lyric chunk has no singable units")
            for item in items:
                name = f"clip_{begin:04}"
                for suffix in (".wav", ".lab"):
                    shutil.copyfile(folder / (item["name"] + suffix), seg_dir / (name + suffix))
                item.update(name=name, offset=lo / settings.SAMPLE_RATE)
                dictionaries.update(Path(item["dictionary"]).read_text().splitlines())
                segments.append(item)
            begin = end
        dictionary = seg_dir.parent / "chunk_dictionary.txt"
        dictionary.write_text("\n".join(sorted(dictionaries)) + "\n")
        for item in segments:
            item["dictionary"] = str(dictionary)
        meta = {k: segments[0][k] for k in ("backend", "lang", "languages", "ckpt", "dictionary")}
        (seg_dir.parent / "align_meta.json").write_text(json.dumps(meta))
        (seg_dir.parent / "segments.json").write_text(json.dumps(segments, ensure_ascii=False, indent=2))
        return segments
    seg_dir.mkdir(parents=True, exist_ok=True)
    clip_end = clip_start + len(lead) / settings.SAMPLE_RATE
    tagged = ([(-1, _in_clip(ln, clip_start - 0.3, clip_end)) for ln in before]
              + [(k, ln.routed_units) for k, ln in enumerate(lines)]
              + [(-1, _in_clip(ln, clip_start, clip_end + 0.3)) for ln in after])
    languages = sorted({lang for _, units in tagged for _, _, lang in units})
    lang = _language or (languages[0] if len(languages) == 1 else "mixed")
    backend = tools.alignment_backend(lang)
    if lang == "zh":
        dictionary = settings.HUBERTFA_DICT if backend == "hubertfa" else settings.SOFA_DICT
        aligner_known = {ln.split("\t", 1)[0] for ln in dictionary.read_text(encoding="utf8").splitlines() if ln}
        known = lambda g, language: g in aligner_known and vb.knows(g, language)  # noqa: E731
    else:
        known = lambda g, language: vb.knows(g, language)  # noqa: E731
    chars, pinyin, line_of, word_units, word_langs = [], [], [], [], []
    for k, units in tagged:
        for c, py, unit_lang in units:
            if (k >= 0 or lang in ("en", "mixed") or (lang == "ja" and backend == "hubertfa")) and not known(py, unit_lang):
                language = {"en": "English", "ja": "Japanese", "zh": "Chinese"}[unit_lang]
                raise TaskError(f"{language} word missing from voicebank dictionary: {py}",
                                "LYRIC_WORD_NOT_FOUND", word=py, language=unit_lang,
                                text=c, line=lines[k].text if k >= 0 else None,
                                start=lines[k].start if k >= 0 else None,
                                source="lyrics" if k >= 0 else "alignment_context")
            if known(py, unit_lang):
                chars.append(c)
                pinyin.append(f"{unit_lang}:{py}" if lang == "mixed" else py)
                line_of.append(k)
                word_units.append(py)
                word_langs.append(unit_lang)
    if not pinyin:
        return []
    audio.save(seg_dir / "clip.wav", lead)
    (seg_dir / "clip.lab").write_text(" ".join(pinyin), encoding="utf8")
    phone_mappings = {}
    if backend == "hubertfa" and lang in ("en", "ja", "mixed"):
        ckpt, dictionary = settings.HUBERTFA_MODEL, seg_dir.parent / "hubertfa_dict.txt"
        vocab = json.loads((ckpt.parent / "vocab.json").read_text())["vocab"]
        pronunciations = {key: vb.g2p(u, language) for key, u, language in zip(pinyin, word_units, word_langs)}
        def align_phone(phone):
            return {"ja/cl": "cl", "en/#r": "en/_r"}.get(phone, phone)
        pronunciations = {key: [align_phone(p) for p in phones] for key, phones in pronunciations.items()}
        if lang == "mixed" and "zh" in word_langs:
            chinese = {}
            for line in settings.HUBERTFA_DICT.read_text(encoding="utf-8-sig").splitlines():
                if "\t" in line:
                    unit, phones = line.split("\t", 1)
                    chinese[unit.strip()] = [p if p.startswith("zh/") else "zh/" + p for p in phones.split()]
            for key, unit, language in zip(pinyin, word_units, word_langs):
                if language == "zh" and key not in phone_mappings:
                    mapping = _chinese_phone_mapping(unit, vb.g2p(unit, "zh"), chinese)
                    phone_mappings[key] = mapping
                    pronunciations[key] = mapping["alignment"]
        unknown = {p for phones in pronunciations.values() for p in phones if p not in vocab}
        if unknown:
            raise TaskError(f"Phonemes missing from HubertFA {lang} vocabulary: {sorted(unknown)}",
                            "PHONEME_UNSUPPORTED", phonemes=sorted(unknown))
        dictionary.write_text("".join(
            f"{key}\t{' '.join(p if lang == 'mixed' else p.removeprefix(lang + '/') for p in phones)}\n"
            for key, phones in sorted(pronunciations.items())), encoding="utf8")
    elif lang == "zh":
        ckpt = settings.HUBERTFA_MODEL if backend == "hubertfa" else settings.SOFA_CKPT
    else:
        # One entry per unit actually sung, one SOFA phoneme per voicebank phoneme,
        # so the alignment maps back onto the voicebank 1:1.
        ckpt, dictionary = settings.SOFA_CKPT_MULTI, seg_dir.parent / "sofa_dict.txt"
        dictionary.write_text("".join(
            f"{u}\t{' '.join(japanese.sofa_phoneme(p) for p in vb.g2p(u, lang))}\n" for u in sorted(set(pinyin))),
            encoding="utf8")
    meta = {"backend": backend, "lang": lang, "languages": languages,
            "ckpt": str(ckpt), "dictionary": str(dictionary)}
    (seg_dir.parent / "align_meta.json").write_text(json.dumps(meta))
    segments = [{"name": "clip", "offset": 0.0, "chars": chars, "pinyin": pinyin, "line_of": line_of,
             "word_units": word_units, "word_langs": word_langs, "phone_mappings": phone_mappings,
             "texts": [ln.text for ln in lines], **meta}]
    (seg_dir.parent / "segments.json").write_text(json.dumps(segments, ensure_ascii=False, indent=2))
    return segments


def _map_phones(vb: Voicebank, py: str, sofa: list[tuple[str, float, float]], lang: str = "zh",
                mapping: dict | None = None
                ) -> list[tuple[str, float, float]]:
    """Map aligned phonemes onto the voicebank without moving matched boundaries."""
    target = vb.g2p(py, lang)
    if mapping is not None:
        actual = [p if "/" in p else f"{lang}/{p}" for p, _, _ in sofa]
        if target != mapping["synthesis"] or actual != mapping["alignment"] or len(target) != len(sofa):
            raise TaskError(f"Aligned phonemes do not match the validated mapping for {py}: "
                            f"expected {mapping['alignment']}, got {actual}; synthesis {target}",
                            "PHONEME_UNSUPPORTED", word=py, language=lang,
                            alignment=actual, synthesis=target)
        return [(t, start, end) for t, (_, start, end) in zip(target, sofa)]
    if len(target) == len(sofa):
        return [(t, s, e) for t, (_, s, e) in zip(target, sofa)]
    s, e = sofa[0][1], sofa[-1][2]
    edges = np.linspace(s, e, len(target) + 1)
    return [(t, float(a), float(b)) for t, a, b in zip(target, edges[:-1], edges[1:])]


def _syllable_notes(start: float, end: float, notes: list[dict], f0_midi: np.ndarray) -> list[Note]:
    pieces = []
    for n in notes:
        if n["rest"]:
            continue
        a, b = max(start, n["start"]), min(end, n["start"] + n["dur"])
        if b - a > 1e-3:
            pieces.append([a, b, int(round(n["midi"]))])
    # Merge repeats, then fold slivers into their longer neighbour.
    merged: list[list] = []
    for p in pieces:
        if merged and merged[-1][2] == p[2] and p[0] - merged[-1][1] < 0.05:
            merged[-1][1] = p[1]
        else:
            merged.append(p)
    while len(merged) > 1:
        i = min(range(len(merged)), key=lambda i: merged[i][1] - merged[i][0])
        if merged[i][1] - merged[i][0] >= MIN_NOTE:
            break
        j = i - 1 if i == len(merged) - 1 or (i > 0 and merged[i - 1][1] - merged[i - 1][0] >= merged[i + 1][1] - merged[i + 1][0]) else i + 1
        lo, hi = min(i, j), max(i, j)
        keep = merged[j]
        merged[lo:hi + 1] = [[merged[lo][0], merged[hi][1], keep[2]]]
    if not merged:
        fps = settings.SAMPLE_RATE / settings.HOP
        seg = f0_midi[int(start * fps):max(int(end * fps), int(start * fps) + 1)]
        merged = [[start, end, int(round(float(np.median(seg))))]]
    merged[0][0], merged[-1][1] = start, end
    return [Note(a, b - a, m) for a, b, m in merged]


def _split(spans: list[tuple[float, float]], languages: list[str] | None = None) -> list[tuple[int, int]]:
    """Group syllable indices into phrases at breaths, capping phrase length."""
    groups, i = [], 0
    for j in range(1, len(spans) + 1):
        if (j == len(spans) or spans[j][0] - spans[j - 1][1] >= PHRASE_GAP
                or (languages is not None and languages[j] != languages[j - 1])):
            groups.append((i, j))
            i = j
    out = []
    while groups:
        a, b = groups.pop(0)
        if spans[b - 1][1] - spans[a][0] <= MAX_PHRASE or b - a < 2:
            out.append((a, b))
            continue
        # A continuous aligned run often has zero (or rounding-sized) gaps.
        # Repeatedly choosing its first boundary peels off single syllables.
        # Prefer substantial groups and balance tied gaps at 10 ms precision.
        candidates = [k for k in range(a + 1, b)
                      if spans[k - 1][1] - spans[a][0] >= PHRASE_GAP
                      and spans[b - 1][1] - spans[k][0] >= PHRASE_GAP]
        k = max(candidates or range(a + 1, b), key=lambda k: (
            round(max(0.0, spans[k][0] - spans[k - 1][1]), 2),
            -abs((spans[k - 1][1] - spans[a][0]) - (spans[b - 1][1] - spans[k][0]))))
        groups[:0] = [(a, k), (k, b)]
    return out


def _held_morae(units: list[str], langs: list[str], line_of: list[int],
                 spans: list[tuple[float, float]]) -> dict[int, int]:
    """{held mora: syllable it prolongs} for Japanese vowels the aligner left no time.

    The singer held the previous vowel instead, so the mora joins that note
    rather than becoming a sliver note (and a phoneme that was never sung).
    """
    held: dict[int, int] = {}
    for i in range(1, len(units)):
        j = held.get(i - 1, i - 1)
        prev = units[i - 1]
        if (langs[i] == langs[j] == "ja" and line_of[i] == line_of[j] and "_" not in prev
                and (prev[-1:], units[i]) in _HELD_PAIRS and spans[i][1] - spans[i][0] < HELD_MORA):
            held[i] = j
    return held


def build_phrases(vb: Voicebank, segs: list[dict], sofa: dict, notes: list[dict], f0: np.ndarray
                  ) -> tuple[list[Phrase], list[dict]]:
    f0_midi = hz_to_midi(f0)
    fps = settings.SAMPLE_RATE / settings.HOP
    phrases, report = [], []
    for seg in segs:
        res = sofa.get(seg["name"])
        if not res or "error" in res:
            report.append({"line": "(clip)", "error": (res or {}).get("error", "missing")})
            continue
        off = seg["offset"]
        words = [(w, s + off, e + off) for w, s, e in res["words"] if w not in ("SP", "AP")]
        if [w for w, _, _ in words] != seg["pinyin"]:
            report.append({"line": "(clip)", "error": "word sequence mismatch"})
            continue
        all_phones = [(p, s + off, e + off) for p, s, e in res["phones"]]

        units = seg.get("word_units", seg["pinyin"])
        langs = seg.get("word_langs", [seg["lang"]] * len(words))
        word_phones, spans = [], []  # per syllable: mapped phones; (vowel onset, end)
        for i, (_, ws, we) in enumerate(words):
            sofa_ph = [x for x in all_phones if x[0] not in ("SP", "AP") and x[1] >= ws - 1e-4 and x[2] <= we + 1e-4]
            mapped = _map_phones(vb, units[i], sofa_ph, langs[i],
                                 seg.get("phone_mappings", {}).get(seg["pinyin"][i]))
            word_phones.append(mapped)
            spans.append((next((s for p, s, _ in mapped if vb.is_vowel(p)), ws), we))
        word_spans = [(ws, we) for _, ws, we in words]
        chars, line_of = list(seg["chars"]), list(seg["line_of"])
        held = _held_morae(units, langs, line_of, word_spans)
        merged = {}
        for i, j in held.items():
            name, start, _ = word_phones[j][-1]
            word_phones[j] = word_phones[j][:-1] + [(name, start, word_spans[i][1])]
            word_spans[j] = (word_spans[j][0], word_spans[i][1])
            spans[j] = (spans[j][0], word_spans[i][1])
            chars[j] += chars[i]
            merged[j] = merged.get(j, 0) + 1
        if held:
            kept = [i for i in range(len(words)) if i not in held]
            merged = {kept.index(j): n for j, n in merged.items()}
            units, langs, chars, line_of, word_phones, spans, word_spans = (
                [xs[i] for i in kept] for xs in (units, langs, chars, line_of, word_phones, spans, word_spans))

        keep = [i for i, k in enumerate(line_of) if k >= 0]
        first, last = keep[0], keep[-1] + 1
        for a, b in _split(word_spans[first:last], langs[first:last]):
            a, b = a + first, b + first
            # Context: up to HEAD of silence each side, never past the neighbours.
            lo = max(word_spans[a][0] - HEAD, word_spans[a - 1][1] if a else 0.0)
            hi = min(word_spans[b - 1][1] + HEAD, word_spans[b][0] if b < len(word_spans) else res["length"] + off)
            phones = [(p, max(s, lo), min(e, hi)) for p, s, e in all_phones
                      if p in ("SP", "AP") and e > lo and s < hi]
            for k in range(a, b):
                phones.extend(word_phones[k])
            phones.sort(key=lambda x: x[1])
            if phones[0][1] > lo + 1e-3:
                phones.insert(0, ("SP", lo, phones[0][1]))
            if phones[-1][2] < hi - 1e-3:
                phones.append(("SP", phones[-1][2], hi))
            # Close tiny holes the aligner leaves so durations tile the phrase.
            phones = [(p, s, phones[i + 1][1] if i + 1 < len(phones) else e) for i, (p, s, e) in enumerate(phones)]

            syls = []
            for i in range(a, b):
                if langs[i] == "en":
                    from .english import phone_groups

                    mapped = word_phones[i]
                    groups = phone_groups([p for p, _, _ in mapped], vb.vowels)
                    for j, (ga, gb) in enumerate(groups):
                        chunk = mapped[ga:gb]
                        s = next((s for p, s, _ in chunk if vb.is_vowel(p)), chunk[0][1])
                        e = (next((s for p, s, _ in mapped[gb:] if vb.is_vowel(p)), mapped[gb][1])
                             if j + 1 < len(groups) else spans[i][1])
                        if j + 1 == len(groups) and i + 1 < b and 0 <= spans[i + 1][0] - e < CONSONANT_GAP:
                            e = spans[i + 1][0]
                        syls.append(Syllable(chars[i] if j == 0 else "·",
                                            f"{units[i]}~{j}",
                                            _syllable_notes(s, e, notes, f0_midi), "en"))
                    continue
                s, e = spans[i]
                if i + 1 < b and 0 <= spans[i + 1][0] - e < CONSONANT_GAP:
                    e = spans[i + 1][0]  # the next consonant still belongs to this note
                syls.append(Syllable(chars[i], units[i], _syllable_notes(s, e, notes, f0_midi), langs[i]))
            phrases.append(Phrase(syls, phones=phones, f0_midi=f0_midi[int(round(lo * fps)):]))
            lines = sorted({line_of[i] for i in range(a, b)})
            report.append({"line": (" " if langs[a] == "en" else "").join(chars[a:b]), "lrc_lines": [seg["texts"][k] for k in lines],
                           "lang": langs[a],
                           "start": round(word_spans[a][0], 2), "backend": seg.get("backend", "sofa"),
                           "confidence": round(res["confidence"], 3) if res.get("confidence") is not None else None,
                           "syllables": len(syls), "notes": sum(len(x.notes) for x in syls),
                           "held_morae": sum(merged.get(i, 0) for i in range(a, b))})
    return phrases, report
