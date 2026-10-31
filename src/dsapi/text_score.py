"""Fixed alternating notation/lyric lines, converted to an absolute score."""

import re
import unicodedata
from dataclasses import asdict, dataclass

from . import english, japanese, lrc
from .score import HEAD, TAIL, Note, Phrase, Syllable

MAX_SECONDS = 1800
MAX_NOTES = 8192
_NOTE = re.compile(r"([#b]?)([0-7])([',]*)(?::(\d+(?:\.\d+)?))?")
_HAN = re.compile(r"[\u3400-\u9fff]")


class ScoreError(ValueError):
    def __init__(self, line: int, message: str, column: int = 1):
        self.detail = {"line": line, "column": column, "message": message}
        super().__init__(f"第 {line} 行：{message}")


@dataclass
class ParsedScore:
    phrases: list[Phrase]
    duration_seconds: float
    bpm: float
    key: str
    meter: str
    language: str
    rows: list[dict]

    def payload(self) -> dict:
        return {"phrases": [[asdict(s) for s in p.syllables] for p in self.phrases],
                "duration_seconds": self.duration_seconds, "bpm": self.bpm,
                "key": self.key, "meter": self.meter, "language": self.language,
                "rows": self.rows}


def _ignorable(c: str) -> bool:
    return c.isspace() or unicodedata.category(c).startswith("P")


def _units(text: str, lang: str, bank, line: int) -> list[tuple[str, str]]:
    if text.strip() == "-":
        return []
    if lang == "zh":
        chars, overrides = [], {}
        i = 0
        while i < len(text):
            c = text[i]
            if _HAN.fullmatch(c):
                chars.append(c)
                i += 1
                if i < len(text) and text[i] in "(（":
                    match = re.match(r"[（(]([a-zA-ZüÜvV:]+)[）)]", text[i:])
                    if not match:
                        raise ScoreError(line, "注音格式为 行(xing)，不带声调", i + 1)
                    overrides[len(chars)-1] = match[1].lower().replace("u:", "v").replace("ü", "v")
                    i += len(match[0])
            elif c not in "()（）" and _ignorable(c):
                i += 1
            else:
                raise ScoreError(line, f"无法识别歌词字符 {c!r}；日文或英文请设置 LANG=ja 或 LANG=en", i+1)
        try:
            readings = lrc.to_pinyin(chars)
        except (IndexError, ValueError):
            raise ScoreError(line, "部分汉字无法注音，请换字或填写注音") from None
        if len(readings) != len(chars):
            raise ScoreError(line, "部分汉字无法注音")
        units = [(c, overrides.get(i, p)) for i, (c, p) in enumerate(zip(chars, readings))]
    elif lang == "ja":
        for i, c in enumerate(text):
            if not (_HAN.fullmatch(c) or japanese.has_kana(c) or c == "々" or _ignorable(c)):
                raise ScoreError(line, f"无法识别日文字符 {c!r}", i+1)
        try:
            reading = japanese.reading(text, strict=True)
        except ValueError as exc:
            raise ScoreError(line, str(exc)) from exc
        units = japanese.morae(reading)
        if reading.startswith(("ッ", "ー")):
            raise ScoreError(line, "促音或长音不能独立放在开头")
        if "".join(display for display, _ in units) != reading:
            raise ScoreError(line, "歌词包含无法识别的假名，请使用普通假名")
    else:
        for i, c in enumerate(text):
            if not (c.isascii() and c.isalpha() or _ignorable(c)):
                raise ScoreError(line, f"无法识别英文字符 {c!r}", i+1)
        units = []
        for display, word in english.words(text):
            try:
                phones = bank.g2p(word, "en")
            except KeyError:
                raise ScoreError(line, f"音源词典不支持英文单词 {display}") from None
            if not any(p in bank.vowels for p in phones):
                raise ScoreError(line, f"单词 {display} 没有可演唱的元音")
            groups = english.phone_groups(phones, bank.vowels)
            units.extend((display if len(groups) == 1 else f"{display}[{i+1}]", f"{word}~{i}")
                         for i in range(len(groups)))
    for display, pronunciation in units:
        try:
            phones = bank.g2p(pronunciation, lang)
        except KeyError:
            raise ScoreError(line, f"音源不支持 {display} 的读音 {pronunciation}（{lang}）") from None
        if not phones or not any(bank.is_vowel(p) for p in phones):
            raise ScoreError(line, f"{display} 没有可演唱的元音")
    return units


def _bars(text: str, line: int) -> list[str]:
    parts = text.strip().strip("|").split("|")
    if any(not p.strip() for p in parts):
        raise ScoreError(line, "小节不能为空")
    return parts


def _events(text: str, tonic: int, line: int) -> list[list[tuple[int | None, float]]]:
    events = []
    pos = 0
    for token in re.finditer(r"\([^()]*\)|[^\s()]+", text):
        if text[pos:token.start()].strip():
            raise ScoreError(line, "连音括号不匹配或发生嵌套", pos+1)
        pos = token.end()
        slur = token[0].startswith("(")
        pieces = token[0][1:-1].split() if slur else [token[0]]
        if slur and len(pieces) < 2:
            raise ScoreError(line, "连音括号内至少填写两个音符", token.start()+1)
        group = []
        for piece in pieces:
            m = _NOTE.fullmatch(piece)
            if not m:
                raise ScoreError(line, f"无效音符 {piece!r}", token.start()+1)
            accidental, degree, octave, duration = m.groups()
            beats = float(duration or 1)
            if not .0625 <= beats <= 64:
                raise ScoreError(line, "音符时值应在 0.0625–64 拍之间", token.start()+1)
            if degree == "0":
                if slur or accidental or octave:
                    raise ScoreError(line, "休止符不能加升降号、八度或放进连音", token.start()+1)
                midi = None
            else:
                if "'" in octave and "," in octave:
                    raise ScoreError(line, "八度标记不能同时升高和降低", token.start()+1)
                midi = tonic + [0, 2, 4, 5, 7, 9, 11][int(degree)-1]
                midi += 12*(octave.count("'")-octave.count(",")) + {"": 0, "#": 1, "b": -1}[accidental]
                if not 0 <= midi <= 127:
                    raise ScoreError(line, "音高超出 MIDI 0–127 范围", token.start()+1)
            group.append((midi, beats))
        events.append(group)
    if text[pos:].strip() or not events:
        raise ScoreError(line, "乐谱为空或连音括号不匹配", pos+1)
    return events


def parse_score(text: str, bank) -> ParsedScore:
    if len(text) > 16384:
        raise ScoreError(1, "乐谱最多 16384 字符")
    lines = [(i+1, s.strip()) for i, s in enumerate(text.splitlines()) if s.strip()]
    bpm, tonic, key, meter, lang = 120., 60, "C4", "4/4", "zh"
    if lines and re.match(r"(?:1\s*=|BPM\s*=|LANG\s*=|\d+/\d+)", lines[0][1], re.I):
        line, header = lines.pop(0)
        seen = set()
        for token in header.split():
            if m := re.fullmatch(r"1=([A-Ga-g])([#b]?)(-?\d)", token):
                name = "key"
                tonic = (int(m[3])+1)*12 + {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[m[1].upper()]
                tonic += {"": 0, "#": 1, "b": -1}[m[2]]
                key = m[1].upper()+m[2]+m[3]
                if not 0 <= tonic <= 127:
                    raise ScoreError(line, "调号超出 MIDI 范围")
            elif m := re.fullmatch(r"BPM=(\d+(?:\.\d+)?)", token, re.I):
                name, bpm = "bpm", float(m[1])
                if not 20 <= bpm <= 400:
                    raise ScoreError(line, "BPM 应在 20–400 之间")
            elif m := re.fullmatch(r"(\d+)/(1|2|4|8|16)", token):
                name, meter = "meter", token
                if not 1 <= int(m[1]) <= 32:
                    raise ScoreError(line, "每小节拍数应在 1–32 之间")
            elif m := re.fullmatch(r"LANG=(zh|ja|en)", token, re.I):
                name, lang = "language", m[1].lower()
            else:
                raise ScoreError(line, f"无效参数 {token!r}；例如 1=C4 BPM=120 4/4 LANG=zh")
            if name in seen:
                raise ScoreError(line, f"参数 {name} 重复")
            seen.add(name)
    if not lines:
        raise ScoreError(1, "请填写一行简谱、一行歌词")
    if len(lines) % 2:
        raise ScoreError(lines[-1][0], "这行乐谱缺少下一行歌词")
    numerator, denominator = map(int, meter.split("/"))
    bar_beats = numerator*4/denominator
    seconds = 60/bpm
    cursor, count, syllables, rows = HEAD, 0, [], []
    for index in range(0, len(lines), 2):
        line, notation = lines[index]
        lyric_line, lyric = lines[index+1]
        event_bars = [_events(p, tonic, line) for p in _bars(notation, line)]
        if "|" in notation:
            for j, events in enumerate(event_bars):
                beats = sum(b for group in events for _, b in group)
                if abs(beats-bar_beats) > 1e-6:
                    raise ScoreError(line, f"第 {j+1} 小节为 {beats:g} 拍，{meter} 需要 {bar_beats:g} 拍")
        lyric_bars = [_units(p, lang, bank, lyric_line) for p in _bars(lyric, lyric_line)]
        expected = [sum(group[0][0] is not None for group in events) for events in event_bars]
        if "|" in lyric and (len(lyric_bars) != len(expected) or [len(p) for p in lyric_bars] != expected):
            raise ScoreError(lyric_line, f"歌词小节字数应为 {' / '.join(map(str, expected))}")
        units = [u for bar in lyric_bars for u in bar]
        if len(units) != sum(expected):
            preview = " ".join(u[0] for u in units)
            raise ScoreError(lyric_line, f"需要 {sum(expected)} 个演唱音节，实际 {len(units)} 个：{preview}")
        rows.append({"score_line": line, "lyric_line": lyric_line, "units": [
            {"text": d, "pinyin": p, "lang": lang} for d, p in units]})
        unit_index = 0
        for group in (g for bar in event_bars for g in bar):
            notes = []
            for midi, beats in group:
                count += 1
                dur = beats*seconds
                if midi is not None:
                    notes.append(Note(cursor, dur, midi))
                cursor += dur
                if cursor+TAIL > MAX_SECONDS or count > MAX_NOTES:
                    raise ScoreError(line, f"乐谱最多 {MAX_SECONDS//60} 分钟、{MAX_NOTES} 个音符（含休止符）")
            if notes:
                if sum(n.dur for n in notes) > 60:
                    raise ScoreError(line, "单个字及其连音最多持续 60 秒")
                display, pronunciation = units[unit_index]
                unit_index += 1
                syllables.append(Syllable(display, pronunciation, notes, lang))
    if not syllables:
        raise ScoreError(lines[0][0], "乐谱至少需要一个演唱音符")
    phrases = []
    for syl in syllables:
        if not phrases or syl.start-phrases[-1].syllables[-1].end >= .8 or syl.end-phrases[-1].syllables[0].start > 20:
            phrases.append(Phrase([]))
        phrases[-1].syllables.append(syl)
    return ParsedScore(phrases, cursor+TAIL, bpm, key, meter, lang, rows)
