"""Score data model and its conversion into DiffSinger phrase inputs.

A Phrase is a run of Syllables on an absolute timeline (seconds from clip
start). DiffSinger's "word" begins at a vowel onset, so each syllable's leading
consonants are borrowed from the end of the previous word (or the head SP).
"""

from dataclasses import dataclass, field

import numpy as np

from .voicebank import Notes, Voicebank, frames_from_bounds, hz_to_midi
from . import scheduling

HEAD = 0.5  # seconds of SP before the first note, room for its consonant
TAIL = 0.3


@dataclass
class Note:
    start: float
    dur: float
    midi: int

    @property
    def end(self) -> float:
        return self.start + self.dur


@dataclass
class Syllable:
    text: str
    pinyin: str  # zh pinyin, ja romaji, or en word~syllable-index
    notes: list[Note]  # >1 note means a slur (melisma)
    lang: str = "zh"

    @property
    def start(self) -> float:
        return self.notes[0].start

    @property
    def end(self) -> float:
        return self.notes[-1].end


@dataclass
class Phrase:
    syllables: list[Syllable]
    # Mimic mode: real phoneme timing and pitch taken from the source vocal.
    phones: list[tuple[str, float, float]] = field(default_factory=list)
    f0_midi: np.ndarray | None = None  # frames from `start`

    @property
    def start(self) -> float:
        return self.syllables[0].start - HEAD

    @property
    def end(self) -> float:
        return self.syllables[-1].end + TAIL


def _items(phrase: Phrase) -> list[tuple[str, float, float, Syllable | None]]:
    """Timeline of SP and syllable spans covering [phrase.start, phrase.end]."""
    items = []
    t = phrase.start
    for syl in phrase.syllables:
        if syl.start - t > 1e-3:
            items.append(("SP", t, syl.start, None))
        items.append(("syl", syl.start, syl.end, syl))
        t = syl.end
    items.append(("SP", t, phrase.end, None))
    return items


def score_inputs(vb: Voicebank, phrase: Phrase):
    """phones, word_div, word_dur, ph_midi and Notes for the dur/pitch models."""
    items = _items(phrase)
    words: list[list[str]] = []
    word_bounds = [items[0][1]]
    ph_midi: list[int] = []
    word_midi: list[list[int]] = []
    note_bounds = [items[0][1]]
    note_midi: list[float] = []
    note_rest: list[bool] = []

    for kind, start, end, syl in items:
        if kind == "SP":
            phs, midi = ["SP"], None
        else:
            phs, midi = vb.g2p(syl.pinyin, syl.lang), syl.notes[0].midi
        k = 0
        while k < len(phs) and not vb.is_vowel(phs[k]):
            k += 1
        if k and words:
            words[-1].extend(phs[:k])
            word_midi[-1].extend([midi] * k)
        words.append(phs[k:])
        word_midi.append([midi] * (len(phs) - k))
        word_bounds.append(end)
        if syl is None:
            note_bounds.append(end)
            note_midi.append(np.nan)
            note_rest.append(True)
        else:
            for n in syl.notes:
                if n.start - note_bounds[-1] > 1e-3:
                    note_bounds.append(n.start)
                    note_midi.append(np.nan)
                    note_rest.append(True)
                note_bounds.append(n.end)
                note_midi.append(n.midi)
                note_rest.append(False)

    # Rests take the nearest sung pitch so the pitch model has a sane baseline.
    nm = np.array(note_midi, dtype=np.float32)
    sung = np.where(~np.isnan(nm))[0]
    for i in np.where(np.isnan(nm))[0]:
        nm[i] = nm[sung[np.argmin(np.abs(sung - i))]]
    fill = {i: int(nm[i]) for i in range(len(nm))}
    phones, word_div = [], []
    for w, mids in zip(words, word_midi):
        phones.extend(w)
        word_div.append(len(w))
        ph_midi.extend(m if m is not None else fill[0] for m in mids)

    notes = Notes(midi=nm, rest=np.array(note_rest), dur=frames_from_bounds(note_bounds))
    return phones, np.array(word_div), frames_from_bounds(word_bounds), np.array(ph_midi), notes


@scheduling.stage("synthesis", gpu=True)
def render_score(vb: Voicebank, phrase: Phrase) -> np.ndarray:
    phones, word_div, word_dur, ph_midi, notes = score_inputs(vb, phrase)
    ph_dur = vb.predict_durations(phones, word_div, word_dur, ph_midi)
    pitch = vb.predict_pitch(phones, ph_dur, notes)
    return vb.render(phones, ph_dur, pitch)


@scheduling.stage("synthesis", gpu=True)
def render_mimic(vb: Voicebank, phrase: Phrase) -> np.ndarray:
    """Use aligned phonemes and the source f0 directly; no dur/pitch prediction."""
    names = [p for p, _, _ in phrase.phones]
    bounds = [phrase.phones[0][1]] + [e for _, _, e in phrase.phones]
    ph_dur = frames_from_bounds(bounds)
    n = int(ph_dur.sum())
    pitch = phrase.f0_midi[:n]
    if len(pitch) < n:
        pitch = np.pad(pitch, (0, n - len(pitch)), mode="edge")
    return vb.render(names, ph_dur, pitch)


def f0_to_midi(f0_hz: np.ndarray) -> np.ndarray:
    return hz_to_midi(f0_hz)
