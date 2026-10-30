"""Tempo/key estimation, grid quantization and numbered-notation (简谱) text."""

from dataclasses import dataclass, replace
from pathlib import Path

import librosa
import numpy as np

from .score import Note, Phrase, Syllable

# Krumhansl-Kessler key profiles.
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_NAMES = ["C", "#C", "D", "bE", "E", "F", "#F", "G", "bA", "A", "bB", "B"]
_DEGREE = ["1", "#1", "2", "b3", "3", "4", "#4", "5", "b6", "6", "b7", "7"]
SUBDIV = 4  # grid = 1/4 beat (sixteenth note in 4/4)


@dataclass
class Grid:
    bpm: float
    origin: float  # time of a beat, seconds

    @property
    def step(self) -> float:
        return 60.0 / self.bpm / SUBDIV

    def snap(self, t: float) -> float:
        return self.origin + round((t - self.origin) / self.step) * self.step


def _onset_error(onsets: np.ndarray, bpm: float) -> tuple[float, float]:
    """Best phase for this tempo and the mean |offset| in units of one grid step.

    Normalising by the step keeps finer grids from winning by default: random
    onsets average ~0.25 at any tempo, a true grid approaches 0.
    """
    step = 60.0 / bpm / SUBDIV
    best = (1.0, 0.0)
    for phase in np.linspace(0, step, 24, endpoint=False):
        frac = ((onsets - phase) / step) % 1.0
        err = float(np.mean(np.minimum(frac, 1 - frac)))
        if err < best[0]:
            best = (err, phase)
    return best


def estimate_grid(instrumental: Path, phrases: list[Phrase]) -> Grid:
    """Beat tracking proposes a tempo; note onsets pick the octave and refine it.

    librosa often lands on half/double or 2/3 tempo and its beat phase need not
    match where the singer actually places notes, so candidates around its
    estimate are scored by how well the transcribed onsets sit on the grid.
    """
    y, sr = librosa.load(instrumental, sr=22050, mono=True)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    base = float(np.atleast_1d(tempo)[0]) or 120.0
    onsets = np.array([n.start for p in phrases for s in p.syllables for n in s.notes])
    if len(onsets) < 8:
        bpm = base
        while bpm < 70:
            bpm *= 2
        while bpm > 180:
            bpm /= 2
        return Grid(round(bpm, 1), float(beats[0]) if len(beats) else 0.0)
    best = (np.inf, base, 0.0)
    for mult in (0.5, 2 / 3, 1.0, 1.5, 2.0):
        centre = base * mult
        if not 60 <= centre <= 200:
            continue
        for bpm in np.arange(centre * 0.97, centre * 1.03, 0.1):
            err, phase = _onset_error(onsets, bpm)
            # Slight preference for the tracker's own tempo when fits are close.
            err += 0.01 * abs(np.log2(mult))
            if err < best[0]:
                best = (err, float(bpm), float(phase))
    # Step-normalised error favours slower tempi (a coarse grid forgives jitter),
    # and half/double tempo is a notation choice anyway: move into the usual
    # 90-180 range when that costs little fit.
    err, bpm, phase = best
    for factor, outside in ((2.0, bpm < 90), (0.5, bpm > 180)):
        if outside:
            alt_err, alt_phase = _onset_error(onsets, bpm * factor)
            if alt_err - err < 0.04:
                bpm, phase = bpm * factor, alt_phase
    return Grid(round(bpm, 1), phase)


def estimate_tonic(phrases: list[Phrase]) -> int:
    """Pitch class of the major-key tonic (minor keys map to their relative major)."""
    hist = np.zeros(12)
    for ph in phrases:
        for s in ph.syllables:
            for n in s.notes:
                hist[n.midi % 12] += n.dur
    best, tonic = -np.inf, 0
    for k in range(12):
        for prof, shift in ((_MAJOR, 0), (_MINOR, 3)):
            r = np.corrcoef(hist, np.roll(prof, k))[0, 1]
            if r > best:
                best, tonic = r, (k + shift) % 12
    return tonic


def quantize(phrases: list[Phrase], grid: Grid) -> list[Phrase]:
    out = []
    for ph in phrases:
        syls = []
        for s in ph.syllables:
            notes = []
            for n in s.notes:
                a, b = grid.snap(n.start), grid.snap(n.end)
                if notes and a < notes[-1].end:
                    a = notes[-1].end
                if b - a >= grid.step - 1e-6:
                    notes.append(Note(a, b - a, n.midi))
            if not notes:  # shorter than one grid step: give it one step
                a = grid.snap(s.start)
                if syls and a < syls[-1].end:
                    a = syls[-1].end
                notes = [Note(a, grid.step, s.notes[0].midi)]
            if syls and notes[0].start < syls[-1].end:
                shift = syls[-1].end - notes[0].start
                notes = [Note(n.start + shift, n.dur, n.midi) for n in notes]
            # A one-step gap is timing jitter, not a rest: sing legato through it.
            if syls and 0 < notes[0].start - syls[-1].end <= grid.step + 1e-6:
                last = syls[-1].notes[-1]
                syls[-1].notes[-1] = Note(last.start, notes[0].start - last.start, last.midi)
            syls.append(Syllable(s.text, s.pinyin, notes, s.lang))
        out.append(replace(ph, syllables=syls))
    return out


def _dur_tokens(units: int) -> list[str]:
    """Split a length in grid units into 简谱 duration suffixes (first token carries the pitch)."""
    table = [(16, " - - -"), (12, " - -"), (8, " -"), (6, "."), (4, ""), (3, "_."), (2, "_"), (1, "=")]
    parts = []
    while units > 0:
        for u, suffix in table:
            if u <= units:
                parts.append(suffix)
                units -= u
                break
    return parts


def _pitch(midi: int, tonic: int, ref: int) -> str:
    rel = midi - ref
    octave, semi = divmod(rel, 12)
    return _DEGREE[semi] + ("'" * octave if octave > 0 else "," * -octave)


def _token(pitch: str, units: int) -> str:
    parts = _dur_tokens(units)
    return " ".join(f"{pitch}{p}" if i == 0 else f"⌒{pitch}{p}" for i, p in enumerate(parts))


def render(phrases: list[Phrase], grid: Grid, tonic: int) -> str:
    """One line of notes plus one line of lyrics per phrase.

    Notation: ' / , = octave up/down, _ = eighth, = = sixteenth, . = dotted,
    - = extra beat, ⌒ = tie or slur, 0 = rest.
    """
    all_midi = [n.midi for p in phrases for s in p.syllables for n in s.notes]
    # Put the tonic octave so most notes sit in the middle register.
    centre = int(np.median(all_midi)) if all_midi else 60
    ref = centre - ((centre - tonic) % 12)
    lines = [f"1={_NAMES[tonic]}  4/4  ♩={grid.bpm:g}", ""]
    for ph in phrases:
        notes, lyrics = [], []
        prev_end = None
        for s in ph.syllables:
            if prev_end is not None:
                gap = round((s.start - prev_end) / grid.step)
                if gap > 0:
                    notes.append(_token("0", gap))
                    lyrics.append("")
            for k, n in enumerate(s.notes):
                notes.append(_token(_pitch(n.midi, tonic, ref), max(1, round(n.dur / grid.step))))
                lyrics.append(s.text if k == 0 else "~")
            prev_end = s.end
        lines.append("  ".join(notes))
        lines.append("  ".join(lyrics))
        lines.append("")
    return "\n".join(lines)
