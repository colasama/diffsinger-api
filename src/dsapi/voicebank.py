"""ONNX inference for an OpenUtau-format DiffSinger voicebank.

Frame rate is sample_rate / hop_size (44100 / 512). All durations handed to
the models are integer frame counts that sum to the phrase length.
"""

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import yaml

from . import settings
from .onnx_backend import OnnxSession, create_session

LANGS = ("zh", "ja", "en")


def _session(path: Path) -> OnnxSession:
    return create_session(path)


class _Model:
    """One dsconfig.yaml directory: its phoneme table plus ONNX sessions."""

    def __init__(self, root: Path):
        self.root = root
        self.cfg = yaml.safe_load((root / "dsconfig.yaml").read_text(encoding="utf8"))
        self.phonemes: dict[str, int] = json.loads((root / self.cfg["phonemes"]).read_text())
        self.languages: dict[str, int] = json.loads((root / self.cfg["languages"]).read_text())
        self._sessions: dict[str, OnnxSession] = {}

    def session(self, key: str) -> OnnxSession:
        if key not in self._sessions:
            self._sessions[key] = _session(self.root / self.cfg[key])
        return self._sessions[key]

    def encode(self, phones: list[str]) -> tuple[np.ndarray, np.ndarray]:
        tokens = np.array([[self.phonemes[p] for p in phones]], dtype=np.int64)
        langs = np.array([[self.languages.get(p.split("/")[0], 0) if "/" in p else 0 for p in phones]], dtype=np.int64)
        return tokens, langs


def midi_to_hz(midi: np.ndarray) -> np.ndarray:
    return (440.0 * 2 ** ((midi - 69.0) / 12.0)).astype(np.float32)


def hz_to_midi(hz: np.ndarray) -> np.ndarray:
    return (69.0 + 12.0 * np.log2(np.maximum(hz, 1e-3) / 440.0)).astype(np.float32)


def frames_from_bounds(bounds_sec: list[float]) -> np.ndarray:
    """Cumulative rounding so per-item frame counts never drift from absolute time."""
    fps = settings.SAMPLE_RATE / settings.HOP
    edges = np.round(np.asarray(bounds_sec) * fps).astype(np.int64)
    return np.diff(edges)


@dataclass
class Notes:
    midi: np.ndarray  # float, per note; rests carry a neighbouring pitch
    rest: np.ndarray  # bool
    dur: np.ndarray  # int frames


class Voicebank:
    def __init__(self, root: Path = settings.VOICEBANK, vocoder: Path = settings.VOCODER):
        self.acoustic = _Model(root)
        self.dur = _Model(root / "dsdur")
        self.pitch = _Model(root / "dspitch")
        self.variance = _Model(root / "dsvariance")
        voc_cfg = yaml.safe_load((vocoder / "vocoder.yaml").read_text())
        self._vocoder_path = vocoder / voc_cfg["model"]
        self._vocoder: OnnxSession | None = None

    @cached_property
    def dictionaries(self) -> dict[str, dict[str, list[str]]]:
        return {lang: self._load_dict(lang)[0] for lang in LANGS}

    @cached_property
    def vowels(self) -> set[str]:
        return set().union(*(self._load_dict(lang)[1] for lang in LANGS)) | {"SP", "AP"}

    @property
    def dictionary(self) -> tuple[dict[str, list[str]], set[str]]:
        """Mandarin entries and all vowels (kept for callers that predate LANGS)."""
        return self.dictionaries["zh"], self.vowels

    def _load_dict(self, lang: str) -> tuple[dict[str, list[str]], set[str]]:
        """grapheme -> phonemes, and the vowel phonemes, for one language.

        The YAML is ~1M lines with every language; a line scan is much faster
        than a full parse and we only need that language's bare graphemes.
        """
        cache = self.__dict__.setdefault("_dict_cache", {})
        if lang in cache:
            return cache[lang]
        path = self.dur.root / f"dsdict-{lang}.yaml"
        entries: dict[str, list[str]] = {}
        vowels: set[str] = set()
        grapheme = symbol = None
        section = None
        for raw in path.read_text(encoding="utf8").splitlines():
            line = raw.strip()
            if raw.startswith("symbols:"):
                section = "symbols"
            elif raw.startswith("entries:"):
                section = "entries"
            elif section == "symbols":
                if line.startswith("- symbol:"):
                    symbol = line.split(":", 1)[1].strip()
                elif line.startswith("type:") and line.split(":", 1)[1].strip() == "vowel":
                    vowels.add(symbol)
            elif section == "entries":
                if line.startswith("- grapheme:"):
                    # YAML quotes graphemes that would otherwise parse as booleans ('no', 'on').
                    g = line.split(":", 1)[1].strip().strip("'\"")
                    grapheme = g if "/" not in g else None
                    if grapheme is not None:
                        entries[grapheme] = []
                elif line.startswith("- ") and grapheme is not None:
                    entries[grapheme].append(line[2:].strip())
        cache[lang] = (entries, vowels)
        return cache[lang]

    def g2p(self, grapheme: str, lang: str = "zh") -> list[str]:
        """Pinyin (zh) or a romaji mora unit such as "mo_cl" (ja) -> phonemes."""
        entries = self.dictionaries[lang]
        if lang == "en" and "~" in grapheme:
            from .english import phone_groups

            word, index = grapheme.rsplit("~", 1)
            phones = entries.get(word)
            if phones is None or not index.isdigit():
                raise KeyError(grapheme)
            groups = phone_groups(phones, self.vowels)
            if int(index) >= len(groups):
                raise KeyError(grapheme)
            a, b = groups[int(index)]
            return phones[a:b]
        if lang == "ja":
            from .japanese import unit_phonemes

            phones = unit_phonemes(grapheme, entries)
        else:
            phones = entries.get(grapheme)
        if phones is None:
            raise KeyError(f"'{grapheme}' not in voicebank {lang} dictionary")
        return phones

    def knows(self, grapheme: str, lang: str = "zh") -> bool:
        try:
            self.g2p(grapheme, lang)
            return True
        except KeyError:
            return False

    def is_vowel(self, phone: str) -> bool:
        return phone in self.vowels

    # ---- stages -------------------------------------------------------

    def predict_durations(self, phones: list[str], word_div: np.ndarray, word_dur: np.ndarray,
                          ph_midi: np.ndarray) -> np.ndarray:
        tokens, langs = self.dur.encode(phones)
        enc, mask = self.dur.session("linguistic").run(None, {
            "tokens": tokens, "languages": langs,
            "word_div": word_div[None].astype(np.int64), "word_dur": word_dur[None].astype(np.int64)})
        pred = self.dur.session("dur").run(None, {
            "encoder_out": enc, "x_masks": mask, "ph_midi": ph_midi[None].astype(np.int64)})[0][0]
        # Rescale inside each word so phonemes exactly fill the word's frames.
        out = np.zeros(len(phones), dtype=np.int64)
        start = 0
        for n, wd in zip(word_div, word_dur):
            seg = np.maximum(pred[start:start + n], 1e-3)
            cum = np.round(np.cumsum(seg) / seg.sum() * wd).astype(np.int64)
            out[start:start + n] = np.diff(cum, prepend=0)
            start += n
        return out

    def predict_pitch(self, phones: list[str], ph_dur: np.ndarray, notes: Notes, steps: int = 10) -> np.ndarray:
        tokens, langs = self.pitch.encode(phones)
        enc, _ = self.pitch.session("linguistic").run(None, {
            "tokens": tokens, "languages": langs, "ph_dur": ph_dur[None].astype(np.int64)})
        n_frames = int(ph_dur.sum())
        base = np.repeat(notes.midi, notes.dur).astype(np.float32)[:n_frames]
        return self.pitch.session("pitch").run(None, {
            "encoder_out": enc, "ph_dur": ph_dur[None].astype(np.int64),
            "note_midi": notes.midi[None].astype(np.float32), "note_rest": notes.rest[None],
            "note_dur": notes.dur[None].astype(np.int64), "pitch": base[None],
            "expr": np.ones((1, n_frames), np.float32), "retake": np.ones((1, n_frames), bool),
            "steps": np.array(steps, np.int64)})[0][0]

    def predict_variance(self, phones: list[str], ph_dur: np.ndarray, pitch_midi: np.ndarray,
                         steps: int = 20) -> tuple[np.ndarray, np.ndarray]:
        tokens, langs = self.variance.encode(phones)
        enc, _ = self.variance.session("linguistic").run(None, {
            "tokens": tokens, "languages": langs, "ph_dur": ph_dur[None].astype(np.int64)})
        n = len(pitch_midi)
        breath, voicing = self.variance.session("variance").run(None, {
            "encoder_out": enc, "ph_dur": ph_dur[None].astype(np.int64), "pitch": pitch_midi[None].astype(np.float32),
            "breathiness": np.zeros((1, n), np.float32), "voicing": np.zeros((1, n), np.float32),
            "retake": np.ones((1, n, 2), bool), "steps": np.array(steps, np.int64)})
        return breath[0], voicing[0]

    def render(self, phones: list[str], ph_dur: np.ndarray, pitch_midi: np.ndarray,
               steps: int = 20, gender: float = 0.0) -> np.ndarray:
        """Acoustic + vocoder for one phrase; pitch_midi has one value per frame."""
        n = int(ph_dur.sum())
        pitch_midi = np.asarray(pitch_midi, np.float32)[:n]
        breath, voicing = self.predict_variance(phones, ph_dur, pitch_midi)
        tokens, langs = self.acoustic.encode(phones)
        f0 = midi_to_hz(pitch_midi)
        mel = self.acoustic.session("acoustic").run(None, {
            "tokens": tokens, "languages": langs, "durations": ph_dur[None].astype(np.int64),
            "f0": f0[None], "breathiness": breath[None], "voicing": voicing[None],
            "gender": np.full((1, n), gender, np.float32), "velocity": np.ones((1, n), np.float32),
            "depth": np.array(self.acoustic.cfg.get("max_depth", 1.0), np.float32),
            "steps": np.array(steps, np.int64)})[0]
        if self._vocoder is None:
            self._vocoder = _session(self._vocoder_path)
        return self._vocoder.run(None, {"mel": mel, "f0": f0[None]})[0][0]
