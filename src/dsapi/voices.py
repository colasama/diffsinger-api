"""Public voice catalog. Paths and per-stage speaker names stay server-side."""
from dataclasses import dataclass
from pathlib import Path

import yaml

from . import settings

DEFAULT_VOICE = "imported-voice-e"


@dataclass(frozen=True)
class Voice:
    id: str
    name: str
    speaker: str | None
    aliases: tuple[str, ...]

    @property
    def root(self) -> Path:
        return settings.MULTI_VOICE if self.speaker else settings.VOICEBANK

    def validate(self) -> None:
        for sub in ("", "dsdur", "dspitch", "dsvariance"):
            root = self.root / sub
            cfg = yaml.safe_load((root / "dsconfig.yaml").read_text(encoding="utf-8-sig"))
            required = {"": ("acoustic",), "dsdur": ("linguistic", "dur"),
                        "dspitch": ("linguistic", "pitch"), "dsvariance": ("linguistic", "variance")}[sub]
            for key in ("phonemes", "languages", *required):
                if key not in cfg or not (root / cfg[key]).is_file():
                    raise FileNotFoundError(f"missing {sub}/{key}")
            if cfg.get("sample_rate", 44100) != settings.SAMPLE_RATE or cfg.get("hop_size", 512) != settings.HOP:
                raise ValueError("unsupported sample rate or hop size")
            if self.speaker:
                matches = [s for s in cfg.get("speakers", []) if s.rsplit(".", 1)[-1] == self.speaker]
                if len(matches) != 1 or (root / (matches[0] + ".emb")).stat().st_size != cfg["hidden_size"] * 4:
                    raise ValueError("missing or invalid speaker embedding")
        for lang in ("zh", "ja", "en"):
            if not (self.root / "dsdur" / f"dsdict-{lang}.yaml").is_file():
                raise FileNotFoundError(f"missing {lang} dictionary")
        voc = yaml.safe_load((settings.VOCODER / "vocoder.yaml").read_text())
        acoustic = yaml.safe_load((self.root / "dsconfig.yaml").read_text())
        if acoustic["vocoder"] != voc["name"] or not (settings.VOCODER / voc["model"]).is_file():
            raise ValueError("incompatible or missing vocoder")

    def snapshot(self) -> dict:
        return {"id": self.id, "name": self.name, "speaker": self.speaker,
                "model_version": "version1" if self.speaker else "imported-fixed-v1"}


VOICES = (
    Voice("imported-fixed", "用户固定音源", None, ("用户固定音源", "固定", "imported-fixed")),
    Voice("imported-voice-a", "用户音源甲", "SpeakerA", ("音源甲", "用户音源甲", "音源甲", "甲音源", "甲语", "voice-a")),
    Voice("imported-voice-b", "用户音源乙", "SpeakerB", ("音源乙", "乙音源", "乙语", "乙", "voice-b", "speakerb")),
    Voice("imported-voice-c", "用户音源丙", "SpeakerC", ("用户音源丙", "丙", "voice-c")),
    Voice("imported-voice-d", "用户音源丁", "SpeakerD", ("用户音源丁", "丁", "voice-d")),
    Voice(DEFAULT_VOICE, "用户音源戊", "SpeakerE", ("用户音源戊", "音源戊", "戊", "voice-e")),
    Voice("imported-voice-f", "用户音源己", "SpeakerF", ("用户音源己", "己", "voice-f")),
    Voice("imported-voice-g", "用户音源庚", "SpeakerG", ("用户音源庚", "庚", "voice-g")),
)


def get(voice_id: str | None = None) -> Voice:
    for voice in VOICES:
        if voice.id == (DEFAULT_VOICE if voice_id is None else voice_id):
            return voice
    raise ValueError(f"unknown voice_id: {voice_id}")


def catalog() -> dict:
    entries = []
    for voice in VOICES:
        try:
            voice.validate()
            available = True
        except (OSError, ValueError, KeyError, yaml.YAMLError):
            available = False
        entries.append({**voice.snapshot(), "aliases": list(voice.aliases), "available": available,
                        "languages": ["zh", "ja", "en"], "modes": ["score", "mimic"]})
    return {"default_voice_id": DEFAULT_VOICE, "voices": entries}
