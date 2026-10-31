"""Public voice catalog. Paths and per-stage speaker names stay server-side."""
import json
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from . import settings


@dataclass(frozen=True)
class Voice:
    id: str
    name: str
    speaker: str | None
    aliases: tuple[str, ...]
    path: Path | None = None
    model_version: str | None = None

    @property
    def root(self) -> Path:
        return self.path if self.path is not None else settings.VOICEBANK

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
                "model_version": self.model_version}


def load_catalog(path: Path | None) -> tuple[str, tuple[Voice, ...]]:
    """Read operator-supplied metadata; no named models are bundled."""
    if path is None:
        return "default", (Voice("default", "用户导入音源", None, ()),)
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    entries = []
    seen = set()
    for item in data["voices"]:
        voice_id = item["id"]
        if not isinstance(voice_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", voice_id) or voice_id in seen:
            raise ValueError("voice IDs must be unique letters, numbers, underscores or hyphens")
        seen.add(voice_id)
        name, root = item["name"], item["path"]
        aliases = item.get("aliases", [])
        speaker, version = item.get("speaker"), item.get("model_version")
        if not isinstance(name, str) or not name.strip() or not isinstance(root, str) or not root.strip():
            raise ValueError("each voice requires a name and local path")
        if not isinstance(aliases, list) or any(not isinstance(a, str) or not a.strip() for a in aliases):
            raise ValueError("aliases must be a list of nonempty strings")
        if any(v is not None and (not isinstance(v, str) or not v.strip()) for v in (speaker, version)):
            raise ValueError("speaker and model_version must be nonempty strings when provided")
        root = Path(root).expanduser()
        if not root.is_absolute():
            root = path.parent / root
        entries.append(Voice(voice_id, name, speaker, tuple(aliases), root.resolve(), version))
    default = data["default_voice_id"]
    if default not in seen:
        raise ValueError("default_voice_id must reference an imported voice")
    return default, tuple(entries)


DEFAULT_VOICE, VOICES = load_catalog(settings.VOICES_FILE)
LEGACY_VOICE = {"id": "unknown", "name": "未记录音源", "speaker": None, "model_version": None}


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
