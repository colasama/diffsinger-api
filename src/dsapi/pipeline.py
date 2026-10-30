"""End-to-end job: song -> lyric-selected clip -> lead vocal -> score -> re-sung audio.

Every stage writes into work/<task_id>/ and records its timing in job.json so
a failed run shows exactly where it stopped.
"""

import json
import shutil
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import audio, extract, jianpu, lrc, selection, settings, sources, tools
from .score import Phrase, render_mimic, render_score
from .voicebank import Voicebank

MAX_CLIP = 40.0


@dataclass
class JobSpec:
    song_id: str | None = None
    audio_path: str | None = None  # local file instead of NetEase
    lrc_path: str | None = None
    start: float | None = None  # clip window override (seconds in the song)
    end: float | None = None
    modes: list[str] = field(default_factory=lambda: ["mimic", "score"])
    selection: str = "random"
    seed: int | None = None
    lyric_query: str | None = None


@dataclass
class JobState:
    task_id: str
    spec: JobSpec
    status: str = "queued"  # queued | running | done | failed
    stage: str = ""
    timings: dict[str, float] = field(default_factory=dict)
    started_at: float | None = None
    finished_at: float | None = None
    song: dict | None = None
    window: list[float] | None = None
    selection: dict | None = None
    lines: list[dict] = field(default_factory=list)
    alignment: list[dict] = field(default_factory=list)
    note_extraction: dict | None = None
    grid: dict | None = None
    tonic: str | None = None
    jianpu: str | None = None
    outputs: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    owner_id: str | None = None


_vb: Voicebank | None = None


def voicebank() -> Voicebank:
    global _vb
    if _vb is None:
        _vb = Voicebank()
    return _vb


class Job:
    def __init__(self, task_id: str, spec: JobSpec, *, owner_id: str | None = None):
        self.dir = settings.WORK / task_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state = JobState(task_id, spec, owner_id=owner_id)
        self.save()

    def save(self) -> None:
        # Readers must never see a partially written state/ownership record.
        path = self.dir / "job.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(asdict(self.state), ensure_ascii=False, indent=2))
        temporary.replace(path)

    def stage(self, name: str):
        job = self

        class _Stage:
            def __enter__(self):
                job.state.stage = name
                job.save()
                self.t = time.perf_counter()

            def __exit__(self, *exc):
                job.state.timings[name] = round(time.perf_counter() - self.t, 2)
                job.save()

        return _Stage()

    def run(self) -> JobState:
        st = self.state
        st.status = "running"
        st.started_at = time.time()
        self.save()
        try:
            self._run()
            st.status, st.stage = "done", ""
        except Exception as e:  # noqa: BLE001 - surfaced to the API caller
            st.status, st.error = "failed", f"{type(e).__name__}: {e}"
            (self.dir / "error.txt").write_text(traceback.format_exc())
        st.finished_at = time.time()
        self.save()
        return st

    def _run(self) -> None:
        st, spec, d = self.state, self.state.spec, self.dir
        vb = voicebank()

        with self.stage("fetch"):
            if spec.audio_path:
                src = d / ("source" + Path(spec.audio_path).suffix)
                shutil.copy(spec.audio_path, src)
                lrc_text = Path(spec.lrc_path).read_text(encoding="utf8")
                st.song = {"name": Path(spec.audio_path).stem}
            else:
                source = sources.get_source()
                st.song = asdict(source.song_info(spec.song_id))
                src = source.download_audio(spec.song_id, d / "source.audio")
                lrc_text = source.lyric(spec.song_id)
            # Not every source reports duration; the file is the ground truth.
            duration = audio.duration(src)
            (d / "lyrics.lrc").write_text(lrc_text, encoding="utf8")

        with self.stage("select"):
            lines = lrc.parse(lrc_text, duration)
            selected = selection.select(lines, duration, start=spec.start, end=spec.end,
                                        strategy=spec.selection, seed=spec.seed, max_clip=MAX_CLIP,
                                        lyric_query=spec.lyric_query)
            chosen, before, after = selected.lines, selected.before, selected.after
            start, end = selected.start, selected.end
            st.selection = selected.metadata()
            if selected.strategy == "random":
                spec.seed = selected.seed
            st.window = [round(start, 2), round(end, 2)]
            st.lines = [{"start": ln.start, "end": ln.end, "text": ln.text,
                         "languages": sorted({lang for _, _, lang in ln.routed_units})} for ln in chosen]
            clip = audio.cut(src, d / "clip.wav", start, end)

        with self.stage("separate"):
            lead_path, inst_path = audio.separate_vocals(clip, d / "sep")
            lead = audio.load(lead_path)

        with self.stage("align"):
            segs = extract.write_segments(chosen, lead, d / "segments", vb, before=before, after=after,
                                          clip_start=start)
            if not segs:
                raise RuntimeError("no lyric units the voicebank can sing in this window")
            sofa = tools.align(d / "segments", d, Path(segs[0]["ckpt"]), Path(segs[0]["dictionary"]),
                               backend=segs[0]["backend"], lang=segs[0]["lang"])

        with self.stage("notes_f0"):
            backend = tools.note_backend()
            if backend == "game-medium":
                f0, _ = tools.extract_f0(lead_path, d)
                # Alignment and F0 alone establish phrase/vowel boundaries;
                # GAME predicts notes conditioned on them, without loading SOME.
                regions, _ = extract.build_phrases(vb, segs, sofa, [], f0)
                if not regions:
                    raise RuntimeError("alignment produced no usable phrases")
                notes = tools.extract_game_notes(lead_path, d, regions, segs[0]["lang"])
            else:
                notes, f0, _ = tools.extract_some_notes_f0(lead_path, d)
            meta = json.loads((d / "notes.json").read_text())
            st.note_extraction = {k: v for k, v in meta.items() if k != "notes"}
            st.note_extraction.setdefault("backend", backend)

        with self.stage("build"):
            phrases, report = extract.build_phrases(vb, segs, sofa, notes, f0)
            st.alignment = report
            if not phrases:
                raise RuntimeError("alignment produced no usable phrases")

        with self.stage("quantize"):
            grid = jianpu.estimate_grid(inst_path, phrases)
            tonic = jianpu.estimate_tonic(phrases)
            q = jianpu.quantize(phrases, grid)
            st.grid, st.tonic = asdict(grid), jianpu._NAMES[tonic]
            st.jianpu = jianpu.render(q, grid, tonic)
            (d / "jianpu.txt").write_text(st.jianpu, encoding="utf8")
            (d / "score.json").write_text(json.dumps(_score_json(q, grid), ensure_ascii=False, indent=1))

        n = len(lead)
        if "mimic" in spec.modes:
            with self.stage("render_mimic"):
                self._mix("mimic", [(p, render_mimic(vb, p)) for p in phrases], n)
        if "score" in spec.modes:
            with self.stage("render_score"):
                self._mix("score", [(p, render_score(vb, p)) for p in q], n)
        st.outputs["lead"] = str(audio.to_mp3(lead_path, d / "lead.mp3", loudnorm=False))
        st.outputs["clip"] = str(audio.to_mp3(clip, d / "clip.mp3", loudnorm=False))

    def _mix(self, kind: str, rendered: list[tuple[Phrase, np.ndarray]], n: int) -> None:
        out = np.zeros(n + settings.SAMPLE_RATE, np.float32)
        for p, wav in rendered:
            s = max(0, int(round(_phrase_start(p, kind) * settings.SAMPLE_RATE)))
            seg = wav[: len(out) - s]
            out[s:s + len(seg)] += seg
        wav_path = audio.save(self.dir / f"{kind}.wav", out[:n])
        self.state.outputs[kind] = str(audio.to_mp3(wav_path, self.dir / f"{kind}.mp3"))


def _phrase_start(p: Phrase, kind: str) -> float:
    return p.phones[0][1] if kind == "mimic" else p.start


def _to_wav(src: Path, dst: Path) -> Path:
    audio.ffmpeg("-i", str(src), "-ac", "1", "-ar", str(settings.SAMPLE_RATE), str(dst))
    return dst


def _score_json(phrases: list[Phrase], grid: jianpu.Grid) -> dict:
    """Engine-independent score: what a future `.sing` 简谱 input would produce."""
    return {
        "bpm": grid.bpm,
        "phrases": [[{"text": s.text, "pinyin": s.pinyin, "lang": s.lang,
                      "notes": [{"start": round(n.start, 4), "dur": round(n.dur, 4), "midi": n.midi} for n in s.notes]}
                     for s in p.syllables] for p in phrases],
    }
