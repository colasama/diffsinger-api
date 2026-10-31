"""End-to-end job: song -> lyric-selected clip -> lead vocal -> score -> re-sung audio.

Every stage writes into work/<task_id>/ and records its timing in job.json so
a failed run shows exactly where it stopped.
"""

import json
import os
import shutil
import time
import traceback
import secrets
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from importlib.metadata import version
from threading import RLock
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import audio, cache, extract, jianpu, lrc, selection, settings, sources, tools, voices
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
    lyric_extent: str = "auto"
    voice_id: str = voices.DEFAULT_VOICE
    synthesis_seed: int | None = None
    with_accompaniment: bool = False
    audio_only: bool = False
    score_text: str | None = None


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
    mixing: dict = field(default_factory=dict)
    voice: dict | None = None
    owner_id: str | None = None
    submitted_at: float = field(default_factory=lambda: time.time())
    cache_hits: dict[str, bool] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)


inference_lock = RLock()


def voicebank(voice_id: str | None = None, seed: int | None = None) -> Voicebank:
    voice = voices.get(voice_id)
    return Voicebank(voice.root, speaker=voice.speaker, seed=seed)


class Job:
    def __init__(self, task_id: str, spec: JobSpec, *, owner_id: str | None = None):
        if spec.audio_only and spec.modes != ["mimic"]:
            raise ValueError("audio_only requires modes=[mimic]")
        self._source_future: Future | None = None
        self.dir = settings.WORK / task_id
        self.dir.mkdir(parents=True, exist_ok=True)
        if spec.synthesis_seed is None:
            spec.synthesis_seed = secrets.randbits(32)
        self.state = JobState(task_id, spec, owner_id=owner_id, voice=voices.get(spec.voice_id).snapshot())
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
        st.metrics["queue_seconds"] = max(0., st.started_at - st.submitted_at)
        self.save()
        try:
            with inference_lock:
                self._run()
            st.status, st.stage = "done", ""
        except Exception as e:  # noqa: BLE001 - surfaced to the API caller
            st.status, st.error = "failed", f"{type(e).__name__}: {e}"
            (self.dir / "error.txt").write_text(traceback.format_exc())
        st.finished_at = time.time()
        self.save()
        return st

    @contextmanager
    def measure(self, name: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.state.metrics[name] = round(time.perf_counter() - started, 4)

    def prefetch(self, executor):
        if settings.PREFETCH_DOWNLOADS and self.state.spec.song_id and not self.state.spec.audio_path:
            self._source_future = executor.submit(self._fetch_source)

    def _fetch_source(self) -> dict:
        # Prefetch never changes JobState: the inference thread owns progress.
        started = time.perf_counter()
        spec, d = self.state.spec, self.dir
        hit = False
        if spec.audio_path:
            src = d / ("source" + Path(spec.audio_path).suffix)
            shutil.copyfile(spec.audio_path, src)
            lyrics = Path(spec.lrc_path).read_text(encoding="utf8")
            song = {"name": Path(spec.audio_path).stem}
            (d / "lyrics.lrc").write_text(lyrics, encoding="utf8")
        else:
            source = sources.get_source()
            artifacts = cache.ArtifactCache(self.state.owner_id)
            key = artifacts.key("fetch", {"song_id": spec.song_id,
                "source": type(source).__module__ + "." + type(source).__qualname__, "config": vars(source)})
            outputs = {"source.audio": d / "source.audio", "lyrics.lrc": d / "lyrics.lrc", "song.json": d / "song.json"}
            with artifacts.guard(key):
                hit = artifacts.restore(key, outputs)
                if not hit:
                    with ThreadPoolExecutor(max_workers=2) as fetchers:
                        info = fetchers.submit(source.song_info, spec.song_id)
                        lyric = fetchers.submit(source.lyric, spec.song_id)
                        source.download_audio(spec.song_id, outputs["source.audio"])
                        song, lyrics = asdict(info.result()), lyric.result()
                    outputs["song.json"].write_text(json.dumps(song, ensure_ascii=False))
                    outputs["lyrics.lrc"].write_text(lyrics, encoding="utf8")
                    # Reject incomplete media before publishing a reusable download.
                    audio.duration(outputs["source.audio"])
                    artifacts.store(key, outputs)
            src = outputs["source.audio"]
            song = json.loads(outputs["song.json"].read_text())
            lyrics = outputs["lyrics.lrc"].read_text(encoding="utf8")
        return {"source": str(src), "song": song, "lyrics": lyrics, "duration": audio.duration(src),
                "cache_hit": hit, "seconds": round(time.perf_counter() - started, 4)}

    def _run(self) -> None:
        st, spec, d = self.state, self.state.spec, self.dir
        with self.stage("fetch"):
            prepared = self._source_future.result() if self._source_future else self._fetch_source()
            self._source_future = None
            src, lrc_text, duration = Path(prepared["source"]), prepared["lyrics"], prepared["duration"]
            st.song = prepared["song"]
            st.cache_hits["fetch"] = prepared["cache_hit"]
            st.metrics["fetch_work_seconds"] = prepared["seconds"]
        with self.measure("voicebank_seconds"):
            vb = voicebank(spec.voice_id, spec.synthesis_seed)
        artifacts = cache.ArtifactCache(st.owner_id)
        runtime = {"packages": {name: version(name) for name in ("onnxruntime", "torch", "torchaudio", "librosa", "numpy", "scipy", "praat-parselmouth")},
                   "backend": os.environ.get("DSAPI_ONNX_BACKEND", "auto"),
                   "device": settings.torch_device()}

        with self.stage("select"):
            lines = lrc.parse(lrc_text, duration, trim_gaps=spec.start is None)
            selected = selection.select(lines, duration, start=spec.start, end=spec.end,
                                        strategy=spec.selection, seed=spec.seed, max_clip=MAX_CLIP,
                                        lyric_query=spec.lyric_query, lyric_extent=spec.lyric_extent)
            if selected.strategy == "lyric_exact":
                # Explicit lyrics must not lose held final syllables to the
                # short-window parser's estimated singing-length heuristic.
                selected = selection.select(lrc.parse(lrc_text, duration, trim_gaps=False), duration,
                                            lyric_query=spec.lyric_query, lyric_extent="exact")
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
            key = artifacts.key("separate", {"audio": cache.digest(clip), "runtime": runtime,
                "separator": version("audio-separator"), "vocals": settings.VOCALS_MODEL,
                "karaoke": settings.KARAOKE_MODEL, "protection": settings.KARAOKE_PROTECTION,
                "compile": settings.SEPARATOR_COMPILE, "sample_rate": settings.SAMPLE_RATE,
                "models": cache.files([settings.SEPARATOR_MODELS / settings.VOCALS_MODEL,
                    *([settings.SEPARATOR_MODELS / settings.KARAOKE_MODEL] if settings.KARAOKE_MODEL else []),
                    *settings.SEPARATOR_MODELS.glob("*.yaml"), *settings.SEPARATOR_MODELS.glob("*.json")])})
            outputs = {"lead.wav": d / "sep/lead.wav", "instrumental.wav": d / "sep/instrumental.wav",
                       "separation_meta.json": d / "sep/separation_meta.json"}
            st.cache_hits["separate"] = artifacts.restore(key, outputs)
            if not st.cache_hits["separate"]:
                lead_path, inst_path = audio.separate_long_vocals(clip, d / "sep")
                reports = []
                for path in sorted((d / "sep").rglob("lead_protection.json")):
                    report = json.loads(path.read_text())
                    reports.append({"chunk": str(path.parent.relative_to(d / "sep")),
                                    **{k: v for k, v in report.items() if k not in ("original", "separated", "output")}})
                outputs["separation_meta.json"].write_text(json.dumps(reports))
                artifacts.store(key, {**outputs, "lead.wav": lead_path, "instrumental.wav": inst_path})
            else:
                lead_path, inst_path = outputs["lead.wav"], outputs["instrumental.wav"]
            lead = audio.load(lead_path)

        with self.stage("align"):
            segs = extract.write_segments(chosen, lead, d / "segments", vb, before=before, after=after,
                                          clip_start=start)
            if not segs:
                raise RuntimeError("no lyric units the voicebank can sing in this window")
            ckpt, dictionary = Path(segs[0]["ckpt"]), Path(segs[0]["dictionary"])
            key = artifacts.key("align", {"runtime": runtime, "backend": segs[0]["backend"],
                "language": segs[0]["lang"], "model": cache.files([ckpt.parent]),
                "dictionary": cache.digest(dictionary),
                "inputs": {p.name: cache.digest(p) for p in sorted((d / "segments").glob("*"))
                           if p.suffix in (".wav", ".lab")},
                "vendor": cache.files((settings.VENDOR / ("HubertFA" if segs[0]["backend"] == "hubertfa" else "SOFA")).rglob("*.py"), content=True)})
            st.cache_hits["align"] = artifacts.restore(key, {"sofa.json": d / "sofa.json"})
            if st.cache_hits["align"]:
                sofa = json.loads((d / "sofa.json").read_text())
            else:
                sofa = tools.align(d / "segments", d, ckpt, dictionary,
                                   backend=segs[0]["backend"], lang=segs[0]["lang"])
                if not any("error" in value for value in sofa.values()):
                    artifacts.store(key, {"sofa.json": d / "sofa.json"})

        with self.stage("notes_f0"):
            backend = tools.note_backend()
            if backend == "game-medium" or spec.audio_only:
                with self.measure("f0_seconds"):
                    key = artifacts.key("f0", {"audio": cache.digest(lead_path), "runtime": runtime,
                        "model": cache.files([settings.RMVPE_CKPT]),
                        "vendor": cache.files([settings.VENDOR / "SOME/modules/rmvpe",
                                               settings.VENDOR / "SOME/utils/pitch_utils.py"], content=True)})
                    st.cache_hits["f0"] = artifacts.restore(key, {"f0.npy": d / "f0.npy"})
                    if st.cache_hits["f0"]:
                        f0 = np.load(d / "f0.npy")[0]
                    else:
                        f0, _ = tools.extract_f0(lead_path, d)
                        artifacts.store(key, {"f0.npy": d / "f0.npy"})
                if spec.audio_only:
                    notes = []
                    st.note_extraction = {"backend": "skipped", "reason": "audio_only"}
                else:
                    with self.measure("game_seconds"):
                        regions, _ = extract.build_phrases(vb, segs, sofa, [], f0)
                        if not regions:
                            raise RuntimeError("alignment produced no usable phrases")
                        boundaries = [{"phones": p.phones, "syllables": [(x.start, x.end, x.lang) for x in p.syllables]} for p in regions]
                        key = artifacts.key("game", {"audio": cache.digest(lead_path), "runtime": runtime,
                            "models": cache.files([settings.GAME_MODEL]), "steps": settings.GAME_STEPS,
                            "regions": boundaries, "language": segs[0]["lang"]})
                        game_outputs = {"notes.json": d / "notes.json", "game_regions.json": d / "game_regions.json"}
                        st.cache_hits["game"] = artifacts.restore(key, game_outputs)
                        if st.cache_hits["game"]:
                            notes = json.loads((d / "notes.json").read_text())["notes"]
                        else:
                            notes = tools.extract_game_notes(lead_path, d, regions, segs[0]["lang"])
                            artifacts.store(key, game_outputs)
            else:
                notes, f0, _ = tools.extract_some_notes_f0(lead_path, d)
            if not spec.audio_only:
                meta = json.loads((d / "notes.json").read_text())
                st.note_extraction = {k: v for k, v in meta.items() if k != "notes"}
                st.note_extraction.setdefault("backend", backend)

        with self.stage("build"):
            phrases, report = extract.build_phrases(vb, segs, sofa, notes, f0)
            st.alignment = report
            if any("error" in item for item in report):
                raise RuntimeError("Alignment failed for part of the requested lyrics")
            if not phrases:
                raise RuntimeError("alignment produced no usable phrases")

        if not spec.audio_only:
            with self.stage("quantize"):
                grid = jianpu.estimate_grid(inst_path, phrases)
                tonic = jianpu.estimate_tonic(phrases)
                notation = jianpu.quantize(phrases, grid)
                st.grid, st.tonic = asdict(grid), jianpu._NAMES[tonic]
                st.jianpu = jianpu.render(notation, grid, tonic)
                (d / "jianpu.txt").write_text(st.jianpu, encoding="utf8")
                (d / "score.json").write_text(json.dumps(_score_json(phrases, grid), ensure_ascii=False, indent=1))

        n = len(lead)
        if "mimic" in spec.modes:
            with self.stage("render_mimic"):
                self._mix("mimic", ((p, render_mimic(vb, p)) for p in phrases), n, inst_path, lead_path)
        if "score" in spec.modes:
            with self.stage("render_score"):
                self._mix("score", ((p, render_score(vb, p)) for p in phrases), n, inst_path, lead_path)
        st.metrics.update({f"synthesis_{key}_seconds": round(value, 4)
                           for key, value in getattr(vb, "timings", {}).items()})
        with self.measure("reference_encode_seconds"):
            st.outputs["lead"] = str(audio.to_mp3(lead_path, d / "lead.mp3", loudnorm=False))
            st.outputs["clip"] = str(audio.to_mp3(clip, d / "clip.mp3", loudnorm=False))

    def _mix(self, kind: str, rendered: Iterable[tuple[Phrase, np.ndarray]], n: int,
             instrumental: Path | None = None, reference: Path | None = None) -> None:
        out = np.zeros(n + settings.SAMPLE_RATE, np.float32)
        boundaries = []
        for p, wav in rendered:
            offset = int(round(_phrase_start(p, kind) * settings.SAMPLE_RATE))
            s, skip = max(0, offset), max(0, -offset)
            seg = wav[skip:skip + max(0, len(out) - s)]
            out[s:s + len(seg)] += seg
            boundaries.append(min(n, s + len(seg)))
        wav_path = audio.save(self.dir / f"{kind}.wav", out[:n], subtype="FLOAT")
        self._export_audio(kind, wav_path, n, boundaries)
        if self.state.spec.with_accompaniment:
            if instrumental is None or reference is None:
                raise ValueError("Accompaniment mixing requires both source stems")
            mixed = self.dir / f"{kind}_mix.wav"
            self.state.mixing[kind] = audio.mix_accompaniment(wav_path, instrumental, reference, mixed)
            self._export_audio(kind + "_mix", mixed, n, boundaries, normalized=True)

    def _export_audio(self, kind: str, wav_path: Path, n: int, boundaries: list[int],
                      *, normalized: bool = False) -> None:
        mp3 = audio.to_mp3(wav_path, self.dir / f"{kind}.mp3", loudnorm=not normalized)
        self.state.outputs[kind] = str(mp3)
        # Mixed parts share the full file's normalization, including instrumental gaps.
        if mp3.stat().st_size > 28 * 1024 * 1024:
            start, i = 0, 0
            while start < n:
                target = min(n, start + settings.SAMPLE_RATE * 600)
                stop = max((b for b in boundaries if start < b <= target), default=target) if target < n else n
                part = audio.cut(wav_path, self.dir / f"{kind}_part_{i:04}.wav",
                                 start / settings.SAMPLE_RATE, stop / settings.SAMPLE_RATE)
                self.state.outputs[f"{kind}_part_{i:04}"] = str(
                    audio.to_mp3(part, part.with_suffix(".mp3"), loudnorm=not normalized))
                part.unlink()
                start, i = stop, i + 1


def _phrase_start(p: Phrase, kind: str) -> float:
    return p.phones[0][1] if kind == "mimic" else p.start


def _to_wav(src: Path, dst: Path) -> Path:
    audio.ffmpeg("-i", str(src), "-ac", "1", "-ar", str(settings.SAMPLE_RATE), str(dst))
    return dst


def _score_json(phrases: list[Phrase], grid: jianpu.Grid) -> dict:
    """Synthesis score with extracted timing in seconds; BPM is notation metadata."""
    return {
        "bpm": grid.bpm,
        "phrases": [[{"text": s.text, "pinyin": s.pinyin, "lang": s.lang,
                      "notes": [{"start": round(n.start, 4), "dur": round(n.dur, 4), "midi": n.midi} for n in s.notes]}
                     for s in p.syllables] for p in phrases],
    }
