"""HTTP API, shaped like the ACE-Step legacy API that Gokohime already polls.

POST /release_task      {"song_id": "123", "start": 60, "end": 85, "modes": ["mimic", "score"]}
GET  /query_result?task_id=...
GET  /v1/audio?task_id=...&kind=mimic|score|lead|clip
POST /release_task_upload  multipart: audio_file + lrc_file|lrc_text (+ start, end, modes)
POST /render_score      score.json body -> mp3 (synthesis only, no extraction)
POST /evaluate?task_id=  objective quality metrics (see evaluate.py)
GET  /tasks, /search?q=  helpers for the optional web UI (DSAPI_WEBUI=1, served at /ui)
"""

import asyncio
import json
import os
import re
import secrets
import shutil
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from threading import Lock
from typing import Annotated, Literal

import numpy as np
import yaml
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import audio, cache, onnx_backend, resolver, settings, sources, tools, voices
from .pipeline import Job, JobSpec, voicebank, inference_lock
from .score import Note, Phrase, Syllable, render_score
from .score_task import ScoreJob
from .text_score import ScoreError, parse_score

API_KEY = os.environ.get("DSAPI_KEY", "")
LEGACY_OWNER = "__legacy__"


def _load_api_keys(raw: str, legacy_key: str) -> dict[str, str]:
    """Stable user IDs let operators rotate credentials without losing jobs."""
    try:
        keys = json.loads(raw)
    except ValueError:
        raise ValueError("DSAPI_KEYS must be a JSON object mapping user IDs to API keys") from None
    if not isinstance(keys, dict) or any(
        not owner.strip() or owner == LEGACY_OWNER or not isinstance(key, str) or not key.strip()
        for owner, key in keys.items()
    ):
        raise ValueError("DSAPI_KEYS requires nonempty user IDs and keys; __legacy__ is reserved")
    if len(set(keys.values())) != len(keys) or (legacy_key and legacy_key in keys.values()):
        raise ValueError("API keys must be unique, including DSAPI_KEY")
    return keys


API_KEYS = _load_api_keys(os.environ.get("DSAPI_KEYS", "{}"), API_KEY)
MAX_LYRIC_TEXT = 16384

# Separation and torch models are memory heavy; run one job at a time.
_executor = ThreadPoolExecutor(max_workers=1)
_prefetch_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="download")
_jobs: dict[str, Job] = {}
_job_futures: dict[str, Future] = {}
_delete_lock = Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _executor, _prefetch_executor
    _executor = ThreadPoolExecutor(max_workers=1)
    _prefetch_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="download")
    try:
        yield
    finally:
        # Finish queued jobs before closing their model workers.
        await asyncio.to_thread(_executor.shutdown, wait=True)
        await asyncio.to_thread(_prefetch_executor.shutdown, wait=True)
        await asyncio.to_thread(tools.close_workers)


app = FastAPI(title="dsapi", lifespan=lifespan)


def auth(authorization: str = Header(default=""), key: str = Query(default="")) -> str:
    # ?key= lets <audio src> in the web UI authenticate without headers.
    if not API_KEY and not API_KEYS:
        return LEGACY_OWNER
    # An explicit header takes precedence over a query-string credential.
    token = authorization.removeprefix("Bearer ") if authorization.startswith("Bearer ") else ""
    if not authorization:
        token = key
    for owner, secret in API_KEYS.items():
        if secrets.compare_digest(token.encode(), secret.encode()):
            return owner
    if API_KEY and secrets.compare_digest(token.encode(), API_KEY.encode()):
        return LEGACY_OWNER
    raise HTTPException(401, "bad api key")


Owner = Annotated[str, Depends(auth)]


class TaskIn(BaseModel):
    song_id: str
    start: float | None = None
    end: float | None = None
    modes: list[str] = Field(default_factory=lambda: ["mimic", "score"])
    with_accompaniment: bool = False
    audio_only: bool = False
    selection: Literal["random", "chorus"] = "random"
    seed: int | None = None
    lyric_query: str | None = Field(default=None, max_length=MAX_LYRIC_TEXT)
    lyric_extent: Literal["auto", "window", "exact"] = "auto"
    voice_id: str = voices.DEFAULT_VOICE
    synthesis_seed: int | None = Field(default=None, ge=0, le=2**32-1)


def validate_voice(voice_id: str) -> None:
    try:
        voice = voices.get(voice_id)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    try:
        voice.validate()
    except (OSError, ValueError, KeyError, yaml.YAMLError) as exc:
        raise HTTPException(503, f"音源 {voice.name} 暂不可用，请先导入并配置本地音源及兼容的声码器") from exc


@app.get("/voices", dependencies=[Depends(auth)])
def list_voices() -> dict:
    return voices.catalog()


@app.post("/release_task", dependencies=[Depends(auth)])
def release_task(body: TaskIn, owner: Owner = LEGACY_OWNER) -> dict:
    validate_voice(body.voice_id)
    if body.audio_only and body.modes != ["mimic"]:
        raise HTTPException(422, "audio_only requires modes=[mimic]")
    task_id = uuid.uuid4().hex[:12]
    job = Job(task_id, JobSpec(song_id=body.song_id, start=body.start, end=body.end, modes=body.modes,
                               selection=body.selection, seed=body.seed, lyric_query=body.lyric_query,
                               lyric_extent=body.lyric_extent, voice_id=body.voice_id, synthesis_seed=body.synthesis_seed,
                               with_accompaniment=body.with_accompaniment, audio_only=body.audio_only), owner_id=owner)
    _jobs[task_id] = job
    job.prefetch(_prefetch_executor)
    _job_futures[task_id] = _executor.submit(job.run)
    return {"task_id": task_id}


MAX_UPLOAD = 60 * 1024 * 1024


class TextScoreIn(BaseModel):
    text: str = Field(min_length=1, max_length=16384)
    voice_id: str = voices.DEFAULT_VOICE
    synthesis_seed: int | None = Field(default=None, ge=0, le=2**32-1)


def _parse_text_score(body: TextScoreIn):
    validate_voice(body.voice_id)
    try:
        return parse_score(body.text, voicebank(body.voice_id))
    except ScoreError as exc:
        raise HTTPException(422, exc.detail) from exc


@app.post("/parse_score", dependencies=[Depends(auth)])
def preview_score(body: TextScoreIn) -> dict:
    return _parse_text_score(body).payload()


@app.post("/release_score_task", dependencies=[Depends(auth)])
def release_score_task(body: TextScoreIn, owner: Owner = LEGACY_OWNER) -> dict:
    parsed = _parse_text_score(body)
    task_id = uuid.uuid4().hex[:12]
    job = ScoreJob(task_id, JobSpec(score_text=body.text, modes=["score"], voice_id=body.voice_id,
                                  synthesis_seed=body.synthesis_seed), parsed, owner_id=owner)
    _jobs[task_id] = job
    _job_futures[task_id] = _executor.submit(job.run)
    return {"task_id": task_id}


@app.post("/release_task_upload", dependencies=[Depends(auth)])
def release_task_upload(audio_file: UploadFile = File(...), lrc_file: UploadFile | None = File(None),
                        lrc_text: str = Form(""), start: float | None = Form(None), end: float | None = Form(None),
                        modes: str = Form("mimic,score"), with_accompaniment: bool = Form(False),
                        audio_only: bool = Form(False),
                        selection: Literal["random", "chorus"] = Form("random"),
                        seed: int | None = Form(None), lyric_query: str | None = Form(None, max_length=MAX_LYRIC_TEXT),
                        voice_id: str = Form(voices.DEFAULT_VOICE),
                        lyric_extent: Literal["auto", "window", "exact"] = Form("auto"),
                        synthesis_seed: int | None = Form(None, ge=0, le=2**32-1),
                        owner: Owner = LEGACY_OWNER) -> dict:
    """Same pipeline for a song the source cannot provide: your own audio + LRC."""
    validate_voice(voice_id)
    if audio_only and [m for m in modes.split(",") if m] != ["mimic"]:
        raise HTTPException(422, "audio_only requires modes=mimic")
    lrc = lrc_file.file.read().decode("utf-8-sig", errors="replace") if lrc_file else lrc_text
    if not lrc.strip():
        raise HTTPException(400, "LRC lyrics are required (lrc_file or lrc_text)")
    task_id = uuid.uuid4().hex[:12]
    d = settings.WORK / "uploads" / task_id
    d.mkdir(parents=True, exist_ok=True)
    # Keep only a safe extension from the client's filename.
    suffix = Path(audio_file.filename or "").suffix.lower()
    suffix = suffix if suffix in {".mp3", ".flac", ".wav", ".m4a", ".ogg", ".aac"} else ".audio"
    stem = Path(audio_file.filename or "upload").stem[:80] or "upload"
    audio_path = d / f"{stem}{suffix}"
    size = 0
    with audio_path.open("wb") as f:
        while chunk := audio_file.file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD:
                raise HTTPException(413, "audio file too large")
            f.write(chunk)
    (d / "lyrics.lrc").write_text(lrc, encoding="utf8")
    job = Job(task_id, JobSpec(audio_path=str(audio_path), lrc_path=str(d / "lyrics.lrc"), start=start, end=end,
                               modes=[m for m in modes.split(",") if m], selection=selection, seed=seed,
                               lyric_query=lyric_query, lyric_extent=lyric_extent, voice_id=voice_id,
                               synthesis_seed=synthesis_seed, with_accompaniment=with_accompaniment,
                               audio_only=audio_only), owner_id=owner)
    _jobs[task_id] = job
    _job_futures[task_id] = _executor.submit(job.run)
    return {"task_id": task_id}


def _task_dir(task_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", task_id):
        raise HTTPException(404, "unknown task")
    path = settings.WORK / task_id
    if path.is_symlink():
        raise HTTPException(404, "unknown task")
    return path


def _state(task_id: str, owner: str) -> dict:
    path = _task_dir(task_id) / "job.json"
    if task_id in _jobs:
        state = asdict(_jobs[task_id].state)
    elif path.is_file():
        state = json.loads(path.read_text())
    else:
        raise HTTPException(404, "unknown task")
    if (state.pop("owner_id", None) or LEGACY_OWNER) != owner:
        raise HTTPException(404, "unknown task")
    state["voice"] = state.get("voice") or dict(voices.LEGACY_VOICE)
    return state


@app.get("/query_result", dependencies=[Depends(auth)])
def query_result(task_id: str, owner: Owner = LEGACY_OWNER) -> dict:
    st = _state(task_id, owner)
    # Older saved jobs have no timestamps; the UI falls back to stage timings.
    started = st.get("started_at")
    end = st.get("finished_at")
    st["elapsed_seconds"] = (
        max(0.0, (end if end is not None else time.time()) - started)
        if started is not None and (end is not None or st["status"] == "running") else None
    )
    st["outputs"] = {k: f"/v1/audio?task_id={task_id}&kind={k}" for k in st["outputs"]}
    st["voice"] = st.get("voice") or dict(voices.LEGACY_VOICE)
    st["attribution"] = f"AI 合成 · 声库：{st['voice']['name']} · 引擎：DiffSinger"
    ev = settings.WORK / task_id / "eval.json"
    st["evaluation"] = json.loads(ev.read_text()) if ev.is_file() else None
    return st


@app.post("/evaluate", dependencies=[Depends(auth)])
def run_evaluation(task_id: str, owner: Owner = LEGACY_OWNER) -> dict:
    from .evaluate import evaluate

    state = _state(task_id, owner)
    if state["status"] != "done":
        raise HTTPException(409, "task not finished")
    if state["spec"].get("score_text") is not None:
        raise HTTPException(422, "自填乐谱没有原唱，无法运行原唱对比评估")
    return evaluate(settings.WORK / task_id)


@app.get("/v1/audio", dependencies=[Depends(auth)])
def get_audio(task_id: str, kind: str = "mimic", owner: Owner = LEGACY_OWNER) -> FileResponse:
    path = _state(task_id, owner)["outputs"].get(kind)
    if not path:
        raise HTTPException(404, f"no {kind} output")
    return FileResponse(path, media_type="audio/mpeg")


@app.get("/v1/score", dependencies=[Depends(auth)])
def get_score(task_id: str, owner: Owner = LEGACY_OWNER) -> FileResponse:
    _state(task_id, owner)
    path = settings.WORK / task_id / "score.json"
    if not path.is_file():
        raise HTTPException(404, "no score yet")
    return FileResponse(path, media_type="application/json")


class NoteIn(BaseModel):
    start: float
    dur: float
    midi: int


class SyllableIn(BaseModel):
    text: str
    pinyin: str
    notes: list[NoteIn]
    lang: str = "zh"


class ScoreIn(BaseModel):
    phrases: list[list[SyllableIn]]
    voice_id: str = voices.DEFAULT_VOICE
    synthesis_seed: int | None = Field(default=None, ge=0, le=2**32-1)


@app.post("/render_score", dependencies=[Depends(auth)])
def render(body: ScoreIn) -> FileResponse:
    validate_voice(body.voice_id)
    with inference_lock:
        return _render(body)


def _render(body: ScoreIn) -> FileResponse:
    vb = voicebank(body.voice_id, body.synthesis_seed)
    phrases = [Phrase([Syllable(s.text, s.pinyin, [Note(n.start, n.dur, n.midi) for n in s.notes], s.lang) for s in p])
               for p in body.phrases if p]
    if not phrases:
        raise HTTPException(400, "empty score")
    end = max(p.end for p in phrases) + 0.5
    out = np.zeros(int(end * settings.SAMPLE_RATE), np.float32)
    for p in phrases:
        wav = render_score(vb, p)
        offset = int(round(p.start * settings.SAMPLE_RATE))
        s, skip = max(0, offset), max(0, -offset)
        seg = wav[skip:skip + max(0, len(out) - s)]
        out[s:s + len(seg)] += seg
    d = settings.WORK / "render" / uuid.uuid4().hex[:12]
    d.mkdir(parents=True, exist_ok=True)
    mp3 = audio.to_mp3(audio.save(d / "out.wav", out), d / "out.mp3")
    return FileResponse(Path(mp3), media_type="audio/mpeg")


@app.get("/tasks", dependencies=[Depends(auth)])
def tasks(limit: Annotated[int, Query(ge=1, le=1000)] = 30, owner: Owner = LEGACY_OWNER) -> list[dict]:
    with _delete_lock:
        jobs = sorted(settings.WORK.glob("*/job.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        out = []
        for path in jobs:
            try:
                st = _state(path.parent.name, owner)
            except HTTPException as exc:
                if exc.status_code == 404:
                    continue
                raise
            out.append({k: st.get(k) for k in ("task_id", "status", "stage", "song", "window", "error", "voice")})
            if len(out) >= limit:
                break
    return out


@app.delete("/tasks/{task_id}", dependencies=[Depends(auth)])
def delete_task(task_id: str, owner: Owner = LEGACY_OWNER) -> dict:
    with _delete_lock:
        directory = _task_dir(task_id)
        state = _state(task_id, owner)
        future = _job_futures.get(task_id)
        # A terminal status can appear just before the worker's final save.
        if state["status"] not in {"done", "failed"} or (future is not None and not future.done()):
            raise HTTPException(409, "cannot delete a queued or running task")
        uploads = settings.WORK / "uploads" / task_id
        if uploads.resolve().parent != (settings.WORK.resolve() / "uploads") or uploads.is_symlink():
            raise HTTPException(409, "unsafe upload directory")
        if uploads.exists():
            shutil.rmtree(uploads)
        if directory.exists():
            shutil.rmtree(directory)
        cache.purge_owner(owner)
        _jobs.pop(task_id, None)
        _job_futures.pop(task_id, None)
    return {"task_id": task_id, "deleted": True}


@app.get("/runtime", dependencies=[Depends(auth)])
def runtime() -> dict:
    return {"synthesis": onnx_backend.backend_status(), "workers": tools.worker_status(),
            "alignment": {lang: tools.alignment_backend(lang) for lang in ("zh", "ja", "en")},
            "note_extraction": {"backend": tools.note_backend(), "game_steps": settings.GAME_STEPS},
            "optimization": {"cache": settings.CACHE_ENABLED, "cache_ttl_seconds": settings.CACHE_TTL,
                             "cache_max_bytes": settings.CACHE_MAX_BYTES, "prefetch": settings.PREFETCH_DOWNLOADS}}


@app.get("/search", dependencies=[Depends(auth)])
def search(q: str, limit: int = 10) -> list[dict]:
    return [vars(s) for s in sources.get_source().search(q, limit)]


class ResolveIn(BaseModel):
    q: str = Field(min_length=1, max_length=MAX_LYRIC_TEXT)
    lyric_extent: Literal["auto", "window", "exact"] = "auto"


def _resolve(q: str, lyric_extent: str = "auto") -> dict:
    try:
        resolved = resolver.resolve(sources.get_source(), q, lyric_extent)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if resolved is None:
        raise HTTPException(404, "no matching song")
    return resolved


@app.get("/resolve_song", dependencies=[Depends(auth)])
def resolve_song(q: str = Query(min_length=1, max_length=MAX_LYRIC_TEXT),
                 lyric_extent: Literal["auto", "window", "exact"] = "auto") -> dict:
    return _resolve(q, lyric_extent)


@app.post("/resolve_song", dependencies=[Depends(auth)])
def resolve_song_post(body: ResolveIn) -> dict:
    return _resolve(body.q, body.lyric_extent)


if settings.WEBUI:
    app.mount("/ui", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="ui")

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/ui/")
