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
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import audio, onnx_backend, resolver, settings, sources, tools
from .pipeline import Job, JobSpec, voicebank
from .score import Note, Phrase, Syllable, render_score

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
ATTRIBUTION = "AI 合成 · 声库：用户固定音源 · 引擎：DiffSinger"

# Separation and torch models are memory heavy; run one job at a time.
_executor = ThreadPoolExecutor(max_workers=1)
_jobs: dict[str, Job] = {}
_job_futures: dict[str, Future] = {}
_delete_lock = Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _executor
    _executor = ThreadPoolExecutor(max_workers=1)
    try:
        yield
    finally:
        # Finish queued jobs before closing their model workers.
        await asyncio.to_thread(_executor.shutdown, wait=True)
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
    selection: Literal["random", "chorus"] = "random"
    seed: int | None = None
    lyric_query: str | None = Field(default=None, max_length=512)


@app.post("/release_task", dependencies=[Depends(auth)])
def release_task(body: TaskIn, owner: Owner = LEGACY_OWNER) -> dict:
    task_id = uuid.uuid4().hex[:12]
    job = Job(task_id, JobSpec(song_id=body.song_id, start=body.start, end=body.end, modes=body.modes,
                               selection=body.selection, seed=body.seed, lyric_query=body.lyric_query), owner_id=owner)
    _jobs[task_id] = job
    _job_futures[task_id] = _executor.submit(job.run)
    return {"task_id": task_id}


MAX_UPLOAD = 60 * 1024 * 1024


@app.post("/release_task_upload", dependencies=[Depends(auth)])
def release_task_upload(audio_file: UploadFile = File(...), lrc_file: UploadFile | None = File(None),
                        lrc_text: str = Form(""), start: float | None = Form(None), end: float | None = Form(None),
                        modes: str = Form("mimic,score"),
                        selection: Literal["random", "chorus"] = Form("random"),
                        seed: int | None = Form(None), lyric_query: str | None = Form(None, max_length=512),
                        owner: Owner = LEGACY_OWNER) -> dict:
    """Same pipeline for a song the source cannot provide: your own audio + LRC."""
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
                               lyric_query=lyric_query), owner_id=owner)
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
    st["attribution"] = ATTRIBUTION
    ev = settings.WORK / task_id / "eval.json"
    st["evaluation"] = json.loads(ev.read_text()) if ev.is_file() else None
    return st


@app.post("/evaluate", dependencies=[Depends(auth)])
def run_evaluation(task_id: str, owner: Owner = LEGACY_OWNER) -> dict:
    from .evaluate import evaluate

    if _state(task_id, owner)["status"] != "done":
        raise HTTPException(409, "task not finished")
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


@app.post("/render_score", dependencies=[Depends(auth)])
def render(body: ScoreIn) -> FileResponse:
    vb = voicebank()
    phrases = [Phrase([Syllable(s.text, s.pinyin, [Note(n.start, n.dur, n.midi) for n in s.notes], s.lang) for s in p])
               for p in body.phrases if p]
    if not phrases:
        raise HTTPException(400, "empty score")
    end = max(p.end for p in phrases) + 0.5
    out = np.zeros(int(end * settings.SAMPLE_RATE), np.float32)
    for p in phrases:
        wav = render_score(vb, p)
        s = max(0, int(round(p.start * settings.SAMPLE_RATE)))
        seg = wav[: len(out) - s]
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
            out.append({k: st.get(k) for k in ("task_id", "status", "stage", "song", "window", "error")})
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
        _jobs.pop(task_id, None)
        _job_futures.pop(task_id, None)
    return {"task_id": task_id, "deleted": True}


@app.get("/runtime", dependencies=[Depends(auth)])
def runtime() -> dict:
    return {"synthesis": onnx_backend.backend_status(), "workers": tools.worker_status(),
            "alignment": {lang: tools.alignment_backend(lang) for lang in ("zh", "ja", "en")},
            "note_extraction": {"backend": tools.note_backend(), "game_steps": settings.GAME_STEPS}}


@app.get("/search", dependencies=[Depends(auth)])
def search(q: str, limit: int = 10) -> list[dict]:
    return [vars(s) for s in sources.get_source().search(q, limit)]


@app.get("/resolve_song", dependencies=[Depends(auth)])
def resolve_song(q: str = Query(min_length=1, max_length=512)) -> dict:
    if not resolver.lyric_match.tokens(q):
        raise HTTPException(422, "query must contain a song title or lyrics")
    resolved = resolver.resolve(sources.get_source(), q)
    if resolved is None:
        raise HTTPException(404, "no matching song")
    return resolved


if settings.WEBUI:
    app.mount("/ui", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="ui")

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/ui/")
