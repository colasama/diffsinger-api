"""HTTP API, shaped like the ACE-Step legacy API that Gokohime already polls.

POST /release_task      {"song_id": "123", "start": 60, "end": 85, "modes": ["mimic", "score"]}
GET  /query_result?task_id=...
GET  /v1/audio?task_id=...&kind=mimic|score|lead|clip
POST /release_task_upload  multipart: audio_file + lrc_file|lrc_text (+ start, end, modes)
POST /render_score      score.json body -> mp3 (synthesis only, no extraction)
POST /evaluate?task_id=  objective quality metrics (see evaluate.py)
GET  /tasks, /search?q=  helpers for the optional web UI (DSAPI_WEBUI=1, served at /ui)
"""

import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import audio, settings, sources
from .pipeline import Job, JobSpec, voicebank
from .score import Note, Phrase, Syllable, render_score

API_KEY = os.environ.get("DSAPI_KEY", "")
ATTRIBUTION = "AI 合成 · 声库：用户固定音源 · 引擎：DiffSinger"

app = FastAPI(title="dsapi")
# Separation and torch models are memory heavy; run one job at a time.
_executor = ThreadPoolExecutor(max_workers=1)
_jobs: dict[str, Job] = {}


def auth(authorization: str = Header(default=""), key: str = Query(default="")) -> None:
    # ?key= lets <audio src> in the web UI authenticate without headers.
    if API_KEY and authorization != f"Bearer {API_KEY}" and key != API_KEY:
        raise HTTPException(401, "bad api key")


class TaskIn(BaseModel):
    song_id: str
    start: float | None = None
    end: float | None = None
    modes: list[str] = Field(default_factory=lambda: ["mimic", "score"])


@app.post("/release_task", dependencies=[Depends(auth)])
def release_task(body: TaskIn) -> dict:
    task_id = uuid.uuid4().hex[:12]
    job = Job(task_id, JobSpec(song_id=body.song_id, start=body.start, end=body.end, modes=body.modes))
    _jobs[task_id] = job
    _executor.submit(job.run)
    return {"task_id": task_id}


MAX_UPLOAD = 60 * 1024 * 1024


@app.post("/release_task_upload", dependencies=[Depends(auth)])
def release_task_upload(audio_file: UploadFile = File(...), lrc_file: UploadFile | None = File(None),
                        lrc_text: str = Form(""), start: float | None = Form(None), end: float | None = Form(None),
                        modes: str = Form("mimic,score")) -> dict:
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
                               modes=[m for m in modes.split(",") if m]))
    _jobs[task_id] = job
    _executor.submit(job.run)
    return {"task_id": task_id}


def _state(task_id: str) -> dict:
    if task_id in _jobs:
        return asdict(_jobs[task_id].state)
    path = settings.WORK / task_id / "job.json"
    if not path.is_file() or "/" in task_id:
        raise HTTPException(404, "unknown task")
    return json.loads(path.read_text())


@app.get("/query_result", dependencies=[Depends(auth)])
def query_result(task_id: str) -> dict:
    st = _state(task_id)
    st["outputs"] = {k: f"/v1/audio?task_id={task_id}&kind={k}" for k in st["outputs"]}
    st["attribution"] = ATTRIBUTION
    ev = settings.WORK / task_id / "eval.json"
    st["evaluation"] = json.loads(ev.read_text()) if ev.is_file() else None
    return st


@app.post("/evaluate", dependencies=[Depends(auth)])
def run_evaluation(task_id: str) -> dict:
    from .evaluate import evaluate

    if _state(task_id)["status"] != "done":
        raise HTTPException(409, "task not finished")
    return evaluate(settings.WORK / task_id)


@app.get("/v1/audio", dependencies=[Depends(auth)])
def get_audio(task_id: str, kind: str = "mimic") -> FileResponse:
    path = _state(task_id)["outputs"].get(kind)
    if not path:
        raise HTTPException(404, f"no {kind} output")
    return FileResponse(path, media_type="audio/mpeg")


@app.get("/v1/score", dependencies=[Depends(auth)])
def get_score(task_id: str) -> FileResponse:
    _state(task_id)
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
def tasks(limit: int = 30) -> list[dict]:
    jobs = sorted(settings.WORK.glob("*/job.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    out = []
    for path in jobs:
        st = json.loads(path.read_text())
        out.append({k: st.get(k) for k in ("task_id", "status", "stage", "song", "window", "error")})
    return out


@app.get("/search", dependencies=[Depends(auth)])
def search(q: str, limit: int = 10) -> list[dict]:
    return [vars(s) for s in sources.get_source().search(q, limit)]


if settings.WEBUI:
    app.mount("/ui", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="ui")

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/ui/")
