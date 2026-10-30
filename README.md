# dsapi

Re-sing a song's chorus with a DiffSinger voicebank: fetch the song, pick the
chorus from its LRC, isolate the lead vocal, align lyrics, transcribe notes,
then render with ImportedFixed (用户固定音源). No LLM anywhere in the pipeline.

```
Song id ─▶ audio + LRC ─▶ chorus window ─▶ BS-RoFormer + karaoke (lead vocal)
          ─▶ HubertFA alignment ─▶ GAME Medium notes + RMVPE f0
          ─▶ phrases ─┬─▶ mimic: real phoneme timing + source f0 ─▶ variance/acoustic/vocoder
                      └─▶ score: grid-quantized notes ─▶ dur/pitch/variance/acoustic/vocoder
          ─▶ 简谱 text + score.json
```

## Setup

```sh
uv sync
scripts/fetch_models.sh   # includes HubertFA v0.0.7 and GAME Medium v1.0.3
```

## Run

```sh
uv run dsapi run 3414700455                     # chorus auto-picked
uv run dsapi run 3414700455 --start 174 --end 195
uv run dsapi serve                              # http://127.0.0.1:8765/docs
```

Artifacts land in `work/<task_id>/`: `clip.mp3`, `lead.mp3`, `mimic.mp3`,
`score.mp3`, `jianpu.txt`, `score.json`, `job.json` (per-stage timings and
alignment report), and `error.txt` on failure.

## API

| Method | Path | Notes |
|---|---|---|
| POST | `/release_task` | `{"song_id", "start"?, "end"?, "modes"?}` → `{"task_id"}` |
| GET | `/query_result?task_id=` | status, current stage, timings, 简谱, output URLs |
| GET | `/v1/audio?task_id=&kind=` | `mimic` / `score` / `lead` / `clip` |
| POST | `/render_score` | body is `score.json`'s shape; synthesis only |

Set `DSAPI_KEY` to require `Authorization: Bearer <key>`.

## Inference backend

Synthesis defaults to **CUDA GPU → WebGPU → CPU**, trying the next backend if
session initialization fails. On Apple Silicon, native WebGPU uses Metal.
The plugin is included in `uv sync`. Set `DSAPI_ONNX_BACKEND=cpu|cuda|webgpu`
to require a specific backend, or leave it at `auto`. Restart after changes.

With `uv run dsapi serve --webui`, the header shows the actual loaded synthesis
backends and any initialization fallback (details on hover). Before loading,
it shows the preference order. `/runtime` exposes the same status. Vocal
separation independently uses PyTorch MPS on this Mac.
See [measurements and compatibility notes](docs/inference-performance.md).

Chinese lyric alignment defaults to **HubertFA v0.0.7**, using the same
`DSAPI_ONNX_BACKEND` selection (CUDA → WebGPU → CPU) in a persistent worker.
Set `DSAPI_ALIGNER=sofa` and restart to use SOFA instead. Japanese retains the
SOFA multilingual model and its custom mora dictionary. `align_meta.json`
records the aligner, language, model and dictionary for each task, so later
evaluation uses the same engine. The alignment artifact keeps its historical
name `sofa.json` for compatibility; HubertFA results identify their backend and
providers inside it. HubertFA does not provide a SOFA confidence score, so the
report displays “—”. `/runtime` includes the HubertFA worker's actual backend.

English-dominant lyrics are detected automatically and use HubertFA with the
voicebank's English pronunciation dictionary, including when the Chinese aligner
is set to SOFA. Missing English words fail explicitly instead of being omitted.
Multi-vowel words are split into singing units for GAME and score synthesis;
`score.json` retains their pronunciation as `word~0`, `word~1`, etc. This is
experimental: syllable boundaries use a phoneme heuristic, and non-English lines
in English-dominant bilingual songs are skipped. Choose a continuous English
window for comparisons. Mimic retains the aligned phoneme timing and source F0.

Note transcription defaults to **GAME Medium v1.0.3 ONNX**, with 8 D3PM steps,
boundary/presence thresholds of 0.2 and language conditioning. It runs after
alignment, using vowel boundaries and phrase context from HubertFA (or SOFA for
Japanese). GAME sessions stay cached in a worker, using CUDA when available or
CPU otherwise; on this Mac the validated path uses 4 CPU threads. RMVPE runs in
a separate worker and preserves the source F0 for mimic; without its checkpoint,
F0 extraction falls back to Praat. The default path does not load SOME's note
network. Set `DSAPI_NOTE_EXTRACTOR=some` and restart to use the previous extractor.
`notes.json` and `job.json` record the chosen extractor; `/runtime` reports its
configured default. Existing saved tasks keep their original outputs.

Vocal separation uses `audio-separator 0.48.0` with **FP32 + regional
`torch.compile`** by default (NumPy 2). Both separator models stay cached so
later requests reuse their compiled blocks. The first request can take longer
while graphs compile. Set `DSAPI_SEPARATOR_COMPILE=0` and restart to disable
compilation for diagnosis; precision remains FP32. Unsupported compilation or a
compile failure is logged by the engine and falls back to eager inference.
MDXC overlap is explicitly set to one window, preserving the previous 8-second
stride for the bundled BS-RoFormer and MelBand karaoke models.

## Synthesis defaults

Acoustic uses 20 sampling steps, pitch uses 10, and variance uses 20.
Acoustic depth follows the imported voicebank's `max_depth`.
Duration prediction and the vocoder have no sampling-step setting.
These defaults apply to both synthesis modes; mimic uses aligned source timing
and extracted pitch directly, so it skips duration and pitch prediction.

## Layout

- `src/dsapi/voicebank.py` – ONNX inference for OpenUtau-format DiffSinger voicebanks
- `src/dsapi/score.py` – score model → DiffSinger word/phoneme/note inputs
- `src/dsapi/extract.py` – HubertFA or SOFA alignment + notes/F0 → phrases
- `src/dsapi/jianpu.py` – tempo, key, quantization, 简谱 text
- `src/dsapi/runners/` – persistent GAME, RMVPE, legacy SOME, HubertFA and SOFA workers

All inference models load on first use and stay resident until server shutdown.
The main process caches both separators and the DiffSinger/vocoder sessions;
separate child processes cache GAME, RMVPE and each alignment engine on demand. SOFA
retains each language checkpoint after it is used, while dictionaries and
alignment inputs remain per request. `/runtime` reports worker PIDs and model
cache counts. Requests to the same worker are serialized. Failed requests are
reported without discarding healthy models; crashed workers restart on their
next request. `DSAPI_WORKER_TIMEOUT` sets the per-request timeout in seconds
(default 600). Restart the server to pick up model-file changes or release caches.

## Imported assets

Supply compatible synthesis assets locally and follow their accompanying terms.

## Web UI (optional)

Off by default. Enable with `uv run dsapi serve --webui`, `DSAPI_WEBUI=1`, or
`DSAPI_WEBUI=1` in a local `.env` (gitignored), then open
http://127.0.0.1:8765/ui/. It submits tasks, polls progress, plays
clip / lead / mimic / score side by side, shows the 简谱 and alignment report,
and can load a task's `score.json` into an editor and re-render it.
The task list's delete button removes a completed or failed task, including its
audio, score, logs and uploaded files, after confirmation. Queued and running
tasks cannot be deleted. The same operation is available through authenticated
`DELETE /tasks/{task_id}`.

## Music sources

`DSAPI_SOURCE` selects where songs, LRC and audio come from (set it in `.env`):

| Value | Needs | Notes |
|---|---|---|
| `musicbox` (default) | nothing | [枫叶音乐盒](https://mu-jie.cc/musicBox/#/search); default platform `netease`, optional `DSAPI_MUSICBOX_SERVER=kuwo` or `qq` |
| `netease` | nothing | direct public NetEase endpoints; VIP/region-locked tracks fail |
| `ncm-api` | `DSAPI_SOURCE_URL`, optional `DSAPI_NCM_COOKIE` | self-hosted [api-enhanced](https://github.com/NeteaseCloudMusicApiEnhanced/api-enhanced) (`docker run -p 3000:3000 moefurina/ncm-api`); signed requests, login cookie unlocks VIP tracks |
| `meting` | `DSAPI_SOURCE_URL`, `DSAPI_METING_SERVER` | self-hosted Meting API; `netease`, `tencent`, `kugou`, `kuwo`, … |

The default MusicBox API is `https://fy-musicbox-api.mu-jie.cc`. Search uses
`/{platform}/search/song/`; song details, LRC and audio use `/meting/` with
the site's required request headers. `DSAPI_SOURCE_URL` can override the API
root (do not include `/musicBox/#/search` or `/meting/`). Search results supply
the song IDs to use with `/release_task` or `dsapi run`; IDs belong to the
configured platform. The default `netease` platform accepts existing NetEase IDs.

For example, to explicitly select the default in `.env`:

```dotenv
DSAPI_SOURCE=musicbox
DSAPI_MUSICBOX_SERVER=netease
```

Remove a previous `DSAPI_SOURCE_URL` override when switching to the public
MusicBox API. Upstream failures (including HTTP 502), missing lyrics, or
unavailable audio fail the task; there is no automatic switch to another source.
Uploading your own audio + LRC remains available. For `ncm-api` and `meting`,
configure `DSAPI_SOURCE_URL` with your own instance.
