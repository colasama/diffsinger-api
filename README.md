# dsapi

Re-sing a song's chorus with a DiffSinger voicebank: fetch the song, pick the
chorus from its LRC, isolate the lead vocal, transcribe notes, align lyrics,
then render with ImportedFixed (用户固定音源). No LLM anywhere in the pipeline.

```
NetEase id ─▶ audio + LRC ─▶ chorus window ─▶ BS-RoFormer + karaoke (lead vocal)
          ─▶ SOME notes + RMVPE f0 ─▶ SOFA alignment per LRC line
          ─▶ phrases ─┬─▶ mimic: real phoneme timing + source f0 ─▶ variance/acoustic/vocoder
                      └─▶ score: grid-quantized notes ─▶ dur/pitch/variance/acoustic/vocoder
          ─▶ 简谱 text + score.json
```

## Setup

```sh
uv sync
scripts/fetch_models.sh   # ~1.8 GB of weights; separator models download on first run (~1 GB)
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

## Layout

- `src/dsapi/voicebank.py` – ONNX inference for OpenUtau-format DiffSinger voicebanks
- `src/dsapi/score.py` – score model → DiffSinger word/phoneme/note inputs
- `src/dsapi/extract.py` – SOFA/SOME/RMVPE output → phrases
- `src/dsapi/jianpu.py` – tempo, key, quantization, 简谱 text
- `src/dsapi/runners/` – SOME and SOFA run as subprocesses (their top-level packages clash)

## Imported assets

Supply compatible synthesis assets locally and follow their accompanying terms.

## Web UI (optional)

Off by default. Enable with `uv run dsapi serve --webui`, `DSAPI_WEBUI=1`, or
`DSAPI_WEBUI=1` in a local `.env` (gitignored), then open
http://127.0.0.1:8765/ui/. It submits tasks, polls progress, plays
clip / lead / mimic / score side by side, shows the 简谱 and alignment report,
and can load a task's `score.json` into an editor and re-render it.

## Music sources

`DSAPI_SOURCE` selects where songs, LRC and audio come from (set it in `.env`):

| Value | Needs | Notes |
|---|---|---|
| `netease` (default) | nothing | direct public NetEase endpoints; VIP/region-locked tracks fail |
| `ncm-api` | `DSAPI_SOURCE_URL`, optional `DSAPI_NCM_COOKIE` | self-hosted [api-enhanced](https://github.com/NeteaseCloudMusicApiEnhanced/api-enhanced) (`docker run -p 3000:3000 moefurina/ncm-api`); signed requests, login cookie unlocks VIP tracks |
| `meting` | `DSAPI_SOURCE_URL`, `DSAPI_METING_SERVER` | self-hosted Meting API; `netease`, `tencent`, `kugou`, `kuwo`, … |

Point these at instances you run yourself; third-party public deployments are
not meant to be called from servers.
