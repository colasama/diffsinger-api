# dsapi

Re-sing a lyric-selected excerpt with a DiffSinger voicebank: fetch the song,
randomly pick a continuous LRC window, isolate the lead vocal, align lyrics, transcribe notes,
then render with ImportedFixed (用户固定音源). No LLM anywhere in the pipeline.

```
Song id ─▶ audio + LRC ─▶ lyric window + language tags ─▶ crop ─▶ BS-RoFormer + karaoke
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
uv run dsapi run 3414700455                     # random continuous lyric window
uv run dsapi run 3413072220 --seed 7            # reproducible English/Japanese excerpt
uv run dsapi run 3414700455 --selection chorus  # highest-scoring window
uv run dsapi run 3414700455 --lyric-query '迎着风向前走' # start at a matching lyric
uv run dsapi run 3414700455 --start 174 --end 195
uv run dsapi serve                              # http://127.0.0.1:8765/docs
```

Automatic selection evaluates all whole-line LRC windows, aiming for 25 seconds
(up to 32 seconds before clip padding). It favors continuous lyric coverage,
avoids estimated lyric gaps over 5 seconds, and uses repetition as a small chorus
bonus. By default it randomly chooses among good windows (within 0.5 points of
the best score), retaining the best ending per candidate start. Set
`selection=chorus` to choose the single highest-scoring window. Songs without repeated lyrics use the same scoring instead of taking the
first four lines. Instrumental labels are excluded, duplicate timestamps are
collapsed, and lyrics outside the audio duration are ignored. Selection uses
lyrics only: incorrect LRC timing or a mismatched audio version can still select
the wrong passage. Use `--start` and `--end` to select a lyric range manually.
Supply both times or neither. Whole lyric lines are retained within the 40-second
audio limit; the crop adds up to 1.5 seconds before and 1 second after them.
`job.json` saves the generated random seed, exact lyric/audio windows, selected
lines and their languages. Reuse the seed with the same lyrics to reproduce the
selection. Explicit start/end overrides automatic selection.

Pass `lyric_query` to prefer a window beginning at that lyric's LRC line.
The selector normalizes full-width characters, case, punctuation and whitespace,
preserves Latin word boundaries and mixed-language character order, and ranks
whole-line matches above prefixes and contained fragments. Short fragments require
a whole-line match. A query may span up to four adjacent lines with gaps of at most
five seconds; the entire match must fit in the window. Repeated matches use window
quality, then the earliest occurrence, to break ties. Near the end of a song, a
short excerpt is allowed without prepending earlier lines. Matching within a line
starts singing at the beginning of that whole line, not halfway through it.
Explicit start/end still wins. An unmatched/unusable query falls back to the
requested `random` or `chorus` strategy; selection metadata records the lyric
query, match text/times/kind, or fallback reason. These matches rely on LRC timing.

`GET /resolve_song?q=...` resolves a title, a title followed by an artist, or lyrics
against the first ten distinct valid search candidates. Exact titles take priority,
then complete title + artist matches (including multiword names and individual
credited artists), then verified lyric matches, then the first search result.
The response includes `song`, `match_type` (`title`, `title_artist`, `lyric`, or
`fallback`), and `lyric_query` (only set for a lyric match). Submit its `song.id`
and `lyric_query` to `/release_task`; the job rechecks lyrics against the actual
audio duration. Empty search results return 404. Existing `/search` is unchanged.
Lyric checks use four workers, an eight-second wait budget after search, and a
bounded five-minute cache; individual candidate failures do not abort resolution.
Successful resolution depends on the upstream search returning the intended song;
this is not a separate full-catalog lyric search.

LRC accepts integer seconds and fractional seconds separated by `.`, `,`, or
`:` (for example `[00:12.95]`, `[00:12,950]`, `[00:12:95]`). Timestamp whitespace,
full-width digits/punctuation, BOM and CRLF are accepted. Hour timestamps must
include decimal seconds (`[01:02:03.500]`); three colon-only fields always mean
minutes, seconds and a fraction. Repeated leading stamps repeat a lyric line.
Enhanced LRC `<mm:ss.xx>` word markers are stripped before phoneme extraction;
a trailing empty marker bounds the line, while HubertFA still aligns the words.
`[offset:+500]` advances all timestamps by 500 ms (negative values delay them);
the last valid offset applies globally, and out-of-audio timestamps are removed.
Metadata tags are excluded; colons in actual lyric text are preserved. Untimed
plain text and encrypted karaoke formats are not assigned guessed timestamps.

Artifacts land in `work/<task_id>/`: `clip.mp3`, `lead.mp3`, `mimic.mp3`,
`score.mp3`, `jianpu.txt`, `score.json`, `job.json` (per-stage timings and
alignment report), and `error.txt` on failure.

## API

| Method | Path | Notes |
|---|---|---|
| POST | `/release_task` | `{"song_id", "start"?, "end"?, "modes"?, "selection"?: "random" or "chorus", "seed"?: integer, "lyric_query"?: string}` → `{"task_id"}` |
| GET | `/resolve_song?q=` | Resolve title, title + artist, or lyrics; returns song and optional lyric anchor |
| GET | `/query_result?task_id=` | status, current stage, timings, 简谱, output URLs |
| GET | `/v1/audio?task_id=&kind=` | `mimic` / `score` / `lead` / `clip` |
| POST | `/render_score` | body is `score.json`'s shape; synthesis only |

Set `DSAPI_KEY` to require `Authorization: Bearer <key>`.
`/release_task_upload` accepts the same `selection`, `seed`, and `lyric_query` fields as multipart
form fields. Existing clients that omit them use random lyric selection.

### Multiple users, shared queue

Set `DSAPI_KEYS` to a JSON object mapping stable user IDs to unique secret keys.
For example, in `.env` (replace the example secrets before use):

```dotenv
DSAPI_KEYS={"alice":"replace-with-alice-secret","bob":"replace-with-bob-secret"}
```

Generate each secret with `python -c 'import secrets; print(secrets.token_urlsafe(32))'`
and distribute it to its user. Restart the server after changing the configuration.
Clients continue sending `Authorization: Bearer <key>`; the web UI also supports
`?key=<key>` for audio playback. If both are present, the header takes precedence.

Each user sees only their own tasks in `/tasks`, and can only query, download,
evaluate, or delete their own tasks. Other users' task IDs return 404. Uploads
follow the same rules. All users submit to the **same single-worker FIFO queue**;
there is no separate execution queue per user. Run one server process/worker to
retain that global queue. `/render_score` remains a synchronous synthesis endpoint,
outside the task queue.

Task ownership is saved as a user ID, never as an API key, and survives restarts.
To rotate a user's key, replace the secret while keeping their user ID unchanged.
Sharing a key means sharing one task history. There is no administrator bypass.

`DSAPI_KEY` remains supported alongside `DSAPI_KEYS`: it accesses the legacy task
history (including old tasks without an owner), but cannot access named users'
tasks. Named users cannot access legacy tasks. `__legacy__` is a reserved user ID.
With neither setting configured, authentication is disabled and only the legacy
history is accessible. Invalid or duplicate key configuration prevents startup.

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

Chinese, English and Japanese lyric alignment default to **HubertFA v0.0.7**, using the same
`DSAPI_ONNX_BACKEND` selection (CUDA → WebGPU → CPU) in a persistent worker.
Set `DSAPI_ALIGNER=sofa` for Chinese or `DSAPI_JA_ALIGNER=sofa` for Japanese
and restart to opt into the legacy aligner for monolingual clips. English and
mixed-language clips always use HubertFA. Japanese no longer needs the
`DSAPI_EXPERIMENTAL_JA` switch. This uses the voicebank's Japanese
phonemes directly; the checkpoint's language-neutral `cl` token represents
closure. Missing units or unsupported phonemes in English/Japanese or mixed
HubertFA inputs fail explicitly. `align_meta.json`
records the aligner, language, model and dictionary for each task, so later
evaluation uses the same engine. The alignment artifact keeps its historical
name `sofa.json` for compatibility; HubertFA results identify their backend and
providers inside it. HubertFA does not provide a SOFA confidence score, so the
report displays “—”. `/runtime` includes the HubertFA worker's actual backend.

To compare on exactly the same saved Japanese clip, separated vocals and F0:

```sh
uv run python scripts/compare_japanese_alignment.py work/<completed-japanese-task> \
  --task-id japanese-hubertfa-comparison --aligner hubertfa
```

The new task includes both synthesis modes and `comparison.json` with input
hashes, timings and aligned boundaries. It never overwrites the reference task.
These diagnostics are not a substitute for listening or hand-labelled timing.

Language routing follows each lyric line and its word/script runs, including
English words embedded in Japanese or Chinese lines. It does not discard Japanese
lines from English-dominant songs. Mixed clips use one alignment over the whole
waveform with language-qualified dictionary keys and phonemes, so LRC timestamps
do not force an audio cut at each language change. Neighboring context lyrics keep
their language tags and are excluded from synthesis. `segments.json` preserves
the unit-to-language mapping; phrases split at language changes so GAME receives
the correct language for each region. Han-only lines use the song's CJK context
to choose Mandarin versus Japanese; this is a text heuristic, not audio language
recognition. At duplicate timestamps the first usable lyric is kept, so put the
original lyric before a translation in uploaded LRC files.

Multi-vowel words are split into singing units for GAME and score synthesis;
`score.json` retains their pronunciation as `word~0`, `word~1`, etc. This is
experimental: English syllable boundaries use a phoneme heuristic.
Mimic retains the aligned phoneme timing and source F0.

Note transcription defaults to **GAME Medium v1.0.3 ONNX**, with 8 D3PM steps,
boundary/presence thresholds of 0.2 and language conditioning. It runs after
alignment, using vowel boundaries and phrase context from HubertFA (or an explicit
SOFA override). GAME sessions stay cached in a worker, using CUDA when available or
CPU otherwise; on this Mac the validated path uses 4 CPU threads. RMVPE runs in
a separate worker and preserves the source F0 for mimic; without its checkpoint,
F0 extraction falls back to Praat. The default path does not load SOME's note
network. Set `DSAPI_NOTE_EXTRACTOR=some` and restart to use the previous extractor.
`notes.json` and `job.json` record the chosen extractor; `/runtime` reports its
configured default. Existing saved tasks keep their original outputs.

Vocal separation defaults to BS-RoFormer `ep_368`
(`model_bs_roformer_ep_368_sdr_12.9628.ckpt`), followed by the MelBand karaoke
model for lead vocal extraction. Override the first model with
`DSAPI_VOCALS_MODEL` and restart the server; missing weights are downloaded on
first use.

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
