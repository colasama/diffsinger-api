# dsapi

Re-sing a lyric-selected excerpt with a DiffSinger voicebank: fetch the song,
randomly pick a continuous LRC window, isolate the lead vocal, align lyrics, transcribe notes,
then render using a locally imported synthesis model. No LLM anywhere in the pipeline.

```
Song id ─▶ audio + LRC ─▶ lyric window + language tags ─▶ crop ─▶ BS-RoFormer + karaoke
          ─▶ HubertFA alignment ─▶ GAME Medium notes + RMVPE f0
          ─▶ phrases ─┬─▶ mimic: real phoneme timing + source f0 ─▶ variance/acoustic/vocoder
                      └─▶ score: original note timing ─▶ dur/pitch/variance/acoustic/vocoder
          ─▶ quantized 简谱 text + original-timing score.json
```

## Setup

```sh
uv sync
scripts/fetch_models.sh   # includes HubertFA v0.0.7 and GAME Medium v1.0.3
```

## Import local synthesis assets

No synthesis models, speaker presets or vocoders are included or downloaded by
this project. Import your own compatible OpenUtau-format DiffSinger ONNX export
before submitting synthesis jobs. `scripts/fetch_models.sh` installs analysis
tools and their weights only.

For one model, place its files in `models/voicebank/` or set `DSAPI_VOICEBANK`
to its directory. Keep the acoustic files, `dsdur`, `dspitch`, `dsvariance` and
language dictionaries together. Put a compatible vocoder in `models/vocoder/`
or set `DSAPI_VOCODER`; this directory must contain `vocoder.yaml` and its model.
The current adapter requires 44.1 kHz audio, a 512-sample hop, all four synthesis
stages, and `dsdict-zh.yaml`, `dsdict-ja.yaml`, `dsdict-en.yaml` under `dsdur/`.
The generic API ID is `default`; it remains unavailable until the files validate.

For multiple models or speaker embeddings, create a local JSON file (for example
`models/voices.json`, ignored by Git) and set `DSAPI_VOICES_FILE` to its path:

```json
{
  "default_voice_id": "my-voice",
  "voices": [
    {"id": "my-voice", "name": "My imported voice", "path": "./voicebank", "aliases": ["my"]}
  ]
}
```

Paths in this file are relative to the JSON file. Optional `speaker` selects an
embedding by its suffix in each stage's `speakers` configuration; omit it for a
fixed-speaker export. Optional `model_version` is your own metadata. Restart after
changing configuration. Invalid catalog configuration prevents startup.

`GET /voices` lists configured IDs, names, aliases and availability without local
paths. `voice_id` is accepted by `/release_task`, `/release_task_upload`,
`/render_score` and CLI `--voice`. Missing model files return 503; unknown IDs
return 422. Historical jobs without saved voice metadata display an unknown
source rather than attributing them to a configured model.

## Complete lyric excerpts

`lyric_extent=auto` (default) preserves short-window selection for one or two
lyric lines. Three or more complete matched LRC lines, or three verified supplied
sentences, select the entire requested continuous span. Newlines and sentence-end
punctuation can delimit supplied sentences; Latin word spaces do not. Full
selection bypasses the 25-second preference, 32/40-second limits, four-line
matching limit and five-second gap restriction. It neither pads with extra lyric
lines nor truncates a long request. Partial first/last lines include those full
LRC lines. Explicit ranges use LRC timestamp ends, not estimated syllable lengths.
Timing accuracy still depends on the source LRC matching the recording.

`lyric_extent=window` keeps the previous short-window behavior; `exact` requires
full lyric matching even for one line. Manual `start/end` wins. Explicit multiline
lyrics that cannot be completely matched fail instead of selecting an unrelated
random excerpt. Text fields accept up to 16,384 characters. Use
`POST /resolve_song` with `{"q":"first line\nsecond line\nthird line","lyric_extent":"auto"}`
for long text; the GET route remains available. Candidate searches use excerpts,
but validate every supplied lyric token against the candidate's continuous LRC.

Long audio is separated in overlapping chunks, alignment is grouped at lyric
boundaries, F0 extraction is chunked, and synthesis proceeds phrase by phrase.
Workers default to a duration-scaled timeout (at least 600 seconds);
`DSAPI_WORKER_TIMEOUT` overrides it. Full MP3s are retained. If a rendered file
exceeds 28 MiB, additional `<mode>_part_0000` outputs provide bounded downloads,
with cuts at phrase ends where possible. API clients use the normal authenticated
`/v1/audio` route for these kinds.

Retake models share sessions across speakers and use per-request stage-specific
embeddings and noise. `synthesis_seed` / `--synthesis-seed` controls that noise and
is saved separately from the lyric-window `seed`. CPU, CUDA and WebGPU use the
same adapter; the actual backend is reported by `/runtime`.

Local smoke test (configured imports × score/mimic; writes `work/voice-smoke/`):

```sh
DSAPI_ONNX_BACKEND=cpu uv run python scripts/smoke_voices.py
uv run dsapi run 3414700455 --voice default --lyric-query '歌词内容'
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
Supply both times or neither. Explicit ranges bypass the 32/40-second limits. Whole lyric lines are retained; the crop adds up to 1.5 seconds before and 1 second after them.
`job.json` saves the generated random seed, exact lyric/audio windows, selected
lines and their languages. Reuse the seed with the same lyrics to reproduce the
selection. Explicit start/end overrides automatic selection.

Pass `lyric_query` to prefer a window beginning at that lyric's LRC line.
The selector normalizes full-width characters, case, punctuation and whitespace,
preserves Latin word boundaries and mixed-language character order, and ranks
whole-line matches above prefixes and contained fragments. Short fragments require
a whole-line match. In short-window mode, a query may span up to four adjacent lines with gaps of at most
five seconds; the entire match must fit in the window. Complete excerpts use the rules above. Repeated matches use window
quality, then the earliest occurrence, to break ties. Near the end of a song, a
short excerpt is allowed without prepending earlier lines. Matching within a line
starts singing at the beginning of that whole line, not halfway through it.
Explicit start/end still wins. An unmatched/unusable short query falls back to the
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
a separate worker and decodes each voiced run with Viterbi (confidence threshold
0.03). Runs shorter than 50 ms retain local decoding, and unvoiced gaps keep
their original mask. This applies to both GAME and legacy SOME paths; F0
interpolation and timing remain unchanged. Without the RMVPE checkpoint,
F0 extraction falls back to Praat. The default path does not load SOME's note
network. Set `DSAPI_NOTE_EXTRACTOR=some` and restart to use the previous extractor.
`notes.json` and `job.json` record the chosen extractor; `/runtime` reports its
configured default. Existing saved tasks keep their original outputs.

Long continuous phrases split at meaningful pauses, balancing equal gaps instead
of repeatedly isolating individual syllables. GAME receives at least 100 ms of
analysis context for short regions (right zero-padding only when the whole input
is too short). Returned notes are clipped to the original region's sample-aligned
start/end, so analysis context cannot lengthen or shift the score.

Repeated jobs reuse downloaded songs, separated stems, alignment, F0 and GAME
notes from an owner-isolated cache under `work/.cache`. Content hashes identify
audio/lyrics; model file size/mtime/ctime, relevant parameters, runtime versions
and pipeline code invalidate analysis entries. Voice and synthesis seed changes
still run synthesis; voice-specific alignment inputs are checked before reuse.
Cached artifacts are verified and copied into each job, so deleting an earlier
job cannot break a later result. Deleting a task also clears that owner's shared
cache. Invalid entries are recomputed. Model-file changes still require a server
restart because loaded neural sessions remain resident.

Defaults: `DSAPI_CACHE=1`, `DSAPI_CACHE_TTL=86400` (24 hours),
`DSAPI_CACHE_MAX_BYTES=5368709120` (5 GiB across owners). Expired entries are not
reused; expiration and least-recently-used capacity cleanup run on cache writes.
Set `DSAPI_CACHE=0` to bypass artifact reuse. The first run must populate the
cache; historical task directories are not automatically imported.

`DSAPI_PREFETCH=1` allows at most two queued song downloads concurrently with
the single inference worker; metadata and lyrics are fetched alongside audio.
Repeated concurrent downloads of the same owner's song are coalesced. Uploaded
files use the normal queue. `job.json`/`query_result` include `cache_hits` and
`metrics` (queue time, fetch work time, F0, GAME, model and reference encoding
time). Fetch work can happen while queued; its duration is not added to the
job's elapsed time a second time. `/runtime` reports optimization settings.

For a faster mimic-only result, send `"modes":["mimic"], "audio_only":true`
to either task endpoint (multipart uses `modes=mimic`), use the Web UI's
“仅模仿音频” checkbox, or run `dsapi run SONG_ID --modes mimic --audio-only`.
This skips GAME and score quantization while preserving aligned phonemes and
RMVPE + Viterbi F0. It produces mimic/lead/clip audio and optional accompaniment,
but no `score.json` or jianpu; note extraction is explicitly marked `skipped`.
The default remains `audio_only=false`, preserving all existing score outputs.

Vocal separation defaults to BS-RoFormer `ep_368`
(`model_bs_roformer_ep_368_sdr_12.9628.ckpt`), followed by the MelBand karaoke
model for lead vocal extraction. Override the first model with
`DSAPI_VOCALS_MODEL` and restart the server; missing weights are downloaded on
first use.

Second-stage dropout protection is enabled by default. If first-stage vocals are
above -45 dBFS while the second-stage output is below -55 dBFS and at least 20 dB
quieter for at least 250 ms of suspect audio, the **entire clip** uses first-stage
vocals for alignment, pitch/note extraction, and the lead preview. Detection uses
20 ms stereo-energy frames and joins gaps up to 100 ms; it does not splice or mix
the two outputs. `sep/lead_protection.json` records the decision, thresholds,
clip-relative anomaly intervals, and selected file. Raw stems stay available.
This guard can also restore backing vocals; it prioritizes retaining the singer
when a severe dropout is detected. Set `DSAPI_KARAOKE_PROTECTION=0` and restart
to disable the guard, or `DSAPI_KARAOKE_MODEL=` to bypass the second stage entirely.
Existing saved tasks are not rewritten.

Vocal separation uses `audio-separator 0.48.0` with **FP32 + regional
`torch.compile`** by default (NumPy 2). Both separator models stay cached so
later requests reuse their compiled blocks. The first request can take longer
while graphs compile. Set `DSAPI_SEPARATOR_COMPILE=0` and restart to disable
compilation for diagnosis; precision remains FP32. Unsupported compilation or a
compile failure is logged by the engine and falls back to eager inference.
MDXC overlap is explicitly set to one window, preserving the previous 8-second
stride for the bundled BS-RoFormer and MelBand karaoke models.

## Synthesis defaults

Score synthesis and `score.json` retain the extracted note timing, including
off-grid onsets, short notes and rests. Only the displayed/exported 简谱 is
quantized to the notation grid. `score.json` times are seconds relative to the
clip; its BPM is descriptive and does not rescale synthesis. `/render_score`
uses those same times, including for previously saved quantized scores.
Timing remains subject to extraction accuracy and the synthesis frame resolution
(512 samples at 44.1 kHz, about 11.6 ms). Score mode still predicts phoneme
durations and pitch expression; mimic retains source phoneme timing and F0.
Existing task outputs are not rewritten.

Acoustic uses 20 sampling steps, pitch uses 10, and variance uses 20.
Acoustic depth follows the imported model's `max_depth`.
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

## Web UI (optional)

Off by default. Enable with `uv run dsapi serve --webui`, `DSAPI_WEBUI=1`, or
`DSAPI_WEBUI=1` in a local `.env` (gitignored), then open
http://127.0.0.1:8765/ui/. It submits tasks, polls progress, plays
clip / lead / mimic / score side by side, shows the 简谱 and alignment report,
and can load a task's `score.json` into an editor and re-render it.
The song composer accepts a song name, lyrics, a song ID, or uploaded audio and LRC.
Voice selectors show singer aliases; four presets select score/mimic vocals with
or without accompaniment. Explicit time ranges and three or more complete lyric
lines can exceed the normal short-clip limit.
The custom-score composer includes fixed alternating notation/lyric lines,
loadable examples, an expandable syntax guide, and validation before submission.
Bar lines are optional. Completed custom-score tasks can restore their original
text to the composer for editing. Both composers use a muted rose theme.
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

### Accompaniment mixes

`POST /release_task` and multipart `/release_task_upload` accept
`with_accompaniment` (default `false`). For each selected mode, enabling it retains
the dry `score` / `mimic` output and adds `score_mix` / `mimic_mix`. Request these
through `/v1/audio?task_id=…&kind=score_mix` (or `mimic_mix`). CLI runs accept
`--with-accompaniment`; the Web UI also has a mix checkbox. Existing tasks and
callers remain dry-only by default. `/render_score` has no source accompaniment
and retains its current behavior.

Mixing reuses the instrumental stem already separated from the exact source
clip; it never performs another voice inference. The mono synthesized voice is
centered over the stereo instrumental. Active 50 ms vocal frames set the voice
gain relative to the original lead, bounded to ±12 dB; silence is excluded.
Stems are processed in blocks, sample rates are aligned, and duration mismatches
over 50 ms fail instead of silently dropping music. The float mix is normalized
once to -16 LUFS / -1.5 dBTP before MP3 encoding. Long outputs use
`score_mix_part_0000` / `mimic_mix_part_0000` etc., retaining the complete mix's
normalization across parts. `job.json.mixing` records gain and frame adjustment.
Separated accompaniment can retain some original vocal leakage.

### 自填简谱和歌词

Gokohime 输入 `.score` 会发送一张谱子格式说明图片，包含演唱示例和符号说明。最简输入：

```text
.score
1 2 3 5
今天天晴
```

命令首行只填写可选歌手，支持 `.voices` 中的别名；省略歌手时使用用户配置的默认音源。
未知歌手会报错，不会将乐谱作为歌曲搜索。后续非空行严格交替为简谱、歌词，
无需“谱/词”前缀。可以连续追加多对行；换行不额外增加停顿。

首个内容行可填写参数，例如 `1=C4 BPM=100 4/4 LANG=zh`；各项均可省略，
默认 `1=C4 BPM=120 4/4 LANG=zh`。BPM 为每分钟四分音符数。
`LANG=ja` 按日语音拍拆词，`LANG=en` 按音源词典中的元音音节拆词。
网页的“检查谱词”和 `POST /parse_score` 会显示实际拆分结果。

- `1`–`7`：音阶，每音默认一拍；`0`：休止，不占歌词。
- `1'` / `1,`：升高 / 降低八度；`#4` / `b7`：升 / 降半音。
- `5:2` / `3:0.5` / `3:1.5`：两拍 / 半拍 / 一拍半，所有时值以四分音符为单位。
- `(2:0.5 3:0.5)`：一字多音，括号内至少两个音符，不允许休止。
- 中文每字一个演唱音节，忽略空格与一般标点；多音字可写 `行(xing)`。
- `|` 分隔小节；写了小节线就校验每小节总拍数。歌词可对应写 `|`；不写则按整行匹配。
- 纯休止行的歌词填写 `-`，因为空行会忽略。

示例（每小节四拍）：

```text
.score
1=C4 BPM=100 4/4
1 (2:0.5 3:0.5) 5 0 | 6:2 5:2
晴 天 好 | 向 前
```

不受搜歌的 32 秒片段限制，长谱按短乐句合成后按原时间轴拼接；超过发送大小时沿用分段发送。
为控制单次模型任务的资源，输入最多 16384 字符、8192 个音符（含休止符）、30 分钟（含首尾静音），
单字及其连音最多 60 秒，单音时值为 0.0625–64 拍，BPM 为 20–400。
自填乐谱生成干声，不提供原唱对比或原曲伴奏。

API 两个入口都接收 `{"text":"1 2 3 5\n今天天晴","voice_id":"default"}`：

- `POST /parse_score`：返回 `phrases`、`rows`（歌词拆分）、`duration_seconds` 和节拍参数；不会创建任务。
- `POST /release_score_task`：服务端重新校验并返回 `task_id`；支持可选 `synthesis_seed`，共用现有队列、鉴权、查询、音频下载及删除接口。

格式错误返回 HTTP 422，`detail` 包含 `line`、`column`、`message`；行号从内容第一行开始，不含 `.score` 命令行。
现有 `/render_score` JSON 接口继续保留；网页“自填乐谱”提供固定行输入和可展开的 JSON 编辑器。
