# Apple Silicon inference measurements

Measured on 2026-10-31: Apple M1 Pro, 32 GB RAM, Python 3.11,
ONNX Runtime 1.31.1, native WebGPU plugin 0.4.0, PyTorch 2.15.0.
The installed voicebank is ImportedFixed v1.0.0 with the compatible vocoder.

## Enable native WebGPU

```sh
uv run dsapi serve --webui
```

The native WebGPU plugin is installed by default. Automatic selection tries
CUDA, then WebGPU, then CPU if session initialization fails. The WebUI header
reports actual loaded model providers via `/runtime`; before loading, it shows
the selection order. Hover over it for model providers and fallback reasons.
Restart the server after changing backends: sessions are cached.

Backend choices:

- `auto` (default): CUDA → WebGPU → CPU. Initialization failures are logged.
- `cpu`: CPU only, useful for comparison.
- `cuda`: require the CUDA provider.
- `webgpu`: require the native WebGPU plugin; use `DSAPI_WEBGPU_DEVICE` to select
  a zero-based index among the plugin's devices (default `0`). This machine
  reports one WebGPU device, vendor `Apple`.

This is the native ONNX Runtime plugin targeting Metal, not a browser service.
The plugin is registered once. Session creation and inference share a process-wide
lock, so `/render_score` requests cannot run WebGPU sessions concurrently with
background jobs. This sacrifices concurrent throughput to avoid the upstream
native concurrent-Run crash. Explicit WebGPU failures surface as errors. Automatic fallback applies only to
session initialization; inference errors are not retried on another backend.

Two nodes need CPU fallback with the current voicebank/plugin:

```
/diffusion/velocity_fn/Gather
/pitch_predictor/velocity_fn/Gather
```

Without `forceCpuNodeNames`, the acoustic/pitch loops fail with
`All inputs must be tensors on WebGPU buffers`. These specific nodes stay on
CPU while the surrounding networks use WebGPU. An acoustic-model ORT profile
recorded 14,172 WebGPU node events and 372 CPU node events over three runs;
these counts are evidence of GPU execution, not a FLOP or time breakdown.
Other voicebanks and export versions need their own compatibility checks.

## Synthesis results

Source: first phrase of local job `work/43a265123ad6`. Mimic uses the saved
phoneme alignment and source F0; score uses its quantized score. Each backend
runs in a separate process. One initial iteration is excluded, then two warm
iterations are averaged. Sampling is explicitly fixed at acoustic 20, pitch 10,
variance 20 steps on both backends. No reduction in sampling steps is used.

| Mode | Output length | CPU | WebGPU | Speedup |
|---|---:|---:|---:|---:|
| mimic | 6.69 s | 4.01 s | 1.92 s | 2.08× |
| score | 7.28 s | 4.66 s | 2.43 s | 1.92× |

These timings include the neural synthesis chain, not downloading, separation,
note extraction, alignment, MP3 encoding, or queue time. They are a small
single-machine sample, not a general speed guarantee. GPU/compiler caches had
already been exercised during diagnosis; first-run latency is not characterized.

Both outputs have the expected length, finite samples, nonzero RMS and peaks
below clipping. For the last repetition, CPU/WebGPU waveform relative RMSE is
0.0353% (mimic) and 0.0287% (score). Praat voiced/unvoiced agreement is 100% on
these samples, with 95th-percentile F0 differences below 0.002 cents. These
signal checks do not replace listening or broader voicebank/phrase testing.

Reproduce with new output directories:

```sh
uv run python scripts/benchmark_inference.py work/43a265123ad6 \
  --backend cpu --output work/bench-cpu --repeats 2
uv run python scripts/benchmark_inference.py work/43a265123ad6 \
  --backend webgpu --output work/bench-webgpu --repeats 2
```

The script records all model providers, stage times and sampling steps, and
saves WAVs. The selected score phrase must match one unambiguous aligned source
phrase. It refuses to overwrite an existing output directory.

Local final artifacts:

- `work/webgpu-benchmark/final-cpu/results.json`
- `work/webgpu-benchmark/final-webgpu/results.json`
- `work/webgpu-benchmark/summary.json`
- `work/webgpu-benchmark/final-webgpu/mimic-2.wav`
- `work/webgpu-benchmark/final-webgpu/score-2.wav`

## Vocal extraction: measured opportunities

The original `audio-separator` 0.30.2 already ran both RoFormer networks on
MPS and cached both models. Simply enabling GPU or adding model caching again
will not improve this path.

An 8-second stereo chunk was tested with the current models, one warmup and
two measured runs per precision mode. Timing brackets the network forward pass
with `torch.mps.synchronize()`; model loading and file I/O are excluded.
The karaoke input is the saved first-stage vocal stem.

| Model | FP32 | MPS autocast | Difference |
|---|---:|---:|---:|
| BS-RoFormer Viperx 1297 | 5.03 s | 4.90 s | about 2.7% faster |
| MelBand Karaoke aufr33/viperx | 2.23 s | 2.24 s | essentially unchanged |

Autocast is not a meaningful speed win on this setup and was not enabled in
the application. Raw output relative RMSE versus FP32 was below 0.05% for both
models. The probe and results are in `work/separation-benchmark/`.

Other opportunities and upgrade constraints:

1. **Reduce CPU/MPS round trips.** The old RoFormer implementations move
   STFT/mask/ISTFT work to CPU, and overlap-add also runs on CPU. New upstream
   implementations keep supported spectral work and bounded buffers on MPS and
   offer additional precision/compilation options. The isolated comparison below
   found almost no FP32 speed improvement without compilation.
   `audio-separator` 0.48.0 requires `numpy>=2`, so upgrading also required
   validating the other pipeline components after removing the previous NumPy 1 pin.
2. **Cache extraction outputs.** Repeated work on the same audio/window/model
   can reuse separation, F0 and alignment results. Include audio content,
   window, model/version and settings in cache keys.
3. **Make karaoke conditional.** `DSAPI_KARAOKE_MODEL=` skips the second stage;
   a prior 34.14-second job spent about 11 seconds in that separation call.
   This changes preprocessing and can hurt pitch extraction on harmony-heavy
   songs. It is not a quality-neutral optimization.

Do not blindly lower `overlap` or increase `batch_size`: this installed
RoFormer path ignores batch size, and its `overlap` parameter is used as a
stride in seconds. The current stride and model chunk are both 8 seconds;
reducing it to 4 would increase computation. Increasing stride beyond the
chunk can leave unprocessed gaps.

## Isolated audio-separator 0.48.0 experiment

The initial baseline used 0.30.2 / NumPy 1.26.4. An independent environment at
`work/separator-new/.venv` installs 0.48.0 / NumPy 2.4.6; both environments use
PyTorch 2.15.0 and exactly the same local model checkpoints. The new runtime's
complex spectral capability probe passes on this M1 Pro, enabling MPS spectral
operations. The initial experiment kept production unchanged; the application
was subsequently upgraded to the FP32 + compile configuration described below.

`scripts/benchmark_separation.py` measures an 8-second forward pass and complete
22.4-second file separation, including export/cleanup but excluding model load.
Each uses one warmup and two measured runs. Both stages receive identical inputs
across versions; karaoke always receives the saved original first-stage vocals,
so this isolates engine changes rather than propagating first-stage differences.

For these two 8-second models, **old `overlap=8` (stride seconds) must become
new `overlap=1` (window count)** to preserve the chunk schedule. Both versions
process three chunks starting at 0, 8, and 14.4 seconds. New defaults use the
YAML's `num_overlap=4`, which would increase work and change blending; it is not
a matched speed comparison.

| Complete per-model separation | Old FP32 | New FP32 | New native FP16 | New FP16 + compile | New FP32 + compile |
|---|---:|---:|---:|---:|---:|
| BS-RoFormer vocals | 15.65 s | 15.58 s | 15.08 s | 11.58 s | **11.09 s** |
| MelBand karaoke | 7.25 s | 7.15 s | 7.02 s | 4.75 s | **4.77 s** |
| Sum of fixed-input stages | 22.90 s | 22.73 s | 22.10 s | 16.33 s | **15.87 s** |

**Prefer new FP32 + regional compilation for further integration testing.**
It retains full precision while reducing this workload's time by 30.7%
(1.44× throughput). Both models report `effective_compile=true`; no compilation
failure/eager fallback was observed. This is a single clip and two warm measured
runs, not a broad quality or performance guarantee. The sum is not a full
end-to-end job: downloading, transcription, alignment and singing are excluded.

Upgrading alone changes the workload by under 1%, within ordinary run variability.
Native FP16 alone saves about 3.5%. The main gain comes from regional compilation,
not the spectral CPU/MPS transfer change or half precision.

| Full-output relative RMSE versus old FP32 | Vocals | Karaoke |
|---|---:|---:|
| New FP32 | 0.00043% | 0.01425% |
| New native FP16 | 0.04015% | 0.54199% |
| New FP16 + compile | 0.03929% | 0.48703% |
| New FP32 + compile | **0.00057%** | **0.01426%** |

All outputs retain their length and contain finite samples with nonzero RMS and
peaks below clipping. Signal checks alone do not establish perceptual quality
or generalize to other recordings. FP16 has no compelling advantage here.

For FP32 + compile, first forward calls in these processes took 6.56 s (vocals)
and 3.94 s (karaoke), versus subsequent means of 3.47 s and 1.39 s. These first
calls exclude model loading and are not guaranteed fresh compiler-cache results;
previous experiments had exercised the compiler. Fresh caches, new shapes or
other models can add more latency. Keep model instances warm for repeated work.

Reproduce in isolated environments while other GPU work is idle:

```sh
uv venv work/separator-old/.venv --python .venv/bin/python
uv pip install --python work/separator-old/.venv/bin/python \
  'audio-separator[cpu]==0.30.2' 'numpy<2' 'torch==2.15.0'
uv venv work/separator-new/.venv --python .venv/bin/python
uv pip install --python work/separator-new/.venv/bin/python \
  'audio-separator[cpu]==0.48.0' 'torch==2.15.0'
work/separator-old/.venv/bin/python scripts/benchmark_separation.py work/43a265123ad6 \
  --output work/sep-old-repeat --full
work/separator-new/.venv/bin/python scripts/benchmark_separation.py work/43a265123ad6 \
  --output work/sep-new-repeat --full
work/separator-new/.venv/bin/python scripts/benchmark_separation.py work/43a265123ad6 \
  --output work/sep-new-compiled-repeat --compile --full
```

Add `--compile`, `--precision native-fp16` or both only for the new environment.
Reports record effective precision/compilation, warmup,
model loading, measured runs and output statistics. New output directories are
required. Exact isolated dependencies are saved in
`work/separator-new/requirements.txt`, and comparison data/WAVs under
`work/separator-new/`. Two interrupted FP16 trials with concurrent GPU work
are marked `INVALID_TIMING.txt` and excluded from comparisons.

## Application upgrade

The main environment now pins `audio-separator==0.48.0` and permits `numpy>=2,<3`
(the Python 3.11 lock resolves NumPy 2.4.6). Separation explicitly disables
autocast/native FP16, enables regional compilation by default, and sets MDXC
`overlap=1`. Existing per-model caching preserves compiled blocks across jobs.
`DSAPI_SEPARATOR_COMPILE=0` disables compilation for diagnosis; restart after
changing it. The engine logs unsupported or failed compilation and uses eager
FP32 inference in that case. The synthesis backend remains independent:
CUDA → WebGPU → CPU.

Upgrade validation passed 40 project tests, `uv pip check` and `uv lock --check`.
Two complete local-input Mandarin jobs passed under NumPy 2, including both
separation stages, SOME/RMVPE, SOFA, quantization, mimic and score. The second
job reused the same separator objects and compiled blocks; both models reported
FP32, `effective_torch_compile=true`, MPS and overlap 1. All floating model
parameters remained float32. Output WAVs/F0 contained finite samples, and both
rendered modes had nonzero audio. Multilingual SOFA also aligned a Japanese clip
successfully in the same persistent worker.

The actual two-stage separation call took 30.09 s in the first job (including
model loading/compilation) and 15.91 s in the second. These use cascading vocals,
unlike the fixed-input per-stage benchmark above. Results and logs are saved in
`work/separator-upgrade/validation.json` and `validation.log`; generated audio
is under `work/separator-upgrade/run-0/` and `run-1/`. The local managed service
is restarted after the environment upgrade so its imports and cached models
use the new dependencies.

## Validation and references

### Persistent-model memory (2026-10-31)

SOME/RMVPE and SOFA now run in two persistent workers. Models load on first
use and remain cached; SOFA caches Mandarin and multilingual checkpoints
separately. Dictionaries and audio inputs are still request-specific. The
main process continues to retain both separators and all used ONNX sessions.
Shutdown drains background jobs and closes the workers. Worker failures are
reported to the current request; the next request starts a replacement.

Measured with `scripts/benchmark_memory.py` on the M1 Pro / 32 GB machine,
audio-separator 0.30.2 (MPS), SOME/RMVPE and SOFA on CPU, and all eight synthesis
sessions on native WebGPU. The local upload from `work/43a265123ad6` was run
through the complete pipeline twice (22.40 seconds of audio, both mimic and
score outputs). After each run, `work/ja1` was aligned with multilingual SOFA.

The metric is the sum of macOS `proc_pid_rusage` physical footprints for the
host and its two model workers, sampled every 100 ms. This includes GPU and
compressed-memory accounting; RSS alone substantially undercounts this setup.
Transient ffmpeg processes are excluded. Models and allocator caches were not
explicitly cleared. Another local GPU benchmark overlapped part of this run,
so timings are diagnostic only and these are process-attributed memory samples,
not measurements on an otherwise idle machine or guaranteed capacity limits.

| Measurement | Physical footprint |
|---|---:|
| Application imports, before model use | 0.24 GiB |
| After first complete Mandarin job | 5.15 GiB |
| After also loading multilingual SOFA | 5.83 GiB |
| After second complete job and multilingual alignment, then 5 s idle | **6.54 GiB** |
| Highest simultaneous sample, during second separation | **7.98 GiB** |

Final idle breakdown: host (both separators, DiffSinger, vocoder and runtime)
3.15 GiB; SOME + RMVPE worker 2.03 GiB; SOFA worker with both checkpoints
1.37 GiB. These are whole-process figures, not isolated model weights. The
difference between first and second idle readings includes retained runtime
allocations; two runs do not establish a long-run plateau.

Both worker PIDs stayed unchanged. SOME's model bundle loaded once and SOFA
loaded exactly two checkpoints; subsequent calls hit those caches. The two
runs produced identical notes, F0/UV arrays, alignment and score JSON. All four
synthesized WAVs were 22.40 seconds, finite and non-silent. Sample timings for
notes/F0 were 8.64 s initially and 2.95 s warm; Mandarin alignment was 6.26 s
initially and 0.23 s warm (not a controlled speed comparison).

Reproduce using a new output directory:

```sh
uv run python scripts/benchmark_memory.py work/43a265123ad6 \
  --multilingual-job work/ja1 --backend webgpu \
  --output work/resident-memory-new
```

Raw samples, stage snapshots and worker cache counters:
`work/resident-memory-20261031/results.json`. Output validation:
`work/resident-memory-20261031/validation.json`.

`python -m pytest -q tests` passed project tests, including provider
selection, explicit failure on unavailable backends and cross-session locking.
An unrestricted `pytest` scan also collected SOFA's vendored metric tests;
four of those failed in `vendor/SOFA/modules/utils/metrics_test.py`, outside
the changed inference code. No changes were made to that vendor code.

- [OpenUtau native WebGPU integration](https://github.com/openutau/OpenUtau/pull/2553)
- [ONNX Runtime native WebGPU provider/options](https://onnxruntime.ai/docs/execution-providers/WebGPU-ExecutionProvider.html)
- [Microsoft Python plugin package](https://pypi.org/project/onnxruntime-ep-webgpu/0.4.0/)
- [audio-separator upstream acceleration options](https://github.com/nomadkaraoke/python-audio-separator)
- [audio-separator 0.48.0 dependency metadata](https://pypi.org/pypi/audio-separator/0.48.0/json)
