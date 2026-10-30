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

The existing `audio-separator` 0.30.2 already runs both RoFormer networks on
MPS and caches both models. Simply enabling GPU or adding model caching again
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

Further opportunities, not yet performance-validated here:

1. **Reduce CPU/MPS round trips.** The installed RoFormer implementations move
   STFT/mask/ISTFT work to CPU, and overlap-add also runs on CPU. New upstream
   implementations keep supported spectral work and bounded buffers on MPS and
   offer additional precision/compilation options. Test an isolated upgrade or
   backport. Current PyPI `audio-separator` 0.48.0 requires `numpy>=2`, whereas
   this application pins `numpy<2`; upgrading is not a drop-in dependency change.
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

## Validation and references

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
