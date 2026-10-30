"""Compare old/new audio-separator on identical local weights and inputs.

Run in separate environments, sequentially (they share one GPU). New 0.48.x
uses overlap count=1 to match 0.30.2's stride=8 seconds for these 8-second models.
Loading is measured separately; one warmup precedes measured forward/full runs.
"""
import argparse
import gc
import importlib.metadata
import json
import logging
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from audio_separator.separator import Separator

MODELS = {
    "vocals": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    "karaoke": "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("job", type=Path)
    p.add_argument("--models", type=Path, default=Path("models/separator"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--precision", choices=["fp32", "native-fp16"], default="fp32")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--full", action="store_true", help="Also time complete per-model separation including I/O")
    p.add_argument("--compile", action="store_true", help="Use new engine regional torch.compile")
    args = p.parse_args()
    if args.repeats < 1:
        p.error("--repeats must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    version = importlib.metadata.version("audio-separator")
    legacy = version == "0.30.2"
    if not legacy and version != "0.48.0":
        p.error("This comparison supports only 0.30.2 and 0.48.0")
    assert torch.backends.mps.is_available(), "MPS required for this benchmark"
    report = {"version": version, "torch": torch.__version__, "numpy": np.__version__,
              "precision": args.precision, "compile": args.compile, "models": {}}
    if not legacy:
        from audio_separator.separator.uvr_lib_v5.device_utils import should_fallback_to_cpu_for_complex_ops
        report["spectral_cpu_fallback"] = should_fallback_to_cpu_for_complex_ops(torch.device("mps"))
    sources = {"vocals": args.job / "clip.wav",
               "karaoke": next((args.job / "sep").glob("*_(Vocals)_*.wav"))}
    for name, filename in MODELS.items():
        out = args.output / name
        out.mkdir()
        options = dict(model_file_dir=str(args.models), output_dir=str(out), output_format="WAV",
                       sample_rate=44100, log_level=logging.WARNING,
                       mdxc_params={"segment_size": 256, "override_model_segment_size": False,
                                    "batch_size": 1, "overlap": 8 if legacy else 1, "pitch_shift": 0})
        if not legacy:
            options.update(use_native_fp16=args.precision == "native-fp16", use_torch_compile=args.compile)
        elif args.precision != "fp32" or args.compile:
            p.error("Native FP16/compile requires the new engine")
        start = time.perf_counter()
        sep = Separator(**options)
        sep.load_model(model_filename=filename)
        torch.mps.synchronize()
        instance = sep.model_instance
        net = instance.model_run
        wav, sr = sf.read(sources[name], dtype="float32", always_2d=True)
        assert sr == 44100 and wav.shape[1] == 2
        cfg = instance.model_data_cfgdict
        assert cfg.audio.hop_length * (cfg.inference.dim_t - 1) == 8 * sr
        item = {"model": filename, "source": str(sources[name]), "input_seconds": len(wav) / sr,
                "load_seconds": time.perf_counter() - start, "device": str(next(net.parameters()).device),
                "effective_precision": getattr(instance, "effective_precision", "fp32"),
                "overlap": instance.overlap, "forward": [], "full": []}
        report["models"][name] = item
        print("MODEL", name, json.dumps(item), flush=True)
        x = torch.from_numpy(wav[:8 * sr].T.copy()).to("mps")
        for mode in (["forward", "full"] if args.full else ["forward"]):
            for rep in range(args.repeats + 1):
                torch.mps.synchronize()
                start = time.perf_counter()
                with torch.inference_mode():
                    if mode == "forward":
                        y = net(x.unsqueeze(0)) if legacy else instance._run_roformer_model(x)
                    else:
                        paths = sep.separate(str(sources[name]))
                torch.mps.synchronize()
                elapsed = time.perf_counter() - start
                if mode == "forward":
                    arr = y.detach().float().cpu().numpy().reshape(-1, y.shape[-1]).T
                    sf.write(out / "forward.wav", arr, sr, subtype="FLOAT")
                else:
                    vocal = next(path for path in paths if "_(Vocals)" in path)
                    arr, rate = sf.read(out / Path(vocal).name, dtype="float32", always_2d=True)
                    assert rate == sr and arr.shape == wav.shape
                    sf.write(out / "full.wav", arr, sr, subtype="FLOAT")
                assert np.isfinite(arr).all() and np.sqrt(np.mean(arr**2)) > 0
                run = {"warmup": rep == 0, "seconds": elapsed, "samples": len(arr),
                       "rms": float(np.sqrt(np.mean(arr**2))), "peak": float(np.max(np.abs(arr)))}
                item[mode].append(run)
                item["effective_compile"] = getattr(instance, "effective_torch_compile", False)
                print("RESULT", name, mode, json.dumps(run), flush=True)
                (args.output / "results.json").write_text(json.dumps(report, indent=2))
        del instance, net, sep, x, y
        gc.collect()
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
