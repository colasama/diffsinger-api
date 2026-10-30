"""Detect sustained near-silence introduced by lead-vocal separation.

This is a conservative dropout guard, not a classifier of lead/backing vocals.
Both inputs must retain the separator's sample alignment and channel layout.
"""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ProtectionConfig:
    frame_seconds: float = 0.02
    input_floor_db: float = -45.0
    output_ceiling_db: float = -55.0
    attenuation_db: float = 20.0
    min_duration_seconds: float = 0.25
    max_gap_seconds: float = 0.10


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def detect_lead_dropouts(vocals: np.ndarray, lead: np.ndarray, sample_rate: int,
                         config: ProtectionConfig = ProtectionConfig()) -> dict:
    """Flag a whole-clip fallback if any sustained severe dropout is found.

    Arrays are (samples, channels). RMS pools channel *energy*, avoiding false
    silence caused by stereo phase cancellation. Times in the report are
    relative to the clip. No independent loudness normalization is applied.
    """
    if vocals.ndim != 2 or vocals.shape != lead.shape or vocals.shape[1] == 0:
        raise ValueError("Vocal protection requires matching (samples, channels) arrays")
    if sample_rate <= 0 or not np.isfinite(vocals).all() or not np.isfinite(lead).all():
        raise ValueError("Vocal protection requires a positive sample rate and finite audio")
    values = asdict(config)
    if (not all(np.isfinite(v) for v in values.values()) or config.frame_seconds <= 0
            or config.min_duration_seconds <= 0 or config.attenuation_db <= 0
            or config.max_gap_seconds < 0):
        raise ValueError("Invalid vocal protection thresholds")
    n = len(vocals)
    report = {"enabled": True, "mode": "whole_clip", "sample_rate": sample_rate,
              "parameters": values, "detected_intervals": [], "fallback": False,
              "fallback_seconds": 0.0, "reason": "no_sustained_dropout"}
    if not n:
        return report
    frame = max(1, round(config.frame_seconds * sample_rate))
    starts = np.arange(0, n, frame)
    lengths = np.minimum(frame, n - starts)

    def levels(wav):
        power = np.mean(np.square(wav, dtype=np.float64), axis=1)
        mean_power = np.add.reduceat(power, starts) / lengths
        return 10 * np.log10(np.maximum(mean_power, 1e-20))

    before, after = levels(vocals), levels(lead)
    suspect = ((before >= config.input_floor_db) & (after <= config.output_ceiling_db)
               & (before - after >= config.attenuation_db))
    joined = suspect.copy()
    # Bridge short holes within a dropout; do not extend to unrelated silence.
    runs = _runs(suspect)
    for (_, end), (start, _) in zip(runs, runs[1:]):
        if (start - end) * frame <= config.max_gap_seconds * sample_rate:
            joined[end:start] = True
    for first, last in _runs(joined):
        # Count actual suspect samples, not bridged gaps or a padded last frame.
        suspect_samples = int(lengths[first:last][suspect[first:last]].sum())
        if suspect_samples < config.min_duration_seconds * sample_rate:
            continue
        start, end = first * frame, min(last * frame, n)
        report["detected_intervals"].append({
            "start": start / sample_rate, "end": end / sample_rate,
            "suspect_seconds": suspect_samples / sample_rate,
            "max_attenuation_db": float(np.max((before - after)[first:last])),
        })
    if report["detected_intervals"]:
        report.update(fallback=True, fallback_seconds=n / sample_rate, reason="sustained_dropout")
    return report
