"""Protect missing syllables without undoing ordinary separation or quiet gaps."""

import numpy as np
import pytest

from dsapi.vocal_protection import detect_lead_dropouts

RATE = 8000


def tone(seconds=3):
    t = np.arange(round(seconds * RATE)) / RATE
    mono = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return np.column_stack([mono, -mono])  # phase cancellation is not silence


def test_any_sustained_dropout_flags_entire_clip_including_unaffected_sections():
    vocals = tone()
    lead = vocals * 0.5
    lead[RATE:2 * RATE] = 0
    report = detect_lead_dropouts(vocals, lead, RATE)
    assert report['fallback'] is True
    assert report['mode'] == 'whole_clip'
    assert report['fallback_seconds'] == 3
    assert report['detected_intervals'][0]['start'] == 1
    assert report['detected_intervals'][0]['end'] == 2


@pytest.mark.parametrize('kind', ['silence', 'quiet', 'short', 'ordinary', 'large_but_audible'])
def test_does_not_restore_normal_or_insufficient_evidence(kind):
    vocals = tone()
    lead = vocals.copy()
    if kind == 'silence':
        vocals[:] = 0
        lead[:] = 0
    elif kind == 'quiet':
        vocals *= 0.001
        lead[:] = 0
    elif kind == 'short':
        lead[RATE:RATE + RATE // 10] = 0
    elif kind == 'ordinary':
        lead *= 0.25
    else:
        lead *= 0.05  # >20 dB reduction, but not near-silence
    report = detect_lead_dropouts(vocals, lead, RATE)
    assert report['fallback'] is False


def test_bridges_brief_surviving_spikes_without_counting_gaps_as_dropout():
    vocals = tone()
    lead = vocals.copy()
    lead[RATE:RATE + int(.18 * RATE)] = 0
    lead[RATE + int(.22 * RATE):RATE + int(.40 * RATE)] = 0
    report = detect_lead_dropouts(vocals, lead, RATE)
    assert report['fallback'] is True
    assert len(report['detected_intervals']) == 1
    lead = vocals.copy()
    lead[RATE:RATE + int(.10 * RATE)] = 0
    lead[RATE + int(.20 * RATE):RATE + int(.30 * RATE)] = 0
    report = detect_lead_dropouts(vocals, lead, RATE)
    assert report['detected_intervals'] == []


@pytest.mark.parametrize('where', ['start', 'end', 'all'])
def test_dropouts_at_clip_edges_and_partial_last_frame(where):
    vocals = tone(1.013)
    lead = vocals.copy()
    region = {'start': slice(None, RATE // 2), 'end': slice(-RATE // 2, None), 'all': slice(None)}[where]
    lead[region] = 0
    report = detect_lead_dropouts(vocals, lead, RATE)
    assert report['fallback'] is True
    assert report['fallback_seconds'] == len(vocals) / RATE
    assert report['detected_intervals'][-1]['end'] <= len(vocals) / RATE


def test_rejects_misalignment_and_nonfinite_audio():
    vocals = tone()
    with pytest.raises(ValueError, match='matching'):
        detect_lead_dropouts(vocals, vocals[:-1], RATE)
    lead = vocals.copy()
    lead[0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        detect_lead_dropouts(vocals, lead, RATE)


def test_short_clip_does_not_count_zero_padding_as_evidence():
    vocals = tone(.249)
    report = detect_lead_dropouts(vocals, np.zeros_like(vocals), RATE)
    assert report['fallback'] is False
    assert report['detected_intervals'] == []
