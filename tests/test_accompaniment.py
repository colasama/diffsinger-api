from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from dsapi import audio, pipeline


def stems(tmp_path, monkeypatch, seconds=12, factor=2, stereo=True):
    sr = 8000
    monkeypatch.setattr(audio.settings, 'SAMPLE_RATE', sr)
    t = np.arange(sr * seconds) / sr
    ref = .1 * np.sin(2*np.pi*220*t)
    ref[(t < 1) | ((t > 4) & (t < 8))] = 0
    bed = np.column_stack([.2*np.sin(2*np.pi*330*t), .15*np.sin(2*np.pi*550*t)])
    paths = [tmp_path/name for name in ('vocal.wav', 'inst.wav', 'ref.wav')]
    for p, data in zip(paths, (ref*factor, bed if stereo else bed[:, 0], ref)):
        sf.write(p, data, sr, subtype='FLOAT')
    return paths, ref, bed


@pytest.mark.parametrize('seconds', [12, 65])
def test_mix_preserves_stereo_time_and_gaps_without_clipping(tmp_path, monkeypatch, seconds):
    paths, ref, bed = stems(tmp_path, monkeypatch, seconds)
    out = tmp_path/'mix.wav'
    report = audio.mix_accompaniment(*paths, out)
    mixed, sr = sf.read(out)
    assert sr == 8000 and mixed.shape == bed.shape
    assert report['vocal_gain_db'] == pytest.approx(-6.0206, abs=.001)
    assert np.isfinite(mixed).all() and np.max(np.abs(mixed)) <= 10**(-1.4/20)
    expected = bed + ref[:, None]
    assert np.corrcoef(mixed.ravel(), expected.ravel())[0, 1] > .99
    # Instrumental gaps remain stereo with music, and no vocal shift occurs.
    assert np.max(np.abs(mixed[8000*5:8000*7])) > .01
    assert np.max(np.abs(mixed[:, 0]-mixed[:, 1])) > .01
    assert not list(tmp_path.glob('mix-*'))


def test_mix_bounds_gain_and_accepts_mono_accompaniment(tmp_path, monkeypatch):
    paths, ref, _ = stems(tmp_path, monkeypatch, factor=.1, stereo=False)
    report = audio.mix_accompaniment(*paths, tmp_path/'mix.wav')
    assert report['vocal_gain_db'] == 12
    mixed, _ = sf.read(tmp_path/'mix.wav')
    np.testing.assert_array_equal(mixed[:, 0], mixed[:, 1])


def test_mix_rejects_truncated_stems_and_silent_reference(tmp_path, monkeypatch):
    paths, ref, bed = stems(tmp_path, monkeypatch)
    sf.write(paths[1], bed[:-8000], 8000, subtype='FLOAT')
    with pytest.raises(ValueError, match='duration'):
        audio.mix_accompaniment(*paths, tmp_path/'mix.wav')
    sf.write(paths[1], bed, 8000, subtype='FLOAT')
    sf.write(paths[2], np.zeros_like(ref), 8000, subtype='FLOAT')
    with pytest.raises(ValueError, match='No overlapping'):
        audio.mix_accompaniment(*paths, tmp_path/'mix.wav')


@pytest.mark.parametrize('mode', ['mimic', 'score'])
@pytest.mark.parametrize('enabled', [False, True])
def test_pipeline_exports_requested_mix_without_rerender(tmp_path, monkeypatch, mode, enabled):
    paths, ref, _ = stems(tmp_path, monkeypatch)
    job = pipeline.Job.__new__(pipeline.Job)
    job.dir = tmp_path
    job.state = pipeline.JobState('test', pipeline.JobSpec(with_accompaniment=enabled))
    calls = []
    def encode(src, dst, loudnorm=True):
        calls.append((dst.stem, loudnorm))
        dst.write_bytes(b'ID3test')
        return dst
    monkeypatch.setattr(audio, 'to_mp3', encode)
    yielded = []
    def render_once():
        yielded.append(1)
        yield SimpleNamespace(start=0., phones=[('a', 0., 1.)]), ref*2
    job._mix(mode, render_once(), len(ref), paths[1], paths[2])
    assert yielded == [1]
    assert calls == [(mode, True)] + ([(mode+'_mix', False)] if enabled else [])
    assert set(job.state.outputs) == ({mode, mode+'_mix'} if enabled else {mode})
    if enabled:
        assert job.state.mixing[mode]['vocal_gain_db'] == pytest.approx(-6.0206, abs=.001)


def test_mixed_parts_keep_global_normalization_and_cover_timeline(tmp_path, monkeypatch):
    monkeypatch.setattr(audio.settings, 'SAMPLE_RATE', 100)
    job = pipeline.Job.__new__(pipeline.Job)
    job.dir = tmp_path
    job.state = pipeline.JobState('test', pipeline.JobSpec(with_accompaniment=True))
    encodings, spans = [], []
    def encode(src, dst, loudnorm=True):
        encodings.append((dst.stem, loudnorm))
        with dst.open('wb') as f:
            f.truncate(29*1024*1024 if dst.stem == 'score_mix' else 3)
        return dst
    def cut(src, dst, start, end):
        spans.append((start, end)); dst.touch(); return dst
    monkeypatch.setattr(audio, 'to_mp3', encode)
    monkeypatch.setattr(audio, 'cut', cut)
    job._export_audio('score_mix', tmp_path/'source.wav', 120100, [57500, 110000], normalized=True)
    assert spans == [(0, 575), (575, 1100), (1100, 1201)]
    assert all(not normalize for _, normalize in encodings)
    assert set(job.state.outputs) == {'score_mix', 'score_mix_part_0000', 'score_mix_part_0001', 'score_mix_part_0002'}
