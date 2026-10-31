import json
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from dsapi import pipeline, settings, sources
from dsapi.pipeline import Job, JobSpec, _score_json
from dsapi.jianpu import quantize, render as render_jianpu
from dsapi.score import Note, Phrase, Syllable


@pytest.fixture
def environment(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, 'WORK', tmp_path / 'work')
    monkeypatch.setattr(settings, 'CACHE_ENABLED', True)
    monkeypatch.setattr(settings, 'CACHE_MAX_BYTES', 10**7)
    for name in ('SEPARATOR_MODELS', 'GAME_MODEL', 'RMVPE_CKPT', 'VENDOR'):
        monkeypatch.setattr(settings, name, tmp_path / name)
    calls = Counter()
    input_wav = tmp_path / 'input.wav'
    sf.write(input_wav, np.sin(np.arange(44100) * .05).astype(np.float32) * .1, 44100)
    ckpt, dictionary = tmp_path / 'align/model.onnx', tmp_path / 'dictionary.txt'
    ckpt.parent.mkdir(); ckpt.write_bytes(b'model'); dictionary.write_text('ni\ti\n')

    class Source:
        def song_info(self, song_id):
            return sources.Song(song_id, 'test', [])

        def lyric(self, song_id):
            return '[00:00.1]你\n[00:00.8]好'

        def download_audio(self, song_id, destination):
            calls['fetch'] += 1
            shutil.copyfile(input_wav, destination)
            return destination

    monkeypatch.setattr(pipeline.sources, 'get_source', Source)
    monkeypatch.setattr(pipeline, 'voicebank', lambda *a: object())

    def separate(clip, out):
        calls['separate'] += 1
        out.mkdir(parents=True)
        shutil.copyfile(clip, out / 'lead.wav')
        shutil.copyfile(clip, out / 'original_inst.wav')
        return out / 'lead.wav', out / 'original_inst.wav'

    def segments(lines, lead, out, vb, **kwargs):
        out.mkdir(parents=True)
        sf.write(out / 'clip.wav', lead, 44100)
        (out / 'clip.lab').write_text('ni')
        return [{'name': 'clip', 'ckpt': str(ckpt), 'dictionary': str(dictionary),
                 'backend': 'hubertfa', 'lang': 'zh'}]

    def align(folder, out, *a, **kwargs):
        calls['align'] += 1
        value = {'clip': {'length': 1.}}
        (out / 'sofa.json').write_text(json.dumps(value))
        return value

    def f0(wav, out):
        calls['f0'] += 1
        data = np.stack([np.full(87, 261.6), np.zeros(87)])
        np.save(out / 'f0.npy', data)
        return data[0], data[1].astype(bool)

    def build(*args):
        return [Phrase([Syllable('你', 'ni', [Note(.1, .5, 60)])],
                       phones=[('i', .1, .6)], f0_midi=np.full(44, 60.))], []

    def game(wav, out, *args):
        calls['game'] += 1
        (out / 'notes.json').write_text(json.dumps({'backend': 'game-medium', 'notes': []}))
        (out / 'game_regions.json').write_text('[]')
        return []

    def mix(self, mode, rendered, *args):
        for phrase, wav in rendered:
            assert len(wav)
        calls['render'] += 1
        target = self.dir / f'{mode}.mp3'; target.write_bytes(b'ID3')
        self.state.outputs[mode] = str(target)

    monkeypatch.setattr(pipeline.audio, 'separate_long_vocals', separate)
    monkeypatch.setattr(pipeline.extract, 'write_segments', segments)
    monkeypatch.setattr(pipeline.tools, 'align', align)
    monkeypatch.setattr(pipeline.tools, 'extract_f0', f0)
    monkeypatch.setattr(pipeline.tools, 'extract_game_notes', game)
    monkeypatch.setattr(pipeline.extract, 'build_phrases', build)
    monkeypatch.setattr(pipeline, 'render_mimic', lambda *a: np.ones(100))
    monkeypatch.setattr(pipeline, '_score_json', lambda *a: {'phrases': []})
    monkeypatch.setattr(pipeline.jianpu, 'estimate_grid', lambda *a: pipeline.jianpu.Grid(120, 0))
    monkeypatch.setattr(pipeline.jianpu, 'estimate_tonic', lambda *a: 0)
    monkeypatch.setattr(pipeline.jianpu, 'quantize', lambda phrases, grid: phrases)
    monkeypatch.setattr(pipeline.jianpu, 'render', lambda *a: 'test score')
    monkeypatch.setattr(Job, '_mix', mix)
    return calls, dictionary


def run(name, **kwargs):
    owner = kwargs.pop('owner', 'alice')
    job = Job(name, JobSpec(song_id='1', start=0, end=1, modes=['mimic'], **kwargs), owner_id=owner)
    result = job.run()
    assert result.status == 'done', result.error
    return job


def test_repeat_skips_analysis_but_renders_new_voice_and_seed(environment):
    calls, dictionary = environment
    first = run('first', synthesis_seed=1)
    second = run('second', synthesis_seed=2, voice_id='imported-voice-g')
    assert not any(first.state.cache_hits.values())
    assert all(second.state.cache_hits.values())
    assert calls == {'fetch': 1, 'separate': 1, 'align': 1, 'f0': 1, 'game': 1, 'render': 2}
    assert (second.dir / 'score.json').is_file()
    shutil.rmtree(first.dir)
    assert (second.dir / 'sep/lead.wav').is_file()
    dictionary.write_text('ni\ta\n')
    third = run('third')
    assert not third.state.cache_hits['align']
    assert third.state.cache_hits['separate']
    other = run('other', owner='bob')
    assert not any(other.state.cache_hits.values())


def test_audio_only_skips_game_and_score_but_keeps_mimic_and_references(environment):
    calls, _ = environment
    fast = run('fast', audio_only=True)
    assert calls['game'] == 0 and calls['render'] == 1
    assert fast.state.note_extraction == {'backend': 'skipped', 'reason': 'audio_only'}
    assert fast.state.jianpu is None and fast.state.grid is None
    assert not (fast.dir / 'score.json').exists() and not (fast.dir / 'notes.json').exists()
    assert set(fast.state.outputs) == {'mimic', 'lead', 'clip'}
    complete = run('complete')
    assert complete.state.cache_hits['f0'] and calls['game'] == 1


def test_prefetch_completes_before_inference_without_changing_job_progress(environment, monkeypatch):
    monkeypatch.setattr(settings, 'PREFETCH_DOWNLOADS', True)
    job = Job('prefetch', JobSpec(song_id='1', start=0, end=1, modes=['mimic']))
    with ThreadPoolExecutor(max_workers=1) as executor:
        job.prefetch(executor)
        prepared = job._source_future.result(timeout=10)
        assert Path(prepared['source']).is_file()
        assert job.state.status == 'queued' and job.state.song is None
        assert job.run().status == 'done'
    assert environment[0]['fetch'] == 1


def test_prefetch_error_is_reported_as_job_failure(environment, monkeypatch):
    monkeypatch.setattr(settings, 'PREFETCH_DOWNLOADS', True)
    job = Job('bad', JobSpec(song_id='1', modes=['mimic']))
    def fail():
        raise RuntimeError('download failed')
    monkeypatch.setattr(job, '_fetch_source', fail)
    with ThreadPoolExecutor(max_workers=1) as executor:
        job.prefetch(executor)
        assert job.run().status == 'failed'
    assert 'download failed' in job.state.error


def test_simultaneous_same_song_prefetch_downloads_only_once(environment, monkeypatch):
    monkeypatch.setattr(settings, 'PREFETCH_DOWNLOADS', True)
    jobs = [Job(f'pre-{i}', JobSpec(song_id='1', modes=['mimic']), owner_id='alice') for i in range(2)]
    with ThreadPoolExecutor(max_workers=2) as executor:
        for job in jobs:
            job.prefetch(executor)
        results = [job._source_future.result(timeout=10) for job in jobs]
    assert environment[0]['fetch'] == 1
    assert sorted(r['cache_hit'] for r in results) == [False, True]
    assert all(job.state.status == 'queued' for job in jobs)


@pytest.mark.parametrize('modes', [['score'], ['mimic', 'score']])
def test_score_keeps_extracted_timing_while_notation_is_quantized(environment, monkeypatch, modes):
    phrases = [Phrase([
        Syllable('你', 'ni', [Note(.11, .08, 60), Note(.19, .12, 62)]),
        Syllable('好', 'hao', [Note(.36, .21, 64)]),
    ], phones=[('i', .11, .31), ('SP', .31, .36), ('ao', .36, .57)],
        f0_midi=np.full(50, 60.))]
    grid = pipeline.jianpu.Grid(120, 0)
    expected = _score_json(phrases, grid)
    notation = quantize(phrases, grid)
    assert _score_json(notation, grid) != expected
    seen = {}

    def render(mode, phrase):
        seen[mode] = _score_json([phrase], grid)
        return np.ones(100)

    monkeypatch.setattr(pipeline.extract, 'build_phrases', lambda *a: (phrases, []))
    monkeypatch.setattr(pipeline.jianpu, 'quantize', quantize)
    monkeypatch.setattr(pipeline.jianpu, 'render', render_jianpu)
    monkeypatch.setattr(pipeline, '_score_json', _score_json)
    monkeypatch.setattr(pipeline, 'render_score', lambda vb, p: render('score', p))
    monkeypatch.setattr(pipeline, 'render_mimic', lambda vb, p: render('mimic', p))
    job = Job('timing', JobSpec(song_id='1', start=0, end=1, modes=modes))
    result = job.run()
    assert result.status == 'done', result.error
    assert json.loads((job.dir / 'score.json').read_text()) == expected
    assert seen == {mode: expected for mode in modes}
    assert _score_json(phrases, grid) == expected
    assert (job.dir / 'jianpu.txt').read_text() == render_jianpu(notation, grid, 0)
