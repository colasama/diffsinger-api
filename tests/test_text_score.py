from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from dsapi import api, score_task, settings
from dsapi.text_score import ScoreError, parse_score


class Bank:
    vowels = {'a', 'e', 'i'}

    def g2p(self, value, lang='zh'):
        if value in {'unknown', 'invalid'}:
            raise KeyError(value)
        if lang == 'en' and value == 'hello':
            return ['h', 'e', 'l', 'a']
        return ['a']

    def is_vowel(self, phone):
        return phone in self.vowels


def parse(text):
    return parse_score(text, Bank())


def syllables(parsed):
    return [s for p in parsed.phrases for s in p.syllables]


def test_defaults_and_contiguous_lines():
    p = parse('1 2\n今天\n\n3 5\n天晴')
    assert (p.bpm, p.key, p.meter, p.language) == (120, 'C4', '4/4', 'zh')
    ss = syllables(p)
    assert [s.notes[0].midi for s in ss] == [60, 62, 64, 67]
    assert [s.start for s in ss] == [.5, 1, 1.5, 2]
    assert p.duration_seconds == pytest.approx(2.8)
    assert p.rows[1]['lyric_line'] == 5


def test_rests_melisma_accidentals_and_octaves():
    p = parse("1=D4 BPM=60\n0 1, (#4:0.5 5':1.5) b7 0:2\n你好吗")
    ss = syllables(p)
    assert len(ss) == 3
    assert ss[0].start == 1.5
    assert [n.midi for s in ss for n in s.notes] == [50, 68, 81, 72]
    assert [(n.start, n.dur) for n in ss[1].notes] == [(2.5, .5), (3, 1.5)]
    assert p.duration_seconds == pytest.approx(7.8)


def test_bars_and_pronunciation():
    p = parse('1=C4 BPM=100 4/4\n1 2 3 5 | 6:2 5:2\n今 天 风 很 | 温 暖\n3 2 1:2\n向 前 行(xing)')
    assert syllables(p)[-1].pinyin == 'xing'
    assert p.rows[0]['units'][0]['pinyin'] == 'jin'
    assert parse('6/8\n1 2 3 | 3 2 1\n今天天晴天晴').duration_seconds == pytest.approx(3.8)


def test_language_units():
    en = parse('LANG=en\n1 (2 3)\nhello')
    assert [s.pinyin for s in syllables(en)] == ['hello~0', 'hello~1']
    ja = parse('LANG=ja\n1 2 3\nさくら')
    assert [s.pinyin for s in syllables(ja)] == ['sa', 'ku', 'ra']


def test_long_score_keeps_absolute_timeline():
    p = parse('1:16\n啦\n' * 12)
    assert p.duration_seconds == pytest.approx(96.8)
    assert len(p.phrases) > 1
    assert [s.start for s in syllables(p)] == [.5+8*i for i in range(12)]
    assert parse('0:2\n-\n1\n啦\n0:2\n-').duration_seconds == pytest.approx(3.3)


@pytest.mark.parametrize('text,line,fragment', [
    ('1 2\n天', 2, '需要 2'),
    ('1 2', 1, '缺少'),
    ('1:0\n天', 1, '时值'),
    ('1:nan\n天', 1, '无效音符'),
    ('1 | 2\n天晴', 1, '小节'),
    ('1 2 3 4 | 1 2 3 4\n天天 | 天天天天天天', 2, '小节字数'),
    ('(1 0)\n天', 1, '休止符'),
    ('((1 2))\n天', 1, '括号'),
    ('(1 2\n天', 1, '括号'),
    ('0\n-', 1, '至少'),
    ('BPM=0\n1\n天', 1, 'BPM'),
    ('BPM=120 BPM=100\n1\n天', 1, '重复'),
    ('1\n天(invalid)', 2, '不支持'),
    ('1\nhello', 2, '无法识别'),
    ('LANG=en\n1\nunknown', 3, '不支持'),
    ('LANG=ja\n1\nゕか', 3, '假名'),
    ('LANG=ja\n1\n龘天', 3, '日语读音'),
    ('1\n天(xing', 2, '注音'),
    ('1\n😀', 2, '无法识别'),
    ('1:64\n啦\n' * 60, 113, '最多'),
])
def test_errors_are_localized(text, line, fragment):
    with pytest.raises(ScoreError) as exc:
        parse(text)
    assert exc.value.detail['line'] == line
    assert fragment in str(exc.value)


@pytest.fixture
def client(monkeypatch, tmp_path, voice_files):
    monkeypatch.setattr(settings, 'WORK', tmp_path)
    monkeypatch.setattr(api, 'API_KEY', '')
    monkeypatch.setattr(api, 'API_KEYS', {'alice': 'a', 'bob': 'b'})
    monkeypatch.setattr(api, '_jobs', {})
    monkeypatch.setattr(api, '_job_futures', {})
    monkeypatch.setattr(api, '_executor', SimpleNamespace(submit=lambda _: None))
    monkeypatch.setattr(api, 'voicebank', lambda *args: Bank())
    return TestClient(api.app, headers={'Authorization': 'Bearer a'})


def test_preview_validation_and_ownership(client, monkeypatch):
    preview = client.post('/parse_score', json={'text': '1 2\n天晴'})
    assert preview.status_code == 200
    assert preview.json()['phrases'][0][1]['pinyin'] == 'qing'
    assert not api._jobs
    invalid = client.post('/release_score_task', json={'text': '1 2\n天'})
    assert invalid.status_code == 422
    assert invalid.json()['detail']['line'] == 2
    assert not api._jobs
    task = client.post('/release_score_task', json={'text': '1\n天', 'synthesis_seed': 7}).json()['task_id']
    st = client.get('/query_result', params={'task_id': task}).json()
    assert st['spec']['modes'] == ['score']
    assert st['spec']['voice_id'] == 'default'
    assert st['spec']['score_text'] == '1\n天'
    assert st['spec']['synthesis_seed'] == 7
    assert 'owner_id' not in st
    for path in ['/query_result', '/v1/score', '/v1/audio']:
        assert client.get(path, params={'task_id': task}, headers={'Authorization': 'Bearer b'}).status_code == 404
    assert client.delete('/tasks/'+task, headers={'Authorization': 'Bearer b'}).status_code == 404
    assert client.get('/tasks', headers={'Authorization': 'Bearer b'}).json() == []


def test_score_job_runs_without_source_and_preserves_rests(client, monkeypatch):
    text = '0:2 1 0:2 2 0:2\n天晴'
    task = client.post('/release_score_task', json={'text': text}).json()['task_id']
    job = api._jobs[task]
    monkeypatch.setattr(score_task, 'voicebank', lambda *args: Bank())
    # Render impulses at each written note, with silent phrase padding.
    def render(_, phrase):
        result = np.zeros(round((phrase.end-phrase.start)*settings.SAMPLE_RATE), np.float32)
        for s in phrase.syllables:
            result[round((s.start-phrase.start)*settings.SAMPLE_RATE)] = 1
        return result
    monkeypatch.setattr(score_task, 'render_score', render)
    captured = {}
    def export(kind, wav, n, boundaries):
        import soundfile as sf
        captured['audio'], captured['sr'] = sf.read(wav)
        path = job.dir / 'score.mp3'
        path.write_bytes(b'ID3test')
        job.state.outputs[kind] = str(path)
    monkeypatch.setattr(job, '_export_audio', export)
    monkeypatch.setattr(api.sources, 'get_source', lambda: pytest.fail('score task must not fetch a song'))
    assert job.run().status == 'done'
    assert np.flatnonzero(captured['audio']).tolist() == [round(1.5*settings.SAMPLE_RATE), round(3*settings.SAMPLE_RATE)]
    assert len(captured['audio']) == round(4.8*settings.SAMPLE_RATE)
    assert client.get('/v1/score', params={'task_id': task}).json()['duration_seconds'] == pytest.approx(4.8)
    assert client.get('/v1/audio', params={'task_id': task, 'kind': 'score'}).content == b'ID3test'
    assert client.post('/evaluate', params={'task_id': task}).status_code == 422
    # Persisted jobs retain ownership even after leaving the in-memory map.
    api._jobs.clear()
    assert client.get('/query_result', params={'task_id': task}).status_code == 200
    assert client.get('/query_result', params={'task_id': task}, headers={'Authorization': 'Bearer b'}).status_code == 404
    assert client.delete('/tasks/'+task).status_code == 200
