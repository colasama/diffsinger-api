import json
import subprocess

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from dsapi import api, extract, lrc, pipeline
from dsapi.errors import TaskError, error_info


@pytest.mark.parametrize('raw,stage,code', [
    ("ValueError: Phonemes missing from HubertFA mixed vocabulary: ['zh/H']", 'align', 'PHONEME_UNSUPPORTED'),
    ('ValueError: English word missing from voicebank dictionary: wavyarty', 'align', 'LYRIC_WORD_NOT_FOUND'),
    ('ValueError: no singable lyric lines parsed from LRC within the audio duration; check lyric timestamps and content', 'select', 'LYRICS_MISSING'),
    ('RuntimeError: song has no LRC lyrics', 'fetch', 'LYRICS_MISSING'),
    ('ConnectError: [SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol', 'fetch', 'SOURCE_UNAVAILABLE'),
    ("CalledProcessError: Command '['ffprobe', 'source.audio']' returned non-zero exit status 1.", 'fetch', 'AUDIO_READ_FAILED'),
    ('ConnectError: model download failed', 'separate', 'UNKNOWN'),
    ('no complete lyric lines in the requested time window', 'select', 'UNKNOWN'),
    ('unexpected model dimensions', 'render_score', 'UNKNOWN'),
])
def test_legacy_categories(raw, stage, code):
    info = error_info(raw, stage)
    assert info['code'] == code
    assert info['retryable'] == (code == 'SOURCE_UNAVAILABLE')


def test_real_exceptions_keep_details():
    assert error_info(httpx.ConnectError('connection reset'), 'fetch')['code'] == 'SOURCE_UNAVAILABLE'
    exc = subprocess.CalledProcessError(1, ['ffprobe', 'source.audio'], stderr='Invalid data found when processing input')
    assert error_info(exc, 'fetch')['diagnostic'] == exc.stderr
    info = error_info(TaskError('original', 'LYRIC_WORD_NOT_FOUND', word='ooooo'), 'align')
    assert info['context']['word'] == 'ooooo'


def test_missing_word_captures_actual_lyric(tmp_path):
    class Bank:
        def knows(self, unit, lang): return False
    line = lrc.Line(1, 2, 'WavyArty', lang='en', _units=[('WavyArty', 'wavyarty')])
    with pytest.raises(TaskError) as caught:
        extract.write_segments([line], np.zeros(132300), tmp_path/'segments', Bank())
    assert str(caught.value) == 'English word missing from voicebank dictionary: wavyarty'
    assert caught.value.context == {'word':'wavyarty', 'text':'WavyArty', 'language':'en',
                                   'line':'WavyArty', 'start':1, 'source':'lyrics'}


def test_failed_job_persists_raw_error_and_classification(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.settings, 'WORK', tmp_path)
    job = pipeline.Job('failure', pipeline.JobSpec(song_id='1'), owner_id='alice')
    def fail():
        with job.stage('align'):
            raise TaskError('English word missing from voicebank dictionary: wavyarty',
                            'LYRIC_WORD_NOT_FOUND', word='wavyarty')
    monkeypatch.setattr(job, '_run', fail)
    state = job.run()
    assert state.status == 'failed'
    saved = json.loads((job.dir/'job.json').read_text())
    assert saved['error'] == 'TaskError: English word missing from voicebank dictionary: wavyarty'
    assert saved['error_info']['code'] == 'LYRIC_WORD_NOT_FOUND'
    assert 'Traceback' in (job.dir/'error.txt').read_text()


def test_old_task_error_info_is_owner_scoped_and_does_not_rewrite_file(tmp_path, monkeypatch):
    monkeypatch.setattr(api.settings, 'WORK', tmp_path)
    monkeypatch.setattr(api, '_jobs', {})
    monkeypatch.setattr(api, 'API_KEY', '')
    monkeypatch.setattr(api, 'API_KEYS', {'alice':'alice-key', 'bob':'bob-key'})
    folder = tmp_path/'old'; folder.mkdir()
    raw = "ValueError: Phonemes missing from HubertFA mixed vocabulary: ['zh/H']"
    saved = json.dumps({'task_id':'old','status':'failed','stage':'align','error':raw,
                        'outputs':{},'owner_id':'alice'})
    (folder/'job.json').write_text(saved)
    client = TestClient(api.app)
    result = client.get('/query_result?task_id=old', headers={'Authorization':'Bearer alice-key'})
    assert result.status_code == 200
    assert result.json()['error'] == raw
    assert result.json()['error_info']['context']['phonemes'] == ['zh/H']
    assert client.get('/query_result?task_id=old', headers={'Authorization':'Bearer bob-key'}).status_code == 404
    assert (folder/'job.json').read_text() == saved
