from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from dsapi import api, sources


@pytest.fixture
def client(monkeypatch, tmp_path, voice_files):
    monkeypatch.setattr(api.settings, 'WORK', tmp_path)
    monkeypatch.setattr(api, 'API_KEY', '')
    monkeypatch.setattr(api, 'API_KEYS', {})
    monkeypatch.setattr(api, '_jobs', {})
    monkeypatch.setattr(api, '_job_futures', {})
    # Do not start inference; assert exactly what the API persists for the worker.
    monkeypatch.setattr(api, '_executor', SimpleNamespace(submit=lambda _: None))
    return TestClient(api.app)


def test_default_and_explicit_selection_are_persisted(client):
    task = client.post('/release_task', json={'song_id':'1'}).json()['task_id']
    spec = client.get('/query_result', params={'task_id':task}).json()['spec']
    assert spec['selection'] == 'random' and spec['seed'] is None
    task = client.post('/release_task', json={'song_id':'1','selection':'chorus','seed':7}).json()['task_id']
    assert api._jobs[task].state.spec.selection == 'chorus'
    assert api._jobs[task].state.spec.seed == 7
    assert client.post('/release_task', json={'song_id':'1','selection':'typo'}).status_code == 422


def test_upload_supports_reproducible_random_selection(client):
    response = client.post('/release_task_upload', files={'audio_file':('clip.wav',b'audio','audio/wav')},
                           data={'lrc_text':'[00:01]Hello','selection':'random','seed':'42'})
    assert response.status_code == 200
    spec = api._jobs[response.json()['task_id']].state.spec
    assert spec.selection == 'random' and spec.seed == 42


def test_explicit_long_range_preserves_zero_start(client):
    response = client.post('/release_task', json={'song_id': '1', 'start': 0, 'end': 120})
    assert response.status_code == 200
    spec = api._jobs[response.json()['task_id']].state.spec
    assert spec.start == 0 and spec.end == 120


def test_lyric_query_persisted_for_both_submission_routes(client):
    query = '迎着风向前走'
    response = client.post('/release_task', json={'song_id': '1', 'lyric_query': query})
    assert response.status_code == 200
    task = response.json()['task_id']
    assert client.get('/query_result', params={'task_id': task}).json()['spec']['lyric_query'] == query
    response = client.post('/release_task_upload', files={'audio_file': ('clip.wav', b'audio', 'audio/wav')},
                           data={'lrc_text': '[00:01]迎着风向前走', 'lyric_query': query})
    assert response.status_code == 200
    assert api._jobs[response.json()['task_id']].state.spec.lyric_query == query
    assert client.post('/release_task', json={'song_id': '1', 'lyric_query': 'a' * (api.MAX_LYRIC_TEXT + 1)}).status_code == 422


def test_resolve_route_title_artist_errors_and_auth(client, monkeypatch):
    class Source:
        def search(self, q, limit):
            return [sources.Song('cover', '雨爱', ['其他歌手']), sources.Song('original', '雨爱', ['杨丞琳'])]

    monkeypatch.setattr(api.sources, 'get_source', Source)
    result = client.get('/resolve_song', params={'q': '雨爱 杨丞琳'})
    assert result.status_code == 200
    assert result.json()['song']['id'] == 'original'
    assert result.json()['match_type'] == 'title_artist'
    for q in ['', '  ', '！！！', 'a' * (api.MAX_LYRIC_TEXT + 1)]:
        assert client.get('/resolve_song', params={'q': q}).status_code == 422
    monkeypatch.setattr(Source, 'search', lambda *args, **kwargs: [])
    assert client.get('/resolve_song', params={'q': '不存在'}).status_code == 404
    monkeypatch.setattr(api, 'API_KEY', 'secret')
    assert client.get('/resolve_song', params={'q': '雨爱'}).status_code == 401
