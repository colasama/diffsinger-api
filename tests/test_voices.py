from types import SimpleNamespace
import json
import numpy as np
import pytest
from fastapi.testclient import TestClient
from dsapi import api, voices, pipeline
from dsapi.voicebank import Voicebank


def test_catalog_aliases_and_availability(voice_files):
    catalog = voices.catalog()
    assert catalog['default_voice_id'] == 'imported-voice-e'
    assert len(catalog['voices']) == 8 and all(v['available'] for v in catalog['voices'])
    for voice_id, alias in [('imported-voice-e','戊'), ('imported-voice-b','乙'), ('imported-voice-a','甲语'), ('imported-voice-g','庚')]:
        assert alias in voices.get(voice_id).aliases
    assert not any(len(alias) == 1 for alias in voices.get('imported-voice-a').aliases)
    (voices.get().root / 'dsdur/stage.SpeakerE.emb').unlink()
    assert not next(v for v in voices.catalog()['voices'] if v['id']==voices.DEFAULT_VOICE)['available']


def test_api_voice_snapshot_unknown_unavailable_and_long_body(voice_files, monkeypatch, tmp_path):
    monkeypatch.setattr(api.settings, 'WORK', tmp_path/'work')
    monkeypatch.setattr(api, 'API_KEY', '')
    monkeypatch.setattr(api, 'API_KEYS', {})
    monkeypatch.setattr(api, '_jobs', {})
    monkeypatch.setattr(api, '_executor', SimpleNamespace(submit=lambda _: None))
    client = TestClient(api.app)
    query = '一二三四五\n'*110
    response = client.post('/release_task', json={'song_id':'1','lyric_query':query})
    assert response.status_code == 200
    state = client.get('/query_result', params={'task_id':response.json()['task_id']}).json()
    assert state['spec']['voice_id']==voices.DEFAULT_VOICE and state['voice']['name']=='用户音源戊'
    assert state['spec']['lyric_query']==query and state['spec']['synthesis_seed'] is not None
    assert '用户音源戊' in state['attribution']
    assert client.post('/release_task', json={'song_id':'1','voice_id':'../../bad'}).status_code==422
    (voices.get().root/'dsconfig.yaml').unlink()
    assert client.post('/release_task', json={'song_id':'1'}).status_code==503
    assert client.post('/render_score', json={'phrases':[],'voice_id':'nope'}).status_code==422


def test_legacy_attribution_stays_imported_fixed(monkeypatch, tmp_path):
    monkeypatch.setattr(api.settings,'WORK',tmp_path)
    monkeypatch.setattr(api,'_jobs',{})
    folder=tmp_path/'old';folder.mkdir()
    (folder/'job.json').write_text(json.dumps({'task_id':'old','status':'done','outputs':{},'spec':{}}))
    assert api.query_result('old')['voice']['id']=='imported-fixed'


def test_retake_inputs_and_speakers_are_request_local(tmp_path):
    for name, value in [('A',1.),('B',2.)]:
        np.full(384,value,dtype='<f4').tofile(tmp_path/f'stage.{name}.emb')
    feeds=[]
    class Session:
        def get_inputs(self):
            return [SimpleNamespace(name=name,shape=shape) for name,shape in {
                'encoder_out':[1,'n_tokens',384], 'ph_dur':[1,'n_tokens'],
                'spk_embed':[1,'n_frames',384], 'encoder_out_b':[1,'n_tokens',384],
                'blend':[1,'n_frames'], 'noise':[1,4,18,'n_frames'],
                'retake':[1,'n_frames',4], 'tension':[1,'n_frames'], 'mouth_opening':[1,'n_frames']}.items()]
        def get_outputs(self):
            return [SimpleNamespace(name=n+'_pred') for n in ['breathiness','voicing','tension','mouth_opening']]
        def run(self, _, feed):
            feeds.append(feed)
            return []
    model=SimpleNamespace(root=tmp_path,cfg={'speakers':['stage.A','stage.B'],'hidden_size':384},session=lambda _:Session())
    for speaker in ['A','B','A']:
        bank=Voicebank.__new__(Voicebank);bank.speaker=speaker;bank.rng=np.random.default_rng(7);bank._embeddings={}
        bank._run(model,'variance',{'encoder_out':np.ones((1,2,384),np.float32),'ph_dur':np.array([[2,3]])},5)
    assert [f['spk_embed'][0,0,0] for f in feeds]==[1,2,1]
    assert feeds[0]['retake'].shape==(1,5,4) and feeds[0]['retake'].all()
    assert feeds[0]['noise'].shape==(1,4,18,5)
    np.testing.assert_array_equal(feeds[0]['noise'],feeds[2]['noise'])
    assert not feeds[0]['blend'].any()


def test_explicit_voice_and_extent_reach_upload_and_score(voice_files, monkeypatch, tmp_path):
    monkeypatch.setattr(api.settings, 'WORK', tmp_path/'work')
    monkeypatch.setattr(api, 'API_KEY', '')
    monkeypatch.setattr(api, 'API_KEYS', {})
    monkeypatch.setattr(api, '_jobs', {})
    monkeypatch.setattr(api, '_executor', SimpleNamespace(submit=lambda _: None))
    client = TestClient(api.app)
    response = client.post('/release_task_upload', files={'audio_file':('song.wav',b'fake')},
                           data={'lrc_text':'[00:00]啦','voice_id':'imported-voice-b',
                                 'lyric_query':'啦','lyric_extent':'exact','synthesis_seed':'7'})
    assert response.status_code == 200
    spec = api._jobs[response.json()['task_id']].state.spec
    assert (spec.voice_id,spec.lyric_extent,spec.synthesis_seed)==('imported-voice-b','exact',7)
    seen=[]
    monkeypatch.setattr(api, 'voicebank', lambda voice,seed: seen.append((voice,seed)))
    # Empty score still validates/resolves the requested voice before refusing synthesis.
    response=client.post('/render_score',json={'voice_id':'imported-voice-g','synthesis_seed':3,'phrases':[]})
    assert response.status_code==400 and seen==[('imported-voice-g',3)]


def test_post_resolver_preserves_long_text(monkeypatch):
    monkeypatch.setattr(api, 'API_KEY', '')
    monkeypatch.setattr(api, 'API_KEYS', {})
    query='一二三四五\n'*110
    seen=[]
    def resolve(source,q,extent):
        seen.append((q,extent))
        return {'song':{'id':'song'},'lyric_query':q}
    monkeypatch.setattr(api.resolver,'resolve',resolve)
    response=TestClient(api.app).post('/resolve_song',json={'q':query,'lyric_extent':'exact'})
    assert response.status_code==200 and seen==[(query,'exact')]
