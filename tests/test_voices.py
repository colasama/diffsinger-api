from types import SimpleNamespace
import json
import numpy as np
import pytest
from fastapi.testclient import TestClient
from dsapi import api, voices, pipeline
from dsapi.voicebank import Voicebank


def test_catalog_aliases_and_availability(voice_files):
    catalog = voices.catalog()
    assert catalog['default_voice_id'] == 'default'
    assert len(catalog['voices']) == 3 and all(v['available'] for v in catalog['voices'])
    assert '测试' in voices.get('default').aliases
    assert '乙' in voices.get('imported-b').aliases
    (voices.get().root / 'dsdur/stage.A.emb').unlink()
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
    assert state['spec']['voice_id']==voices.DEFAULT_VOICE and state['voice']['name']=='测试音源'
    assert state['spec']['lyric_query']==query and state['spec']['synthesis_seed'] is not None
    assert '测试音源' in state['attribution']
    assert client.post('/release_task', json={'song_id':'1','voice_id':'../../bad'}).status_code==422
    (voices.get().root/'dsconfig.yaml').unlink()
    assert client.post('/release_task', json={'song_id':'1'}).status_code==503
    assert client.post('/render_score', json={'phrases':[],'voice_id':'nope'}).status_code==422


def test_legacy_attribution_is_unknown(monkeypatch, tmp_path):
    monkeypatch.setattr(api.settings,'WORK',tmp_path)
    monkeypatch.setattr(api,'_jobs',{})
    folder=tmp_path/'old';folder.mkdir()
    (folder/'job.json').write_text(json.dumps({'task_id':'old','status':'done','outputs':{},'spec':{}}))
    assert api.query_result('old')['voice']['id']=='unknown'


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
                           data={'lrc_text':'[00:00]啦','voice_id':'imported-b',
                                 'lyric_query':'啦','lyric_extent':'exact','synthesis_seed':'7'})
    assert response.status_code == 200
    spec = api._jobs[response.json()['task_id']].state.spec
    assert (spec.voice_id,spec.lyric_extent,spec.synthesis_seed)==('imported-b','exact',7)
    seen=[]
    monkeypatch.setattr(api, 'voicebank', lambda voice,seed: seen.append((voice,seed)))
    # Empty score still validates/resolves the requested voice before refusing synthesis.
    response=client.post('/render_score',json={'voice_id':'imported-fixed','synthesis_seed':3,'phrases':[]})
    assert response.status_code==400 and seen==[('imported-fixed',3)]


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


def test_no_import_has_no_available_voice(monkeypatch, tmp_path):
    monkeypatch.setattr(voices.settings, 'VOICEBANK', tmp_path/'not-imported')
    default, entries = voices.load_catalog(None)
    monkeypatch.setattr(voices, 'VOICES', entries)
    assert default == 'default'
    assert len(entries) == 1 and entries[0].speaker is None
    assert not voices.catalog()['voices'][0]['available']
    with pytest.raises(api.HTTPException) as error:
        api.validate_voice(default)
    assert error.value.status_code == 503
    assert '导入' in error.value.detail


def test_import_catalog_relative_paths_and_private_metadata(tmp_path, monkeypatch, voice_files):
    path = tmp_path/'voices.json'
    path.write_text(json.dumps({'default_voice_id':'custom', 'voices':[
        {'id':'custom','name':'Imported model','path':'bank','speaker':'A',
         'aliases':['mine'],'model_version':'local-1'},
        {'id':'fixed','name':'Fixed model','path':str(tmp_path/'fixed')}
    ]}))
    default, entries = voices.load_catalog(path)
    assert default == 'custom'
    assert entries[0].root == tmp_path/'bank'
    assert entries[0].aliases == ('mine',)
    assert entries[1].speaker is None
    for entry in entries:
        entry.validate()
    monkeypatch.setattr(voices, 'VOICES', entries)
    monkeypatch.setattr(voices, 'DEFAULT_VOICE', default)
    catalog = voices.catalog()
    assert catalog['default_voice_id'] == 'custom'
    assert all(item['available'] for item in catalog['voices'])
    assert str(tmp_path) not in json.dumps(catalog)
    assert 'path' not in catalog['voices'][0]
    assert voices.get().snapshot()['model_version'] == 'local-1'


@pytest.mark.parametrize('change', ['duplicate', 'bad-id', 'missing-default', 'bad-alias', 'bad-path', 'bad-speaker'])
def test_invalid_import_configuration_fails(tmp_path, change):
    item = {'id':'custom','name':'Imported','path':'bank'}
    data = {'default_voice_id':'custom','voices':[item]}
    if change == 'duplicate': data['voices'].append(dict(item))
    elif change == 'bad-id': item['id'] = '../escape'
    elif change == 'missing-default': data['default_voice_id'] = 'absent'
    elif change == 'bad-alias': item['aliases'] = 'not-a-list'
    elif change == 'bad-path': item['path'] = ''
    elif change == 'bad-speaker': item['speaker'] = 123
    path = tmp_path/'voices.json'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        voices.load_catalog(path)
