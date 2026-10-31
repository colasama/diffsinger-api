"""Long-input scheduling without loading neural networks or vendor packages."""
import json
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize('runner', ['f0_runner', 'some_runner'])
def test_long_runners_bound_model_inputs_and_preserve_timeline(monkeypatch, tmp_path, runner):
    # f0_runner fixes 44.1 kHz/512; SOME reads the same settings from its model.
    sr, hop = 44100, 512
    waveform = np.zeros(round(sr*81.5), np.float32)
    pitch_lengths, note_lengths = [], []
    def pitch(samples, **kwargs):
        pitch_lengths.append(len(samples))
        return np.full(kwargs['length'], 261.6), np.zeros(kwargs['length'], bool)
    modules = ModuleType('modules')
    modules.__path__ = []
    rmvpe = ModuleType('modules.rmvpe');rmvpe.RMVPE = object
    modules.rmvpe = rmvpe
    utils = ModuleType('utils');utils.__path__ = []
    for name, value in {
        'librosa':SimpleNamespace(load=lambda *a,**k:(waveform,sr)),
        'inference':SimpleNamespace(), 'modules':modules, 'modules.rmvpe':rmvpe,
        'utils':utils, 'utils.binarizer_utils':SimpleNamespace(get_pitch_parselmouth=pitch),
        'utils.slicer2':SimpleNamespace(Slicer=lambda **k:SimpleNamespace(slice=lambda w:[{'offset':0,'waveform':w}])),
    }.items():
        monkeypatch.setitem(sys.modules,name,value)
    path=Path(__file__).resolve().parents[1]/'src/dsapi/runners'/f'{runner}.py'
    monkeypatch.setattr(sys, 'path', sys.path.copy())
    monkeypatch.syspath_prepend(str(path.parent))
    # Do not retain a helper subclass based on this test's stub vendor class.
    helper = ModuleType('rmvpe_pitch')
    helper.__dict__.update(runpy.run_path(str(path.with_name('rmvpe_pitch.py'))))
    monkeypatch.setitem(sys.modules, 'rmvpe_pitch', helper)
    namespace=runpy.run_path(str(path))
    main=namespace['main']
    model=SimpleNamespace(get_pitch=pitch)
    output=tmp_path/'f0.npy'
    if runner=='f0_runner':
        main.__globals__['load_models']=lambda *a:model
        main('checkpoint','audio',str(output),'cpu')
    else:
        def infer(inputs):
            assert len(inputs)==1
            note_lengths.append(len(inputs[0]))
            return [{'note_midi':[60],'note_dur':[len(inputs[0])/sr],'note_rest':[False]}]
        main.__globals__['load_models']=lambda *a:({'audio_sample_rate':sr,'hop_size':hop},SimpleNamespace(infer=infer),model)
        notes=tmp_path/'notes.json'
        main('checkpoint','rmvpe','audio',str(notes),str(output),'cpu')
        assert max(note_lengths)<=40*sr
        results=json.loads(notes.read_text())['notes']
        assert [n['start'] for n in results]==[0,40,80]
        assert sum(n['dur'] for n in results)==81.5
    data=np.load(output)
    assert data.shape==(2,(len(waveform)+hop-1)//hop)
    assert (data[0]==261.6).all() and not data[1].any()
    assert len(pitch_lengths)==3 and max(pitch_lengths)<45*sr
