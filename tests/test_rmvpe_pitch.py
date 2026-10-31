"""Verify decoder boundaries without loading a neural checkpoint."""

import runpy
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


@pytest.mark.parametrize('local,expected_runs', [
    ([0] * 20, []),
    ([220] * 4, []),
    ([220] * 5, [(0, 5)]),
    ([220] * 6 + [0] + [330] * 4 + [0] + [440] * 5, [(0, 6), (12, 17)]),
])
def test_viterbi_preserves_unvoiced_gaps_and_short_runs(monkeypatch, local, expected_runs):
    local = np.asarray(local, dtype=np.float32)
    hidden = np.arange(len(local))[None, :, None]
    calls = []
    vendor = ModuleType('modules.rmvpe')
    vendor.RMVPE = object
    decoders = ModuleType('modules.rmvpe.utils')

    def local_decode(scores, thred):
        assert thred == .03
        return local.copy()

    def viterbi_decode(scores, thred):
        assert thred == .03
        start, stop = int(scores[0, 0, 0]), int(scores[0, -1, 0]) + 1
        calls.append((start, stop))
        return local[start:stop] + 10

    decoders.to_local_average_f0 = local_decode
    decoders.to_viterbi_f0 = viterbi_decode
    monkeypatch.setitem(sys.modules, 'modules.rmvpe', vendor)
    monkeypatch.setitem(sys.modules, 'modules.rmvpe.utils', decoders)
    path = Path(__file__).resolve().parents[1] / 'src/dsapi/runners/rmvpe_pitch.py'
    model = runpy.run_path(str(path))['SegmentedRMVPE']()
    # The inherited vendor inference passes False explicitly by default.
    result = model.decode(hidden, use_viterbi=False)
    expected = local.copy()
    for start, stop in expected_runs:
        expected[start:stop] += 10
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(result == 0, local == 0)
    assert calls == expected_runs
