"""Re-run a completed Japanese job with identical stems/F0 and another aligner.

Example: uv run python scripts/compare_japanese_alignment.py \
  work/japanese-orange-20261031 --task-id japanese-orange-hubertfa-20261031
"""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--task-id', required=True)
    parser.add_argument('--aligner', choices=('hubertfa', 'sofa'), default='hubertfa')
    args = parser.parse_args()
    os.environ['DSAPI_EXPERIMENTAL_JA'] = '1'
    os.environ['DSAPI_JA_ALIGNER'] = args.aligner
    from dsapi import audio, settings, tools
    from dsapi.pipeline import Job, JobSpec

    source = args.source.resolve()
    previous = json.loads((source/'job.json').read_text())
    meta = json.loads((source/'align_meta.json').read_text())
    if previous['status'] != 'done' or meta['lang'] != 'ja':
        raise ValueError('Source must be a completed Japanese job')
    if Path(args.task_id).name != args.task_id or args.task_id in ('.', '..'):
        raise ValueError('task-id must be a directory name')
    output = settings.WORK/args.task_id
    output.mkdir(exist_ok=False)

    def reuse_stems(clip, directory):
        # Do not compare different clip windows or stale separation results.
        if clip.read_bytes() != (source/'clip.wav').read_bytes():
            raise ValueError('Clip changed; cannot reuse source stems')
        shutil.copytree(source/'sep', directory)
        inst, = directory.glob('*_(Instrumental)_*.wav')
        return directory/'lead.wav', inst

    def reuse_f0(wav, directory):
        shutil.copy2(source/'f0.npy', directory/'f0.npy')
        f0, uv = np.load(directory/'f0.npy')
        return f0, uv.astype(bool)

    spec = JobSpec(**previous['spec'])
    spec.song_id = None
    spec.audio_path = str(next(source.glob('source.*')))
    spec.lrc_path = str(source/'lyrics.lrc')
    ort.set_seed(123)
    with patch.object(audio, 'separate_vocals', reuse_stems), patch.object(tools, 'extract_f0', reuse_f0):
        state = Job(args.task_id, spec).run()
    if state.status != 'done':
        raise RuntimeError(state.error)
    if (source/'segments/clip.lab').read_text() != (output/'segments/clip.lab').read_text():
        raise ValueError('Lyric units changed; this is not an aligner-only comparison')

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    comparison = {'source': str(source), 'output': str(output), 'aligner': args.aligner,
                  'reused': {name: digest(output/name) for name in ('clip.wav', 'sep/lead.wav', 'f0.npy')},
                  'sampling_seed': 123, 'timings': state.timings,
                  'note': 'Separation and F0 reused. GAME, quantization and synthesis rerun after alignment. '
                          'No human boundary labels; these are diagnostic comparisons, not accuracy scores.'}
    for name in comparison['reused']:
        assert digest(source/name) == comparison['reused'][name]
    for key, folder in [('before', source), ('after', output)]:
        aligned = json.loads((folder/'sofa.json').read_text())['clip']
        words = [w for w in aligned['words'] if w[0] not in ('SP', 'AP')]
        comparison[key] = {'words': words, 'first_onset': words[0][1], 'last_offset': words[-1][2],
                           'units_below_50ms': sum(e-s < .05 for _, s, e in words),
                           'units_above_1s': sum(e-s > 1 for _, s, e in words)}
    (output/'comparison.json').write_text(json.dumps(comparison, ensure_ascii=False, indent=2))
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
