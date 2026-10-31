# Japanese HubertFA validation — 2026-10-31

Two Japanese excerpts from NetEase song `3413072220` (直到大地变成一颗酸橙)
completed alignment, GAME note extraction, score quantization, mimic synthesis,
and score synthesis with native Japanese HubertFA phonemes.

The comparison used the same clip, separated lead, instrumental and RMVPE F0 for
each pair. `comparison.json` stores input SHA-256 hashes. Both control and
HubertFA render runs set ONNX Runtime's seed to 123; this does not guarantee
bitwise reproducibility across execution providers. Separation and F0 were
cached, while alignment-dependent note extraction and synthesis were rerun.

| Excerpt | Source window | Units | SOFA control task | HubertFA task |
|---|---|---:|---|---|
| First | 76.203–84.253 s | 9 | `japanese-orange-sofa-control-20261031` | `japanese-orange-hubertfa-20261031` |
| Second | 89.850–99.250 s | 12 | `japanese-orange-second-sofa-control` | `japanese-orange-second-hubertfa` |

## Common automatic timing check

The source and both mimic outputs were aligned with the **same HubertFA model
and dictionary**. The table measures synthesized versus source unit onset
differences. This is model-based timing consistency, **not** a human-labelled
alignment accuracy or naturalness score; using HubertFA as evaluator can favour
HubertFA-derived timings. SOFA confidence and HubertFA outputs are not directly
comparable, and HubertFA does not expose a confidence value here.

| Excerpt | SOFA median difference | HubertFA median difference | SOFA within 50 ms | HubertFA within 50 ms |
|---|---:|---:|---:|---:|
| First | 528.5 ms | 18.0 ms | 22.2% | 88.9% |
| Second | 934.4 ms | 4.6 ms | 8.3% | 83.3% |

HubertFA produced no nonpositive phoneme durations in either excerpt. The second
excerpt's units shorter than 50 ms fell from three to zero. The first still has
one roughly 11 ms vowel; this requires listening or manual annotation to judge.
Both synthesized modes contain finite, non-silent audio.

## Artifacts and reproduction

All task artifacts live under `work/<task-id>/`. Combined listening files in
`work/japanese-hubertfa-comparison/` concatenate the two excerpts with a one-second
gap (second excerpt begins at 9.05 seconds):

- `source.mp3`: separated original voice
- `hubertfa-mimic.mp3` / `sofa-mimic.mp3`: original F0, different alignment
- `hubertfa-score.mp3` / `sofa-score.mp3`: each alignment's extracted score
- `metrics.json`: common-evaluator measurements and their limitation

Run `scripts/compare_japanese_alignment.py` against a completed Japanese task to
create a fresh comparison task. `--aligner sofa` creates a control;
`--aligner hubertfa` creates the native Japanese alternative.

At the time of this comparison, Japanese alignment was opt-in and language
routing was song-level. Subsequent integration made Japanese HubertFA the default,
removed the experimental language gate, and added mixed-language routing and
random lyric selection; see README for the current workflow. This comparison covers
two excerpts from one song and does not establish general Japanese quality.

The integration's regression run passed 55 Python tests, including Japanese
backend selection, native dictionary routing, closure vocabulary handling,
unsupported-phoneme rejection, and existing Chinese/English alignment paths.

Audio and local synthesis assets are not distributed with this project.

## Mixed-language integration check

`random-mixed-orange-20261031` uses the complete original LRC with random seed 7,
without a language override or a Japanese-only input. The selected lyric interval
is 63.961–91.350 seconds; its padded audio window is 62.461–92.350 seconds.
All 27 selected English words and 9 Japanese morae reached HubertFA in order.
The alignment metadata identifies `mixed`, and GAME and `score.json` contain both
`en` and `ja`. Mimic and score produced finite, non-silent 29.889-second outputs.
The integration regression run passed 87 Python and 2 browser tests.
