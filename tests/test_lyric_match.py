import pytest

from dsapi import lrc, lyric_match, selection


def line(start, text, duration=5):
    return lrc.Line(start, start + duration, text)


@pytest.mark.parametrize('text,query,kind', [
    ('迎着风，向前走！', '迎着风向前走', 'exact'),
    ('迎着风向前走', '迎着风向前', 'prefix'),
    ('我们迎着风向前走', '迎着风向前', 'contains'),
    ('ＨＥＬＬＯ，World!', 'hello world', 'exact'),
    ('我说Hello君は笑う', '我说 hello 君は笑う', 'exact'),
    ("Don't stop believing", 'dont stop', 'prefix'),
    ('天空', '天空', 'exact'),
])
def test_normalization_and_match_kinds(text, query, kind):
    matches = lyric_match.find_matches([line(0, text)], query)
    assert len(matches) == 1 and matches[0].kind == kind


@pytest.mark.parametrize('text,query', [
    ('迎着风向前走', '风'), ('hello world', 'hell'),
    ('the world goes on', 'he world'), ('你好世界', '世界'),
    ('世界你好', '你好世界'), ('你你你你', '！！！'),
])
def test_no_short_substring_or_reordered_match(text, query):
    assert lyric_match.find_matches([line(0, text)], query) == []


def test_cross_line_match_must_be_included_and_cannot_cross_gap():
    lines = [line(10, '迎着风'), line(15, '向前走'), line(20, '我们出发吧')]
    result = selection.select(lines, 100, lyric_query='迎着风向前走')
    assert result.strategy == 'lyric'
    assert result.lines[:2] == lines[:2]
    assert result.match['text'] == '迎着风 / 向前走'
    lines[1].start, lines[1].end = 30, 35
    assert not lyric_match.find_matches(lines, '迎着风向前走')


def test_anchor_beats_global_score_and_does_not_prepend_near_end():
    lines = [line(t, '我们一起向前走') for t in range(0, 30, 5)]
    lines.append(line(95, '迎着风向前走'))
    result = selection.select(lines, 100, lyric_query='迎着风向前走', seed=123)
    assert result.lines == lines[-1:]
    assert result.strategy == 'lyric' and result.seed is None
    assert result.start == 93.5 and result.end == 100
    assert result.metadata()['match']['start'] == 95


def test_repeated_anchor_uses_quality_then_earliest_tie():
    lines = [line(0, '迎着风向前走')]
    lines += [line(t, '迎着风向前走' if t in (30, 80) else '我们一起出发吧')
              for t in [*range(30, 55, 5), *range(80, 105, 5)]]
    result = selection.select(lines, 120, lyric_query='迎着风向前走')
    assert result.lines[0].start == 30


def test_manual_override_and_unmatched_fallback_are_reported():
    lines = [line(t, '迎着风向前走' if t == 0 else '我们一起出发吧') for t in range(0, 30, 5)]
    manual = selection.select(lines, 40, start=10, end=25, lyric_query='迎着风向前走')
    assert manual.strategy == 'manual' and manual.lines[0].start == 10
    assert manual.match is None
    fallback = selection.select(lines, 40, lyric_query='找不到这一句', seed=7)
    baseline = selection.select(lines, 40, seed=7)
    assert fallback.lines == baseline.lines
    assert fallback.strategy == 'random' and fallback.fallback_reason == 'no_usable_lyric_match'


def test_unusable_cross_line_match_does_not_truncate_query():
    lines = [line(0, '迎着风', 20), line(20, '向前走', 20), line(50, '我们一起出发吧')]
    result = selection.select(lines, 60, lyric_query='迎着风向前走', seed=1)
    assert result.strategy == 'random' and result.match is None


def test_pipeline_forwards_anchor_and_saves_selected_window(tmp_path, monkeypatch):
    import json
    from dsapi import pipeline

    audio = tmp_path / 'input.wav'
    audio.write_bytes(b'fake')
    lyrics = tmp_path / 'input.lrc'
    lyrics.write_text('[00:00]我们一起出发吧\n[00:20]迎着风向前走\n[00:25]我们一起出发吧')
    monkeypatch.setattr(pipeline.settings, 'WORK', tmp_path / 'work')
    monkeypatch.setattr(pipeline, 'voicebank', lambda: None)
    monkeypatch.setattr(pipeline.audio, 'duration', lambda _: 30.)

    def stop_at_cut(source, target, start, end):
        assert start == 18.5 and end <= 30
        raise RuntimeError('stop before inference')

    monkeypatch.setattr(pipeline.audio, 'cut', stop_at_cut)
    job = pipeline.Job('test', pipeline.JobSpec(audio_path=str(audio), lrc_path=str(lyrics),
                                               lyric_query='迎着风向前走'))
    with pytest.raises(RuntimeError, match='stop before inference'):
        job._run()
    saved = json.loads((job.dir / 'job.json').read_text())
    assert saved['selection']['strategy'] == 'lyric'
    assert saved['selection']['match']['start'] == 20
    assert saved['lines'][0]['text'] == '迎着风向前走'
