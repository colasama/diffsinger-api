"""Random selection retains the existing lyric coverage and clipping guarantees."""
import pytest

from dsapi import lrc, selection


def song():
    return [lrc.Line(t, t+5, '迎着风向前走') for t in list(range(0, 30, 5)) + list(range(80, 110, 5))]


def test_random_seed_reproduces_and_varies_good_windows():
    lines = song()
    chosen = {lrc.pick_random(lines, seed=seed) for seed in range(20)}
    assert len(chosen) > 1
    assert lrc.pick_random(lines, seed=4) == lrc.pick_random(lines, seed=4)
    for i, j in chosen:
        assert lines[j-1].end - lines[i].start <= 32
        assert all(b.start-a.end <= 5 for a, b in zip(lines[i:j], lines[i+1:j]))
    assert lrc.pick_random([], seed=1) == (0, 0)


def test_selection_records_generated_seed_and_uses_complete_lyrics():
    first = selection.select(song(), 120)
    second = selection.select(song(), 120, seed=first.seed)
    assert first.metadata() == second.metadata()
    assert first.start <= first.lines[0].start
    assert first.end >= first.lines[-1].end
    assert first.end-first.start <= 40
    assert first.metadata()['languages'] == ['zh']


def test_manual_window_wins_and_never_clips_half_a_line():
    result = selection.select(song(), 120, start=0, end=110)
    assert result.strategy == 'manual' and result.seed is None
    assert all(ln.end <= result.end for ln in result.lines)
    assert result.lines == song()
    assert result.end-result.start == 111
    assert result.after == []


@pytest.mark.parametrize('kwargs', [{'start': 1}, {'start': 2, 'end': 1},
                                    {'start': float('nan'), 'end': 5}, {'strategy': 'bad'}])
def test_invalid_selection_fails_explicitly(kwargs):
    with pytest.raises(ValueError):
        selection.select(song(), 120, **kwargs)


def test_random_does_not_fall_back_to_sparse_opening():
    lines = [lrc.Line(0, 2, '天空'), lrc.Line(25, 27, '大地')]
    lines += [lrc.Line(t, t+5, '迎着风向前走') for t in range(60, 90, 5)]
    assert all(lrc.pick_random(lines, seed=s)[0] >= 2 for s in range(20))


def test_empty_lyrics_and_unusable_windows_have_distinct_errors():
    with pytest.raises(ValueError, match="parsed from LRC"):
        selection.select([], 120)
    with pytest.raises(ValueError, match="requested time window"):
        selection.select(song(), 120, start=40, end=50)
    with pytest.raises(ValueError, match="fits the clip limit"):
        selection.select([lrc.Line(0, 50, "迎着风")], 120)
