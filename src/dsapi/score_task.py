"""User-authored scores use the normal queue, ownership and audio delivery."""

import json

from . import settings
from .pipeline import Job, voicebank
from .score import render_score
from .text_score import ParsedScore


class ScoreJob(Job):
    def __init__(self, task_id, spec, parsed: ParsedScore, *, owner_id=None):
        self.parsed = parsed
        super().__init__(task_id, spec, owner_id=owner_id)
        self.state.song = {"name": "自填乐谱"}
        self.state.window = [0, parsed.duration_seconds]
        self.state.jianpu = spec.score_text
        self.save()

    def _run(self):
        parsed, st = self.parsed, self.state
        with self.stage("parse_score"):
            (self.dir / "score.json").write_text(json.dumps(parsed.payload(), ensure_ascii=False, indent=2))
            st.grid = {"bpm": parsed.bpm, "meter": parsed.meter}
            st.tonic = parsed.key
            st.lines = [{"start": p.syllables[0].start, "end": p.syllables[-1].end,
                         "text": "".join(s.text for s in p.syllables)} for p in parsed.phrases]
        vb = voicebank(st.spec.voice_id, st.spec.synthesis_seed)
        with self.stage("render_score"):
            rendered = ((p, render_score(vb, p)) for p in parsed.phrases)
            self._mix("score", rendered, round(parsed.duration_seconds*settings.SAMPLE_RATE))
