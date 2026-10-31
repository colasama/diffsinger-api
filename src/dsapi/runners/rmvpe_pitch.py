"""RMVPE decoding shared by the GAME F0 worker and legacy SOME worker."""

import numpy as np
from modules.rmvpe import RMVPE


class SegmentedRMVPE(RMVPE):
    def decode(self, hidden, thred=0.03, use_viterbi=True):
        """Decode voiced runs independently, retaining local F0 below 50 ms.

        Keep RMVPE's original confidence mask so the Viterbi path cannot carry
        pitch across unvoiced gaps. The inherited get_pitch handles interpolation
        and time resampling exactly as before.
        """
        from modules.rmvpe.utils import to_local_average_f0, to_viterbi_f0

        f0 = to_local_average_f0(hidden, thred=thred)
        # infer_from_audio passes use_viterbi=False by default; this worker's
        # policy always uses segmented Viterbi, regardless of that vendor flag.
        edges = np.flatnonzero(np.diff(np.r_[False, f0 > 0, False].astype(int)))
        for start, stop in zip(edges[::2], edges[1::2]):
            if stop - start >= 5:
                f0[start:stop] = to_viterbi_f0(hidden[:, start:stop], thred=thred)
        return f0
