"""
Rank-cutoff baseline: score each candidate by its negative retrieval rank.

Thresholding this score keeps the same number of top candidates for every
mention, so calibrating it with RCPS gives a single fixed top-r cutoff (the
"calibrated top-k" baseline, cf. Fisch et al. 2021, Appendix D.2). It ignores
how confident the retriever is on a given mention, which is what the
softmax-based scores S1 and S2 are meant to exploit.
"""

from .scorer import Scorer, pl

import numpy as np

from logging import getLogger

logger = getLogger(__name__)


class rankScorer(Scorer):
    raw_text_col = "text"
    entity_name_col = "match_scores"

    def __init__(self, name: str):
        """
        args:
            name (str): name of base retrieval method
        """
        super().__init__()
        self.name = f"Rank {name}"

    def score_sample(self, entity: str, candidates: list[float]) -> list[float]:
        """
        Return -rank (1 = best) for each candidate, in the candidates' original order.

        The evaluator keeps candidates with score >= q, so q in (-(r+1), -r] retains
        exactly the top r. Ranks come from the raw retrieval similarity; ties and
        missing scores are broken by dataset order (stable sort, missing scores last),
        which makes the ranking, and so the nested sets, deterministic.
        """
        if candidates is None or len(candidates) == 0:
            return []
        raw = np.asarray(
            [-np.inf if c is None else c for c in candidates], dtype=float
        )
        raw = np.where(np.isnan(raw), -np.inf, raw)
        order = np.argsort(-raw, kind="stable")
        ranks = np.empty(len(raw), dtype=float)
        ranks[order] = np.arange(1, len(raw) + 1, dtype=float)
        return (-ranks).tolist()

    def processing_function(self, data_frame: pl.DataFrame):
        return super().processing_function(data_frame)
