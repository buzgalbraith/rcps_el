"""
Gilda scorer class
"""

from .scorer import Scorer, pl


class krissbertScorer(Scorer):
    raw_text_col = "text"
    entity_name_col = "match_scores"
    name = "KrissBERT_scores"

    def score_sample(self, entity: str, candidates: list[float]) -> list[float]:
        """Scores already assumed to be present"""
        return candidates

    def processing_function(self, data_frame: pl.DataFrame):
        return super().processing_function(data_frame)
