"""
fuzzy string scorer class
"""

from .scorer import Scorer, pl
from rapidfuzz import fuzz


class fuzzyStringScore(Scorer):
    raw_text_col = "text"
    entity_name_col = "match_names"
    name = "fuzzy_string_scores"

    def score_sample(self, entity: str, candidates: list[str]) -> list[float]:
        scores = []
        for candidate_name in candidates:
            score = fuzz.ratio(entity, candidate_name) / 100
            scores.append(score)
        return scores

    def processing_function(self, data_frame: pl.DataFrame):
        return super().processing_function(data_frame)
