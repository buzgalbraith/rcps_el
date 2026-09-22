"""
A general class for retrieval focused scoring functions, weighted by the cumulative probability of other values
"""

from .scorer import Scorer, pl

from scipy.special import softmax
import numpy as np

from logging import getLogger

logger = getLogger(__name__)

class cumulativeRetrievalScorer(Scorer):
    raw_text_col = "text"
    entity_name_col = "match_scores"
    normalized_score = True

    def __init__(self, name:str, softmax_temperature:float = 0.5):
        """
        Score function directly using estimated value from retrieval method (with cumulative weighting)

        args:
            name (str): name of base retval method 
            softmax_temperature (float) : Temperature to use during softmax normalization, lower temperature sharpens distribution and lowers entropy
        """
        super().__init__()
        self.name = f"Cumulative {name}"
        self.softmax_temperature = softmax_temperature
    def score_sample(self, entity: str, candidates: list[float]) -> list[float]:
        """Assume that data frame already contains scores from retrieval method"""   
        raw = np.asarray(candidates, dtype=float)
        ## a mention with nothing retrieved has no mass to accumulate, and softmax
        ## raises on an empty array rather than returning one ##
        if raw.size == 0:
            return []
        ## the evaluator sorts only after scoring, so score_sample still sees raw
        ## dataset order; the cumulative sum has to impose the ranking itself ##
        order = np.argsort(-raw, kind="stable")
        ## calculate the score ## 
        p = softmax(raw[order] / self.softmax_temperature)
        tail = 1.0 - np.concatenate([[0.0], np.cumsum(p)[:-1]])
        scores = np.empty_like(tail)
        ## return to original order ## 
        scores[order] = tail
        return scores

    def processing_function(self, data_frame: pl.DataFrame):
        return super().processing_function(data_frame)
