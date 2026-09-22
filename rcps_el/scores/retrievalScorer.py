"""
A general class for retrieval focused scoring functions
"""

from .scorer import Scorer, pl

from scipy.special import softmax
import numpy as np

from logging import getLogger

logger = getLogger(__name__)

class retrievalScorer(Scorer):
    raw_text_col = "text"
    entity_name_col = "match_scores"

    def __init__(self, name:str, normalized_score:bool = False, softmax_temperature:float = 0.5):
        """
        Score function directly using estimated value from retrieval method

        args:
            name (str): name of base retval method 
            normalize (bool) : If to normalize scores of retrieved candidates between zero and one
            softmax_temperature (float) : Temperature to use during softmax normalization, lower temperature sharpens distribution and lowers entropy
        """
        super().__init__()
        self.name = name
        self.normalized_score = normalized_score
        self.softmax_temperature = softmax_temperature
    def score_sample(self, entity: str, candidates: list[float]) -> list[float]:
        """Assume that data frame already contains scores from retrieval method"""   
        if not self.normalized_score:
            return candidates
        return softmax(np.asarray(candidates, dtype=float) / self.softmax_temperature)


    def processing_function(self, data_frame: pl.DataFrame):
        return super().processing_function(data_frame)
