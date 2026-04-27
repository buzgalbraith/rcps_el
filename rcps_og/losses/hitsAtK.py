from .lossFunction import lossFunction, pl
from rcps_og.aggregators import Aggregator, safeMinAggregator


class hitsAtK(lossFunction):
    label_curie_col = "obj_synonyms"
    candidate_curie_col = "match_curies"
    default_agg_method: Aggregator = safeMinAggregator()
    name = ""

    def __init__(
        self,
        k_size: int,
        agg_method=None,
    ):
        super().__init__(agg_method)
        self.k_size = k_size
        self.name = f"Hits@{k_size} loss"

    def calc_loss(self, labels: list[str], candidate_set: list[str]) -> float:
        term_losses = [
            float(label not in candidate_set[: self.k_size]) for label in labels
        ]
        return self.agg_method.execute(term_losses)

    def processing_function(self, data_frame: pl.DataFrame):
        return super().processing_function(data_frame)
