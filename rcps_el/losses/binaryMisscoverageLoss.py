from .lossFunction import lossFunction, pl
from rcps_el.aggregators import Aggregator, safeMinAggregator


class binaryMisscoverageLoss(lossFunction):
    label_curie_col = "obj_synonyms"
    candidate_curie_col = "match_curies"
    default_agg_method: Aggregator = safeMinAggregator()
    name = "binary_misscoverage_loss"
    ## candidate sets are nested as q rises, so miscoverage can only increase,
    ## whatever order the candidates are listed in ##
    monotone_in_threshold = True
    monotone_when_score_ordered = True

    def calc_loss(self, labels: list[str], candidate_set: list[str]) -> float:
        term_losses = [float(label not in candidate_set) for label in labels]
        return self.agg_method.execute(term_losses)

    def processing_function(self, data_frame: pl.DataFrame):
        return super().processing_function(data_frame)
