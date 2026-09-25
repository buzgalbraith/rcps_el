from .lossFunction import lossFunction, pl
from rcps_el.aggregators import Aggregator, safeMinAggregator


class commonAncestorsAtKLoss(lossFunction):
    """
    ICD-10 code prefix coverage: covered if any candidate shares the first
    prefix_len characters (ignoring the dot) of the target code. prefix_len=3 is
    the ICD-10 category (E11.9 -> E11), prefix_len=4 the first subcategory
    level (E11.9 -> E119), and prefix_len=7 reduces to exact code matching.
    """

    label_curie_col = "obj_synonyms"
    candidate_curie_col = "match_curies"
    default_agg_method: Aggregator = safeMinAggregator()

    def __init__(
        self,
        prefix_len: int = 3,
        agg_method=None,
    ):
        super().__init__(agg_method)
        self.prefix_len = prefix_len
        self.name = f"ICD prefix@{prefix_len} loss"

    def _prefix(self, code: str) -> str:
        return code.replace(".", "").strip().upper()[: self.prefix_len]

    def calc_loss(self, labels: list[str], candidate_set: list[str]) -> float:
        candidate_prefixes = {self._prefix(c) for c in candidate_set}
        term_losses = [
            float(self._prefix(label) not in candidate_prefixes) for label in labels
        ]
        return self.agg_method.execute(term_losses)
