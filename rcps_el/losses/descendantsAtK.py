from .hierarchicalLossFunction import hierarchicalLossFunction, Aggregator, safeMinAggregator

from polars import DataFrame

class descendantsAtK(hierarchicalLossFunction):
    """check if a target label is within the closest K+1 descendent of candidate matches"""
    label_curie_col = "obj_synonyms"
    candidate_curie_col = "match_curies"
    default_agg_method: Aggregator = safeMinAggregator()

    def __init__(
        self,
        k_size: int,
        agg_method:Aggregator=None,
        k_candidates: bool = False
    ):
        super().__init__(agg_method)
        self.candidate_aggregator = safeMinAggregator()
        self.k_size = k_size
        self.k_candidates:bool = k_candidates
        self.name = f"Descendants@{k_size} loss"
        ## k is hierarchy depth, not a candidate slice, unless k_candidates is set.
        ## Without the slice, shrinking the candidate set can only raise the loss. ##
        self.monotone_in_threshold = not k_candidates
        ## With the slice, monotonicity still holds if the candidate list is
        ## ordered by the score being thresholded. ##
        self.monotone_when_score_ordered = True

    def calc_loss(self, labels: list[str], candidate_set: list[str]) -> float:
        ## filter for only UMLS labels ## 
        umls_labels = [x.removeprefix("umls:") for x in labels if x.startswith("umls:")]
        umls_candidates = [x.removeprefix("umls:") for x in candidate_set]
        if self.k_candidates:
            umls_candidates = umls_candidates[:self.k_size]
        if len(umls_labels) == 0:
            raise ValueError(f"No UMLS code present {labels}")
        ## I do not think this should ever happen but lets confirm ## 
        if len(umls_labels) > 1: 
            raise ValueError(f"Multiple UMLS codes for {labels}")
        umls_target_label = umls_labels[0]
        ## get list of true sources ##
        target_sources = list(self.path_index[umls_target_label].keys())
        src_losses = [self._check_src(umls_target_label=umls_target_label, src=src, umls_candidates=umls_candidates) for src in target_sources]
        return self.agg_method.execute(src_losses)

    def _check_src(self, umls_target_label:str, src, umls_candidates:dict)->float:
        """for a given src check if any candidate has the target label in its paths"""
        native_target_id = self.path_index.get(umls_target_label).get(src).get("native_id")
        containment = [] 
        for candidate_umls_id in umls_candidates:
            record = self.path_index.get(candidate_umls_id)
            if src not in record:
                continue
            native_paths = record.get(src).get("paths")
            ## get only ancestors of candidate (ie potential descendent of target) k steps away (k+1 including true label) ## 
            native_paths = [native_path[-(self.k_size + 1):] for native_path in native_paths]
            containment.append(self.agg_method.execute([float(native_target_id not in pth) for pth in native_paths]))
        return self.candidate_aggregator.execute(containment)
    
    def processing_function(self, data_frame: DataFrame):
        return super().processing_function(data_frame)
