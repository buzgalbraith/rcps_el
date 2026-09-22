from .hierarchicalLossFunction import hierarchicalLossFunction, Aggregator, safeMinAggregator

from polars import DataFrame

class ancestorsAtK(hierarchicalLossFunction):
    """check if any terms is within the closest K+1 ancestors of the target label """
    label_curie_col = "obj_synonyms"
    candidate_curie_col = "match_curies"
    default_agg_method: Aggregator = safeMinAggregator()


    def __init__(
        self,
        k_size: int,
        agg_method:Aggregator = None,
        k_candidates: bool = False
    ):
        super().__init__(agg_method)
        self.candidate_aggregator = safeMinAggregator()
        self.k_size = k_size
        self.k_candidates:bool = k_candidates
        self.name = f"Ancestors@{k_size} loss"
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
        ## get paths for each sources ##
        source_paths = self.path_index.get(umls_labels[0])
        ## check the paths in each sources ## 
        src_losses = [self._check_src(src, source_paths, umls_candidates) for src in source_paths]
        return self.agg_method.execute(src_losses)
    def _check_src(self, src:str, labels_paths:dict, umls_candidates:list[str])->float:
        """for a given ontology source check if any label exists in its path"""
        ## drop candidates that do not have a mapping in the desired src ontology ## 
        native_candidates = [self.path_index.get(x).get(src).get("native_id") for x in umls_candidates if src in self.path_index.get(x) ]

        containment = []

        for native_path in labels_paths.get(src).get("paths"):
            ## get only ancestors k steps away (k+1 including true label) ## 
            native_path = native_path[-(self.k_size + 1):]
            containment.append(self.candidate_aggregator.execute([float(x not in native_path) for x in native_candidates]))
        ## agg at the source level for ease ##     
        return self.agg_method.execute(containment)
