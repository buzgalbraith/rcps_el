from .hierarchicalLossFunction import hierarchicalLossFunction, Aggregator, safeMinAggregator

from polars import DataFrame

class hierarchyAtK(hierarchicalLossFunction):
    """check if any term is within the closest K+1 ancestors or descendants of the target label """
    label_curie_col = "obj_synonyms"
    candidate_curie_col = "match_curies"
    default_agg_method: Aggregator = safeMinAggregator()


    def __init__(
        self,
        k_size: int,
        k_candidates:bool = False, 
        agg_method:Aggregator = None,
    ):
        super().__init__(agg_method)
        self.candidate_aggregator = safeMinAggregator()
        self.k_size = k_size
        self.k_candidates:bool =  k_candidates
        self.name = f"Hierarchy@{k_size} loss"
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
        ## get paths for each sources ##
        source_paths = self.path_index.get(umls_target_label)
        ancestor_losses, descendent_losses = [], []
        for src in source_paths:
            ancestor_losses.append(self._check_ancestor_src(src, source_paths, umls_candidates))
            descendent_losses.append(self._check_descendant_src(umls_target_label, src, umls_candidates))
        ## take the min of both losses ##  
        return min(
            self.agg_method.execute(ancestor_losses),
            self.agg_method.execute(descendent_losses)
        )

    def _check_ancestor_src(self, src:str, labels_paths:dict, umls_candidates:list[str])->float:
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
    
    def _check_descendant_src(self, umls_target_label:str, src, umls_candidates:dict)->float:
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
