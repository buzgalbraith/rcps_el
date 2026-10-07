from .lossFunction import lossFunction, pl
from rcps_el.aggregators import Aggregator, safeMinAggregator
from rcps_el.utils.constants import MEDPATH_DIR

import json
import logging

logger = logging.getLogger(__name__)

class hierarchicalLossFunction(lossFunction):
    """Loss functions depending on path in some ontology structure"""
    def __init__(
        self,
        agg_method=None
    ):
        super().__init__(agg_method)
        self.path_index = self._load_path_index() 


    def _load_path_index(self)->dict:
        """Load and merge all UMLS IDs we have paths for"""
        logger.info("loading path dictionary")
        paths_path = MEDPATH_DIR.joinpath("hierarchical_paths")
        all_paths = {}
        for onto_paths in paths_path.iterdir():
            with open(onto_paths) as f:
                lines = f.read().strip()
                jsn = json.loads(lines)
                for umls_code in jsn:
                    record = jsn.get(umls_code)
                    code_obj = record.get("codes")
                    native_id = list(code_obj.keys())[0]
                    code_paths = []
                    for path_obj in code_obj.get(native_id).get("paths"):
                        found_path = [x.get('code') for x in path_obj]

                        ## MeSH (abbreviated here as MSH) paths are backwards relative to others so need to flip order in that case ##
                        if record.get("vocabulary") == "MSH":
                            found_path = found_path[::-1]
                        ## there are edge cases where the target id has changed since the path was written in this case we just use the actual path end as the target (this only effects 57 of ~22K paths) ## 
                        if found_path[-1] != native_id:
                            native_id = found_path[-1]
                        code_paths.append(found_path)
                    new_record = {
                        'native_id' : native_id, 
                        'paths' : code_paths
                    }
                    if umls_code not in all_paths:
                        all_paths[umls_code] = {}
                    all_paths[umls_code][record.get("vocabulary")] = new_record 
        return all_paths
