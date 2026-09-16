from rcps_el.dataset import medPathBenchmark
from rcps_el.losses import ancestorsAtK

from pystow import module

import json
# #
paths_path = module("medpath").base.joinpath("hierarchical_paths")
all_paths = {}
path_lengths = {i:0 for i in range(40)}
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
                new_path = [x.get('code') for x in path_obj]
                if record.get("vocabulary") == "MSH":
                    new_path = new_path[::-1]
                if native_id != new_path[-1]:
                    print(umls_code, record.get("vocabulary"))
                path_lengths[len(new_path)] += 1
                code_paths.append(new_path)
            
            new_record = {
                'native_id' : native_id, 
                'paths' : code_paths
            }
            if umls_code not in all_paths:
                all_paths[umls_code] = {}
            all_paths[umls_code][record.get("vocabulary")] = new_record 
loss = ancestorsAtK(k_size=5)
less_than_4 = path_lengths.get(0) + path_lengths.get(1) + path_lengths.get(2) + path_lengths.get(3)
all_paths = loss.path_index
## new use it to find ids ## 
df = medPathBenchmark().calibration_set
target_id = df['obj_synonyms'][0][1]
prediction_ids = df['match_curies'][0]
ex_prediction = prediction_ids[0].removeprefix("umls:")

## now check those codes ## 
true_paths = all_paths.get(target_id.removeprefix("umls:"))
pred_path = all_paths.get(ex_prediction.removeprefix("umls:"))
for src in true_paths:
    if src not in pred_path:
        continue
    src = "MSH"
    native_id_pred = pred_path.get(src).get("native_id")
    native_paths = true_paths.get(src).get("paths")
    native_id_label = true_paths.get(src).get("native_id")
    for native_path in native_paths:
        if native_id_pred in native_path:
            print("good")
