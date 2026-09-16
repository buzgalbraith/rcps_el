import json
from pathlib import Path

def process_json(raw_path):
    with open(raw_path, mode='r') as f:
        jsons = f.readlines()
        jsons = [x.strip() for x in jsons]
    for json_dict in jsons:
        jsn = json.loads(json_dict)
        full_text = jsn.get("text")
        document_id = jsn.get("doc_id")
        split = jsn.get("split")
        for mention in jsn.get("mentions"):
            nid = mention.get("native_id")
            semantic_type = mention.get("semantic_type")
            semantic_types.add(semantic_type)


if __name__ == "__main__":
    base = Path("/Users/buzgalbraith/workspace/MedPath/data_processed/documents/")
    semantic_types = set()
    for x in base.iterdir():
        print(str(x))
        process_json(str(x))
        print(len(semantic_types))