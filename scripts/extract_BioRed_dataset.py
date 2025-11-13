"""
Processes the BioRed training data into a tabular format that can be more easily worked with.
"""

import polars as pl
import json
import tqdm
from pyobo import get_name
from rcps_og.utils.constants import RESOURCE_PATH, ENTITY_TYPE_MAPS


def extract_dataset(jsn: dict) -> list:
    """
    extract tabular data from the json dataset representation

    """
    records = []
    for document in tqdm.tqdm(jsn["documents"]):
        for passage in document["passages"]:
            full_text = passage["text"]
            for annotation in passage["annotations"]:
                entity_type = annotation["infons"]["type"]
                db = ENTITY_TYPE_MAPS[entity_type]
                identifier = (
                    annotation["infons"]["identifier"].strip().split(",")[0]
                )  # it is possible that there are more than one names that something is annotated as. for now we will just take the first
                # This is a weird formatting this i dont get
                if "|" in identifier:
                    by_split = identifier.split("|")
                    if len(by_split) > 2:
                        continue
                    else:
                        identifier = by_split[0]
                ## am getting some weird stuff for this
                if identifier.startswith("OMIM"):
                    # db = 'omim'
                    # identifier = identifier.split(':')[-1]
                    continue
                try:
                    db_name = get_name(db, identifier)
                except:
                    print(db, identifier)
                records.append(
                    {
                        "full_text": full_text,
                        "entity_raw_text": annotation["text"],
                        "entity_type": entity_type,
                        "db": db,
                        "identifier": identifier,
                        "normalized_name": db_name,
                    }
                )
    return records


if __name__ == "__main__":
    with open(f"{RESOURCE_PATH}/Train.BioC.JSON", mode="r") as f:
        jsn = json.load(f)
    records = extract_dataset(jsn=jsn)
    df = pl.from_records(records)
    df.write_csv(f"{RESOURCE_PATH}/BioRed_training_named.tsv", separator="\t")
