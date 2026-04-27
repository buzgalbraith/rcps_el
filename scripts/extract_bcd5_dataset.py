from datasets import load_dataset
from bioregistry import normalize_curie
import polars as pl
from rcps_og.utils.constants import BCD5_DIR
import os

import logging

logger = logging.getLogger(__name__)


def process_split(split, dataset):
    logger.info(f"Processing {split}")
    records = []
    for x in dataset.get(split):
        passages = x.get("passages", [])
        ## get title should be in the first passage location but keep flexible i guess ## 
        title = 'missing_title'
        for passage in passages:
            if passage.get("type", '') == 'title':
                title = passage.get("text", title)
        ## extract entity mentions ## 
        for passage in passages:
            doc_id = passage.get("document_id", "missing_document_id")
            full_text = passage.get("text", "full_text_missing")
            for entity in passage.get("entities", []):
                text = entity.get("text", ["text_missing"])[0]
                entity_type = entity.get("type", "type_missing")
                grounding = entity.get("normalized", [{}])
                dbs = ";".join(set([y.get("db_name") for y in grounding]))
                names = ";".join(
                    set(
                        [
                            normalize_curie(f'{y.get("db_name")}:{y.get("db_id")}')
                            for y in grounding
                        ]
                    )
                )
                records.append(
                    {
                        "text": text,
                        "entity_type": entity_type,
                        "obj_synonyms": names,
                        "obj_dbs": dbs,
                        "document_id": doc_id,
                        "title": title,
                        "full_text": full_text,
                    }
                )
    df = pl.from_dicts(records)
    ## there are some terms that are not getting properly labeled
    df = df.filter(~pl.col("obj_dbs").eq(""))
    logger.info(f"Extracted {len(df)} named entities for {split} split.")
    split = split if split != "train" else "calibration"
    write_loc = os.path.join(BCD5_DIR, f"{split}_set.tsv")
    os.makedirs(BCD5_DIR, exist_ok=True)
    df.write_csv(write_loc, separator="\t")


def main():
    bcd5_data = load_dataset("bigbio/bc5cdr")
    for split in bcd5_data:
        process_split(split=split, dataset=bcd5_data)


if __name__ == "__main__":
    main()
