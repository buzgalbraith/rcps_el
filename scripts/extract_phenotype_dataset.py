import polars as pl
from pyobo import get_name
from rcps_og.utils.constants import RESOURCE_PATH


def main(df=pl.DataFrame):
    """
    processes the raw data frame
    """
    records = []
    for row in df.iter_rows(named=True):
        full_text = row["Text"]
        split_span = row["Spans"].split("-", maxsplit=3)
        if len(split_span) == 2:
            entity_raw_text = full_text[int(split_span[0]) : int(split_span[1])]
        entity_type = "phenotype"
        db = "hp"
        identifier = row["Term"]
        if identifier != "NA":
            if identifier.startswith("HP"):
                identifier = identifier.split(":")[1]
            normalized_name = get_name(db, identifier)
            records.append(
                {
                    "full_text": full_text,
                    "entity_raw_text": entity_raw_text,
                    "entity_type": entity_type,
                    "db": db,
                    "identifier": identifier,
                    "normalized_name": normalized_name,
                    "document_id": "id_missing",
                }
            )
    return pl.from_records(records)


if __name__ == "__main__":
    ## Can download from git here https://github.com/Ian-Campbell-Lab/Clinical-Genetics-Training-Data/blob/main/Physical-Exam/Clinical-Genetics-Physical-Exam-Data-Train.tsv
    df = pl.read_csv("Clinical-Genetics-Physical-Exam-Data-Train.tsv", separator="\t")
    processed_df = main(df=df)
    processed_df.write_csv(f"{RESOURCE_PATH}/phenotype_train.tsv", separator="\t")
