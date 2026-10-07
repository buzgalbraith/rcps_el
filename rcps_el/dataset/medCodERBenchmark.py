"""
MedCodER benchmark dataset (https://zenodo.org/records/13308316?preview_file=Readme.md)
evaluated with https://github.com/thomaslim6793/rag_grounder/tree/main
"""

from .dataset import Dataset, pl, Path
from rcps_el.utils.constants import MEDCODER_DIR


import os
import json
import logging

logger = logging.getLogger(__name__)
 
## There were some changes between the ICD10 version that was used to generate this data set and what ##
## is available to me, so used LLMs to get some simple mappings ##
LABEL_FIXES = {
    "G35": "G35.D",  # Multiple sclerosis -> MS, unspecified
    "S16.1A": "S16.1XXA",  # Neck strain
    "S13.9XA": "S13.9XXA",  # Neck sprain
    "S33.5A": "S33.5XXA",  # Lumbar spine strain
    "M1A.90": "M1A.9XX0",  # Gout -> chronic gout, unspecified, without tophus
    "N25.561": "M25.561",  # Right knee pain
    "834.11A": "S83.411A",  # Right knee MCL strain
    "S83.91A": "S83.91XA",  # Right knee sprain
}


def fix_labels(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("obj_synonyms").list.eval(pl.element().replace(LABEL_FIXES))
    )


class medCodERBenchmark(Dataset):
    name = "MedCodER"
    document_id_column = "doc_id"
    original_dataframe_path: Path = MEDCODER_DIR.joinpath(
        "retriever_only_ada002_billable_main.jsonl"
    )
    processed_dataframe_path: Path = MEDCODER_DIR.joinpath(
        "medcoder_billable_calibration.parquet"
    )
    known_methods = ["medcoder"]

    def __init__(
        self, seed=100, split_size=0.2, method="medcoder", original_dataframe_path=None
    ):
        self.method = method.lower().strip()
        assert (
            self.method in self.known_methods
        ), f"Method: {self.method} not available known methods for dataset {self.name} are {self.known_methods}"
        self.preprocess_dataset()

    def _extract_df(self, result_path: str) -> pl.DataFrame:
        records = []
        with open(result_path, mode="r") as f:
            for line in f.readlines():
                load = json.loads(line)
                doc_id = load["doc_id"]
                for m in load["mentions"]:
                    gold_code = [m.get("gold_code")]
                    mention = m.get("mention")
                    candidate_codes = []
                    candidate_scores = []
                    candidate_names = []
                    for x in m.get("retrieved"):
                        candidate_codes.append(x[0])
                        candidate_scores.append(x[1])
                        candidate_names.append(x[1])
                    records.append(
                        {
                            "doc_id": doc_id,
                            "text": mention,
                            "obj_synonyms": gold_code,
                            "match_names": candidate_names,
                            "match_curies": candidate_codes,
                            "match_scores": candidate_scores,
                        }
                    )
        df = pl.from_records(records)
        return df.with_row_index()

    def preprocess_dataset(self):
        split_map = {"main": "calibration", "holdout": "validation"}
        json_path_map = lambda x: MEDCODER_DIR.joinpath(
            f"retriever_only_ada002_billable_{x}.jsonl"
        )
        output_path_map = lambda x: MEDCODER_DIR.joinpath(
            f"medcoder_billable_{x}.parquet"
        )
        for split in split_map:
            logger.warning(f"Loading {split}")
            output_path = output_path_map(split_map[split])
            if not os.path.exists(output_path):
                json_path = json_path_map(split)
                df = self._extract_df(json_path)
                df.write_parquet(output_path)
                logger.warning(f"{json_path} extracted to {output_path}")
        self.calibration_set = fix_labels(pl.read_parquet(output_path_map("calibration")))
        self.validation_set = fix_labels(pl.read_parquet(output_path_map("validation")))

    def load_dataframe(self, dataframe_path=None):
        if not dataframe_path:
            return self.calibration_set
        return pl.read_parquet(dataframe_path)

class medCodERResplitBenchmark(medCodERBenchmark):
    def preprocess_dataset(self):
        path = lambda x: MEDCODER_DIR.joinpath(f"medcoder_resplit_billable_{x}.parquet")
        self.calibration_set = fix_labels(pl.read_parquet(path("calibration")))
        self.validation_set = fix_labels(pl.read_parquet(path("validation")))
