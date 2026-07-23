"""
MedCodER benchmark dataset (https://zenodo.org/records/13308316?preview_file=Readme.md)
evaluated with https://github.com/thomaslim6793/rag_grounder/tree/main
"""

import json
import logging
import os

import pystow

from .dataset import Dataset, Path, pl

logger = logging.getLogger(__name__)
module = pystow.module("medcoder")


class medCodERBenchmark(Dataset):
    document_id_column = "doc_id"
    original_dataframe_path: Path = module.base.joinpath(
        "retriever_only_ada002_billable_main.jsonl"
    )
    processed_dataframe_path: Path = module.base.joinpath(
        "medcoder_billable_calibration.parquet"
    )
    known_methods = ["medcoder-retrieve", "medcoder-rerank"]

    def __init__(
        self,
        seed: int = 100,
        method: str = "medcoder-retrieve",
        billable: bool = True,
        resplit: bool = False,
        n_retrieved: int = 20,
    ):
        """
        Arguments:
            - seed (int): seed used when resplitting
            - method (str): grounding method, must be one of ``known_methods``
            - billable (bool): use billable codes only if True, otherwise full dataset
            - resplit (bool): if False use original splits, otherwise evenly split
            - n_retrieved (int): number of retrieved candidates to consider
        """
        self.method = method.lower().strip()
        assert (
            self.method in self.known_methods
        ), f"Method: {self.method} not available known methods for dataset {self.name} are {self.known_methods}"
        self.seed = seed
        self.billable_str = "billable" if billable else "full"
        self.n_retrieved = n_retrieved
        if self.method.startswith("medcoder"):
            self.mode = self.method.split("-")[1]
        else:
            self.mode = 'retrieve'
        self.resplit = resplit
        self.resplit_str = "_resplit" if self.resplit else ""
        self.name = f"MedCodER{self.resplit_str}_{self.billable_str}_{self.n_retrieved}_candidates"

        self.preprocess_dataset()

    def _set_n_retrieved(self, input: pl.DataFrame) -> pl.DataFrame:
        """Filter a MedCoder dataset to a fixed number of retrieved candidates per sample."""
        return input.with_columns(
            pl.col("match_names").list.slice(0, self.n_retrieved),
            pl.col("match_curies").list.slice(0, self.n_retrieved),
            pl.col("match_scores").list.slice(0, self.n_retrieved),
        )

    def _extract_df(self, result_path: str) -> pl.DataFrame:
        """Extract the dataset from raw json and return it."""
        records = []
        with open(result_path, mode="r") as f:
            for line in f:
                load = json.loads(line)
                doc_id = load["doc_id"]
                for m in load["mentions"]:
                    candidate_codes = []
                    candidate_scores = []
                    candidate_names = []
                    rerank_scores = []
                    rerank_codes = []
                    rerank_names = []
                    for code, score, name in m.get("retrieved"):
                        candidate_codes.append(code)
                        candidate_scores.append(score)
                        candidate_names.append(name)
                    for code, score, name in m.get("reranked"):
                        rerank_codes.append(code)
                        rerank_scores.append(score)
                        rerank_names.append(name)
                    records.append(
                        {
                            "doc_id": doc_id,
                            "text": m.get("mention"),
                            "obj_synonyms": [m.get("gold_code")],
                            "match_names": candidate_names,
                            "match_curies": candidate_codes,
                            "match_scores": candidate_scores,
                            "rerank_names": rerank_names,
                            "rerank_curies": rerank_codes,
                            "rerank_scores": rerank_scores,
                        }
                    )
        return pl.from_records(records).with_row_index()

    def _resplit(
        self, unsplit_path: str, calibration_path: Path, validation_path: Path
    ) -> None:
        """Evenly split the unsplit dataset by document id and cache each half."""
        unsplit_df = self._extract_df(unsplit_path)
        doc_ids = (
            unsplit_df.get_column("doc_id").unique().sort().shuffle(seed=self.seed)
        )
        half = doc_ids.len() // 2
        left_ids = doc_ids.slice(0, half)
        left_mask = unsplit_df.get_column("doc_id").is_in(left_ids)
        unsplit_df.filter(left_mask).write_parquet(calibration_path)
        unsplit_df.filter(~left_mask).write_parquet(validation_path)

    def preprocess_dataset(self):
        """Load the processed dataset from cache, extracting from raw json if needed."""
        json_path_map = lambda x: module.base.joinpath(
            f"retriever_only_ada002_{self.billable_str}_{x}.jsonl"
        )
        if not self.resplit:
            output_path_map = lambda x: module.base.joinpath(
                f"medcoder_{self.billable_str}_{x}.parquet"
            )
            split_map = {"main": "calibration", "holdout": "validation"}
            for split, output_split in split_map.items():
                logger.warning(f"Loading {split}")
                output_path = output_path_map(output_split)
                if not os.path.exists(output_path):
                    json_path = json_path_map(split)
                    self._extract_df(json_path).write_parquet(output_path)
                    logger.warning(f"{json_path} extracted to {output_path}")
        else:
            output_path_map = lambda x: module.base.joinpath(
                f"medcoder_resplit_{self.billable_str}_{x}.parquet"
            )
            calibration_path = output_path_map("calibration")
            validation_path = output_path_map("validation")
            if not (
                os.path.exists(calibration_path) and os.path.exists(validation_path)
            ):
                json_path = json_path_map("unsplit")
                logger.warning(f"Resplitting {json_path}")
                self._resplit(json_path, calibration_path, validation_path)
                logger.warning(
                    f"{json_path} resplit into {calibration_path} and {validation_path}"
                )
        self.calibration_set = pl.read_parquet(output_path_map("calibration"))
        self.validation_set = pl.read_parquet(output_path_map("validation"))
        if self.mode == 'rerank':
            self.calibration_set = self.calibration_set.drop(['match_names', 'match_curies', 'match_scores']).rename({"rerank_names" : 'match_names', "rerank_curies": 'match_curies', "rerank_scores":  'match_scores'})
            self.validation_set = self.validation_set.drop(['match_names', 'match_curies', 'match_scores']).rename({"rerank_names" : 'match_names', "rerank_curies": 'match_curies', "rerank_scores":  'match_scores'})
        ## get only k ## 
        self.calibration_set = self._set_n_retrieved(self.calibration_set)
        self.validation_set = self._set_n_retrieved(self.validation_set)
    def load_dataframe(self, dataframe_path=None):
        """Helper for loading: return the calibration set, or a parquet file if given."""
        if not dataframe_path:
            return self.calibration_set
        return pl.read_parquet(dataframe_path)
