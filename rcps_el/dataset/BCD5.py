"""
Loader for BioID benchmark
"""

from .dataset import Dataset, pl, Path
from rcps_el.utils import safeMatch
import os
import gilda
from rcps_el.utils.constants import BCD5_DIR, KRISSBERT_DIR
from bioregistry import normalize_curie
from datasets import load_dataset
import json
from indra.databases import mesh_client
import logging

logger = logging.getLogger(__name__)
gilda_terms_path = "~/.data/gilda/1.5.0/grounding_terms.tsv.gz"


class BCD5(Dataset):
    name = "BCD5"
    document_id_column = "document_id"
    title_column = "title"
    known_methods = ["gilda", "krissbert"]
    normalization_parameters: tuple[float] | None = None

    def __init__(
        self, seed: int = 100, split_size: float = 0.2, method: str = "gilda", original_dataframe_path: str = None
    ) -> None:
        """going to need to do separate initialization"""
        self.method = method.lower().strip()
        assert (
            self.method in self.known_methods
        ), f"Method: {self.method} not available known methods for dataset {self.name} are {self.known_methods}"
        self.preprocess_dataset()

    def load_dataframe(self, dataframe_path: Path | None = None) -> pl.DataFrame:
        if dataframe_path is None:
            dataframe_path = self.original_dataframe_path
        return (
            pl.read_csv(dataframe_path, separator="\t")
            .with_columns(
                pl.col("obj_synonyms")
                .str.strip_prefix("{")
                .str.strip_suffix("}")
                .str.split(",")
            )
            .with_columns(
                pl.col("obj_synonyms").list.eval(pl.element().str.strip_chars("'' "))
            )
        )

    def preprocess_dataset(
        self,
    ) -> pl.DataFrame:
        """pre-process the dataset"""
        logger.info("Loading calibration set...")
        self.calibration_set = self.preprocess_split("calibration")
        logger.info("Loading validation set...")
        self.validation_set = self.preprocess_split("validation")
        logger.info("Loading test set...")
        self.test_set = self.preprocess_split("test")
        ## KrissBert used the training set to train the model so we need to calibrate and evaluate on the validation and test sets
        if self.method == "krissbert":
            self.model_train_set = self.calibration_set
            self.calibration_set = self.validation_set
            self.validation_set = self.test_set

        return self.calibration_set

    def preprocess_split(self, split: str) -> pl.DataFrame:
        """pre-process the dataset"""
        write_path = BCD5_DIR.joinpath(f"processed_{split}_{self.method}.parquet")
        if os.path.exists(write_path):
            logger.info(f"Loading dataset from cache at {write_path}")
            return pl.read_parquet(write_path)
        if self.method == "gilda":
            df = self.gilda_process(split)
        elif self.method == "krissbert":
            df = self.krissbert_process(split)
        else:
            raise ValueError(
                f"Method: {self.method} not available known methods for dataset {self.name} are {self.known_methods}"
            )
        logger.info(f"Writing results to {write_path}")
        df.write_parquet(write_path)
        return df

    def krissbert_process(self, split):
        """
        Load krissbert predictions for dataset
        """
        bcd5_data = load_dataset(
            "bigbio/bc5cdr", split=split if split != "calibration" else "train"
        )
        logger.info(f"extracting {split} data...")
        load_path = KRISSBERT_DIR.joinpath(f"bc5cdr_{split}.json")
        bcd5_split = self._load_bcd5_fulltext_split(dataset=bcd5_data)
        with open(load_path, mode="r") as f:
            jsn = json.load(f)
        krissbert_df = self._load_kirssbert_split(jsn)
        merged_split = krissbert_df.join(
            bcd5_split,
            on=["document_id", "text", "offsets"],
            how="left",
            validate="1:1",
        ).with_row_index()
        return self._krissbert_normalize(merged_dataset=merged_split)

    def gilda_process(self, split):
        """
        Predict on dataset with Gilda.
        """
        load_path = BCD5_DIR.joinpath(f"{split}_set.tsv")
        df = (
            self.load_dataframe(load_path)
            .with_columns(
                gilda_matches=pl.struct(["text", "full_text"]).map_elements(
                    lambda x: self.get_gilda_candidates_bcd5(x["text"], x["full_text"]),
                    return_dtype=pl.List(
                        pl.Struct(
                            {
                                "name": pl.String,
                                "curie": pl.String,
                                "score": pl.Float64,
                            }
                        )
                    ),
                )
            )
            .with_columns(
                match_names=pl.col("gilda_matches").list.eval(
                    pl.element().struct.field("name")
                ),
                match_curies=pl.col("gilda_matches").list.eval(
                    pl.element().struct.field("curie")
                ),
                gilda_scores=pl.col("gilda_matches").list.eval(
                    pl.element().struct.field("score")
                ),
            )
            .with_row_index()
        )
        return df

    def _krissbert_normalize(self, merged_dataset: pl.DataFrame):
        """
        Normalize the krissbert score to be between zero and one and add a column marking if a value is a short circuit (ie not scored with Kirssbert)
        """
        if self.normalization_parameters is None:
            ## if there are no normalization parameters yet assume we are currently on the calibration set and get them ##
            tmp_df = merged_dataset.with_columns(
                maxs=pl.col("match_scores").list.max(),
                mins=pl.col("match_scores").list.min(),
            )
            self.normalization_parameters = (tmp_df["mins"].min(), tmp_df["maxs"].max())
        return merged_dataset.with_columns(
            short_circuit=pl.col("match_scores").list.eval(pl.element().is_null()),
            match_scores=pl.col("match_scores").list.eval(
                pl.when(pl.element().is_null())
                .then(1.0)
                .otherwise(
                    (pl.element() - self.normalization_parameters[0])
                    / (
                        self.normalization_parameters[1]
                        - self.normalization_parameters[0]
                    )
                )
            ),
        )

    def _load_bcd5_fulltext_split(self, dataset):
        """load same split data from BCD5 to get full text"""
        records = []
        for x in dataset:
            passages = x.get("passages", [])
            ## get title should be in the first passage location but keep flexible i guess ##
            title = "missing_title"
            for passage in passages:
                if passage.get("type", "") == "title":
                    title = passage.get("text", title)
            for passage in passages:
                doc_id = passage.get("document_id", "missing_document_id")
                full_text = passage.get("text", "full_text_missing")
                for entity in passage.get("entities", []):
                    text = entity.get("text", ["text_missing"])[0]
                    offsets = entity.get("offsets", ["offsets"])[0]
                    records.append(
                        {
                            "text": text,
                            "offsets": offsets,
                            "document_id": doc_id,
                            "title": title,
                            "full_text": full_text,
                        }
                    )
        ## enforce uniqueness just in case ##
        return pl.from_dicts(records).unique()

    def _load_kirssbert_split(self, jsn):
        """
        Load precomputed predictions from KrissBert
        """
        records = []
        for record in jsn:
            text = record.get("text")
            document_id = record.get("document_id")
            etype = record.get("type")[0]
            db_ids = record.get("db_ids")[0]
            offsets = record.get("offsets")[0]
            raw_candidates = record.get("candidates")
            obj_synonyms = [normalize_curie(db_ids)]
            candidate_scores = [x["score"] for x in raw_candidates]
            candidate_curies = [normalize_curie(x["cuis"][0]) for x in raw_candidates]
            candidate_names = [
                mesh_client.get_mesh_name(x.upper().removeprefix("MESH:"))
                for x in candidate_curies
            ]
            assert len(candidate_curies) == len(
                candidate_names
            ), f"Have {len(candidate_curies)} candidate curies and {len(candidate_names)} candidate names!"
            records.append(
                {
                    "text": text,
                    "document_id": document_id,
                    "offsets": offsets,
                    "entity_type": etype,
                    "obj_synonyms": obj_synonyms,
                    "match_curies": candidate_curies,
                    "match_names": candidate_names,
                    "match_scores": candidate_scores,
                }
            )
        return pl.from_dicts(records)

    def get_gilda_candidates(self, text: str, context: str | None) -> list[safeMatch]:
        matches = gilda.ground(text=text, context=context, namespaces=["MESH"])
        records = []
        for match in matches:
            ## make sure that we get the mapping to mesh specifically
            mesh_id = next(id_ for db, id_ in match.get_groundings() if db == "MESH")
            records.append(
                {
                    "name": match.term.entry_name,
                    "curie": f"mesh:{mesh_id}",
                    "score": match.score,
                }
            )
        return records

    def get_gilda_candidates_bcd5(self, text: str, full_text: str) -> list[safeMatch]:
        return self.get_gilda_candidates(text=text, context=full_text)
