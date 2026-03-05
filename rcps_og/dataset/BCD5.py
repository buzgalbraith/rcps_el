"""
Loader for BioID benchmark
"""

from .dataset import Dataset, pl, Path
from rcps_og.utils import safeMatch
import os
import gilda
from rcps_og.utils.constants import BCD5_DIR
import logging

logger = logging.getLogger(__name__)


class BCD5(Dataset):
    name = "BCD5"
    document_id_column = "document_id"

    def __init__(self, seed: int = 100, split_size: float = 0.2) -> None:
        """going to need to do separate initialization"""
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
        self.calibration_set = self.preprocess_split('calibration')
        logger.info("Loading validation set...")
        self.validation_set = self.preprocess_split('validation')
        logger.info("Loading test set...")
        self.test_set = self.preprocess_split('test')
        return self.calibration_set
    def preprocess_split(
        self, split:str
    ) -> pl.DataFrame:
        """pre-process the dataset"""
        load_path = BCD5_DIR.joinpath(f"{split}_set.tsv")
        write_path = BCD5_DIR.joinpath(
        f"processed_{split}_set.parquet"
        )
        if os.path.exists(write_path):
            logger.info(
                f"Loading dataset from cache at {write_path}"
            )
            return pl.read_parquet(write_path)
        df = self.load_dataframe(load_path).with_columns(
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
        ).with_columns(
            match_names=pl.col("gilda_matches").list.eval(
                pl.element().struct.field("name")
            ),
            match_curies=pl.col("gilda_matches").list.eval(
                pl.element().struct.field("curie")
            ),
            gilda_scores=pl.col("gilda_matches").list.eval(
                pl.element().struct.field("score")
            ),
        ).with_row_index()
        df.write_parquet(write_path)
        return df

    def get_gilda_candidates(self, text: str, context: str | None) -> list[safeMatch]:
        matches = gilda.ground(text=text, context=context, namespaces=['MESH'])
        records = []
        for match in matches:
            ## make sure that we get the mapping to mesh specifically
            mesh_id = next(id_ for db, id_ in match.get_groundings() if db == 'MESH')
            records.append(
                {
                    "name": match.term.entry_name,
                    "curie": f"mesh:{mesh_id}",
                    "score": match.score,
                }
            )
        return records

    def get_gilda_candidates_bcd5(
        self, text: str, full_text: str
    ) -> list[safeMatch]:
        return self.get_gilda_candidates(text=text, context=full_text)


