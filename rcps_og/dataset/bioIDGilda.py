"""
Loaded for the BioRed dataset with the groundings directly from the Gilda Paper.
"""

from .dataset import Dataset, pl, Path
from rcps_og.utils import safeMatch
import xml.etree.ElementTree as etree
import pystow
import os
from functools import lru_cache
import gilda
from rcps_og.utils.constants import BIOID_DIR
import logging
import re
import ast

logger = logging.getLogger(__name__)

gilda_terms_path = '~/.data/gilda/1.5.0/grounding_terms.tsv.gz'

class bioIDGildaBenchmark(Dataset):
    name = "bioIDGilda"
    document_id_column = "don_article"
    mod = pystow.module("gilda", "biocreative")
    url = "https://github.com/buzgalbraith/BioCreative-VI-Track-1/raw/refs/heads/main/data/BioIDtraining_2.tar.gz"

    original_dataframe_path: Path = BIOID_DIR.joinpath("gilda_dataset.tsv")
    processed_dataframe_path: Path = BIOID_DIR.joinpath(
        "processed_gilda_dataset_gilda_paper.parquet"
    )

    def __init__(self, seed: int = 100, split_size: float = 0.2) -> None:
        super().__init__(seed, split_size)

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
        if os.path.exists(self.processed_dataframe_path):
            logger.info(
                f"Loading dataset from cache at {self.processed_dataframe_path}"
            )
            return pl.read_parquet(self.processed_dataframe_path)
        self.build_term_map()
        df = self.original_dataframe.with_columns(
            pl.col("groundings")
            .map_elements(
                self.parse_grounding,  # pass the function directly, no lambda needed
                return_dtype=pl.List(pl.Struct({"name": pl.String, "curie": pl.String, "score": pl.Float64}))

            )
        ).with_columns(
                match_names=pl.col("groundings").list.eval(
                    pl.element().struct.field("name")
                ),
                match_curies=pl.col("groundings").list.eval(
                    pl.element().struct.field("curie")
                ),
                gilda_scores=pl.col("groundings").list.eval(
                    pl.element().struct.field("score")
                ),
            )
        df.write_parquet(self.processed_dataframe_path)
        return df
    def parse_grounding(self, s):
        if s is None:
            return []
        matches = re.findall(r"\('([^']+)',\s*np\.float64\(([\deE.+-]+)\)", s)
        if matches:
            return [{'name' : self.term_map.get(name), "curie": name, "score": float(val)} for name, val in matches]
        return [{'name' : self.term_map.get(name), "curie": name, "score": float(val),} for name, val in ast.literal_eval(s)]

    def build_term_map(self):
        gilda_terms_df = pl.read_csv(gilda_terms_path, separator='\t')
        curie_to_term = gilda_terms_df.with_columns(
            curie = pl.col('db') + ':' + pl.col('id'),
        ).drop_nulls(subset='curie').group_by(['curie']).first()
        source_curie_to_term = gilda_terms_df.with_columns(
            curie = pl.col('source_db') + ':' + pl.col('source_id'),
        ).drop_nulls(subset='curie').group_by(['curie']).first()
        term_lookup = curie_to_term.vstack(source_curie_to_term).select(['curie', 'entry_name', 'db']).group_by('curie').first()
        self.term_map = dict()
        for row in term_lookup.iter_rows(named=True):
            self.term_map[row.get('curie')] = row.get('entry_name')
