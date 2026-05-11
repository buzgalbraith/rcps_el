"""
Loader for BioID benchmark,
set method to original if want to directly ground
set method to gilda to load groundings directly used in Gilda paper.
"""

from .dataset import Dataset, pl, Path
from rcps_el.utils import safeMatch
import xml.etree.ElementTree as etree
import pystow
import os
from functools import lru_cache
import gilda
from rcps_el.utils.constants import BIOID_DIR
import logging
import ast
import re

logger = logging.getLogger(__name__)
from typing import TypedDict

gilda_terms_path = "~/.data/gilda/1.5.0/grounding_terms.tsv.gz"


class bioIDBenchmark(Dataset):
    name = "bioID"
    document_id_column = "don_article"
    title_column = "title"
    known_methods = ["original", "gilda"]
    mod = pystow.module("gilda", "biocreative")
    url = "https://github.com/buzgalbraith/BioCreative-VI-Track-1/raw/refs/heads/main/data/BioIDtraining_2.tar.gz"
    original_dataframe_path: Path = BIOID_DIR.joinpath("gilda_dataset.tsv")

    def __init__(
        self, seed: int = 100, split_size: float = 0.2, method: str = "original", original_dataframe_path: str = None
    ) -> None:
        self.method: str = method.lower().strip()
        assert (
            self.method in self.known_methods
        ), f"Method: {self.method} not available known methods for dataset {self.name} are {self.known_methods}"
        if self.method == "original":
            self.processed_dataframe_path = BIOID_DIR.joinpath(
                "processed_gilda_dataset.parquet"
            )
        elif self.method == "gilda":
            self.processed_dataframe_path: Path = BIOID_DIR.joinpath(
                "processed_gilda_dataset_gilda_paper.parquet"
            )
        super().__init__(seed, split_size, method, original_dataframe_path)

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

    def preprocess_dataset(self) -> pl.DataFrame:
        if self.method == "original":
            return self._preprocess_dataset_original()
        elif self.method == "gilda":
            return self._preprocess_dataset_gilda()

    ## methods for using the original raw bioID dataset ##

    def _preprocess_dataset_original(
        self,
    ) -> pl.DataFrame:
        """pre-process the dataset"""
        if os.path.exists(self.processed_dataframe_path):
            logger.info(
                f"Loading dataset from cache at {self.processed_dataframe_path}"
            )
            return pl.read_parquet(self.processed_dataframe_path)

        df = self.original_dataframe.with_columns(
            gilda_matches=pl.struct(["text", "don_article"]).map_elements(
                lambda x: self.get_gilda_candidates_bioid(x["text"], x["don_article"]),
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
            title=pl.col("don_article").map_elements(
                lambda x: self.get_plaintext(x, title_only=True), return_dtype=pl.String
            ),
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
        df.write_parquet(self.processed_dataframe_path)
        return df

    @lru_cache(maxsize=None)
    def get_plaintext(self, don_article: str, title_only: bool = False) -> str:
        """Get plaintext content from XML file in BioID corpus

        Parameters
        ----------
        don_article :
            Identifier for paper used within corpus.
        title_only:
            Returns only the title or first found text

        Returns
        -------
        :
            Plaintext of specified article
        """
        directory = self.mod.ensure_untar(url=self.url, directory="BioIDtraining_2")
        path = directory.joinpath(
            "BioIDtraining_2", "fulltext_bioc", f"{don_article}.xml"
        )
        tree = etree.parse(path.as_posix())
        paragraphs = tree.findall("//text")
        paragraphs = [" ".join(text.itertext()) for text in paragraphs]
        if not title_only:
            return "/n".join(paragraphs) + "/n"
        non_empty = [p for p in paragraphs if p.strip()]
        return non_empty[0] if non_empty else ""

    def get_gilda_candidates(self, text: str, context: str | None) -> list[safeMatch]:
        matches = gilda.ground(text=text, context=context)
        records = []
        for match in matches:
            records.append(
                {
                    "name": match.term.entry_name,
                    "curie": f"{match.term.db}:{match.term.id}",
                    "score": match.score,
                }
            )
        return records

    def get_gilda_candidates_bioid(
        self, text: str, don_article: str
    ) -> list[safeMatch]:
        context = self.get_plaintext(don_article)
        return self.get_gilda_candidates(text=text, context=context)

    ## methods for loading the BioID dataset with groundings directly from the gilda paper##
    def _preprocess_dataset_gilda(
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
            pl.col("groundings").map_elements(
                self.parse_grounding,  # pass the function directly, no lambda needed
                return_dtype=pl.List(
                    pl.Struct(
                        {"name": pl.String, "curie": pl.String, "score": pl.Float64}
                    )
                ),
            )
        ).with_columns(
            title=pl.col("don_article").map_elements(
                lambda x: self.get_plaintext(x, title_only=True), return_dtype=pl.String
            ),
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
        os.makedirs(self.processed_dataframe_path.parent, exist_ok=True)
        df.write_parquet(self.processed_dataframe_path)
        return df

    def parse_grounding(self, s):
        if s is None:
            return []
        matches = re.findall(r"\('([^']+)',\s*np\.float64\(([\deE.+-]+)\)", s)
        if matches:
            return [
                {"name": self.term_map.get(name), "curie": name, "score": float(val)}
                for name, val in matches
            ]
        return [
            {
                "name": self.term_map.get(name),
                "curie": name,
                "score": float(val),
            }
            for name, val in ast.literal_eval(s)
        ]

    def build_term_map(self):
        gilda_terms_df = pl.read_csv(gilda_terms_path, separator="\t")
        curie_to_term = (
            gilda_terms_df.with_columns(
                curie=pl.col("db") + ":" + pl.col("id"),
            )
            .drop_nulls(subset="curie")
            .group_by(["curie"])
            .first()
        )
        source_curie_to_term = (
            gilda_terms_df.with_columns(
                curie=pl.col("source_db") + ":" + pl.col("source_id"),
            )
            .drop_nulls(subset="curie")
            .group_by(["curie"])
            .first()
        )
        term_lookup = (
            curie_to_term.vstack(source_curie_to_term)
            .select(["curie", "entry_name", "db"])
            .group_by("curie")
            .first()
        )
        self.term_map = dict()
        for row in term_lookup.iter_rows(named=True):
            self.term_map[row.get("curie")] = row.get("entry_name")
