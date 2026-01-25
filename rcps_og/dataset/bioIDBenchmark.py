"""
Loader for BioID benchmark
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

logger = logging.getLogger(__name__)


class bioIDBenchmark(Dataset):
    name = "bioID"
    document_id_column = "don_article"
    mod = pystow.module("gilda", "biocreative")
    url = "https://github.com/buzgalbraith/BioCreative-VI-Track-1/raw/refs/heads/main/data/BioIDtraining_2.tar.gz"
    original_dataframe_path: Path = BIOID_DIR.joinpath("gilda_dataset.tsv")
    processed_dataframe_path: Path = BIOID_DIR.joinpath(
        "processed_gilda_dataset.parquet"
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
    def get_plaintext(self, don_article: str) -> str:
        """Get plaintext content from XML file in BioID corpus

        Parameters
        ----------
        don_article :
            Identifier for paper used within corpus.

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
        return "/n".join(paragraphs) + "/n"

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
