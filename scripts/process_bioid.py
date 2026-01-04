import polars as pl
from typing import Tuple
from functools import lru_cache
from lxml import etree
from rcps_og.utils.constants import CALIBRATION_DATA_PATH
from rcps_og.utils.utils import (
    get_gilda_predictions,
)
import pystow
import subprocess
import os

MODULE = pystow.module("gilda", "biocreative")
URL = "https://github.com/buzgalbraith/BioCreative-VI-Track-1/raw/refs/heads/main/data/BioIDtraining_2.tar.gz"  # used to be ('https://biocreative.bioinformatics.udel.edu/media/store/files/2017/BioIDtraining_2.tar.gz')
BIOID_PATH = "~/.data/BioIDtraining_2/gilda_dataset.tsv"


def get_gilda_terms_df() -> pl.DataFrame:
    gilda_compressed_terms_path = (
        "/Users/buzgalbraith/.data/gilda/1.4.1/grounding_terms.tsv.gz"
    )
    gilda_terms_path = "/Users/buzgalbraith/.data/gilda/1.4.1/grounding_terms.tsv"
    if not os.path.exists(gilda_terms_path):
        cmd = ["gunzip", "-k", gilda_compressed_terms_path]
        subprocess.run(cmd)
    ungrouped_gilda_terms_df = pl.read_csv(gilda_terms_path, separator="\t")
    return (
        ungrouped_gilda_terms_df.group_by(["db", "id"])
        .agg(pl.col("norm_text"))
        .with_columns(curie=pl.col("db") + ":" + pl.col("id"))
        .select(["curie", "norm_text"])
    )


def split_calibration_and_validation(
    base_df_path: str = CALIBRATION_DATA_PATH,
    base_df: pl.DataFrame = None,
    split_col="document_id",
    validate_prop: float = 0.2,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """splits a base dataset into calibration and validation sets. can either load from memory or take a base df as input"""
    if base_df is None:
        base_df = pl.read_csv(base_df_path, separator="\t")
    documents = base_df.select(split_col).unique()
    validate_size = int(validate_prop * len(documents))
    validation_document_ids, calibration_document_ids = documents.head(
        validate_size
    ), documents.tail(-validate_size)
    validation_df = base_df.filter(
        pl.col(split_col).is_in(validation_document_ids.to_numpy().flatten().tolist())
    ).with_row_index()
    calibration_df = base_df.filter(
        pl.col(split_col).is_in(calibration_document_ids.to_numpy().flatten().tolist())
    ).with_row_index()
    return calibration_df.sort(by="index"), validation_df.sort(by="index")


@lru_cache(maxsize=None)
def get_bioid_plaintext(don_article: str) -> str:
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
    directory = MODULE.ensure_untar(url=URL, directory="BioIDtraining_2")
    path = directory.joinpath("BioIDtraining_2", "fulltext_bioc", f"{don_article}.xml")
    tree = etree.parse(path.as_posix())
    paragraphs = tree.xpath("//text")
    paragraphs = [" ".join(text.itertext()) for text in paragraphs]
    return "/n".join(paragraphs) + "/n"


def get_bioid_df():
    return (
        pl.read_csv(BIOID_PATH, separator="\t")
        .with_columns(
            pl.col("obj_synonyms")
            .str.strip_prefix("{")
            .str.strip_suffix("}")
            .str.split(",")
        )
        .with_columns(
            pl.col("obj_synonyms").list.eval(pl.element().str.strip_chars("'' "))
        )
        .with_columns(
            full_text=pl.col("don_article").map_elements(
                get_bioid_plaintext, return_dtype=pl.String
            ),
        )
        .rename(
            {
                "don_article": "document_id",
                "text": "entity_raw_text",
            }
        )
        .with_row_index()
    )


def expand_gilda_names(bioid, gilda_terms) -> pl.DataFrame:
    boom = bioid.explode("obj_synonyms")
    bam = boom.join(
        gilda_terms,
        left_on=pl.col("obj_synonyms"),
        right_on=pl.col("curie"),
        how="left",
        validate="m:1",
    )
    pow = bam.group_by("index", maintain_order=True).agg(
        pl.col("norm_text").flatten().drop_nulls().unique()
    )
    processed_df = bioid.join(pow, on="index", how="inner", validate="1:1")
    processed_df.drop_in_place("index")
    return processed_df


if __name__ == "__main__":
    ## load the df and do some initial processing
    bioid_df = get_bioid_df()
    gilda_terms_df = get_gilda_terms_df()
    processed_df = expand_gilda_names(bioid=bioid_df, gilda_terms=gilda_terms_df)
    calibration_df, validation_df = split_calibration_and_validation(
        base_df=processed_df,
        split_col="document_id",
    )
    ## there are still some terms that have no like normalized texts
    processed_df.filter(pl.col("norm_text").list.len() < 1)

    index_to_candidates_map = get_gilda_predictions(calibration_df=calibration_df)
