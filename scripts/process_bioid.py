import polars as pl
from typing import Tuple
from functools import lru_cache
from lxml import etree
from rcps_og.utils.constants import CALIBRATION_DATA_PATH
from rcps_og.utils.utils import (
    get_gilda_predictions,
    get_gilda_prediction_stats,
    filter_calibration_data,
    calibration_evaluation_generator,
)
from rcps_og.utils.losses import binary_miscoverage_loss
from rcps_og.utils.scores import sapBert_score, get_Sapbert_embeddings
import pystow
import subprocess
import os
import numpy as np

MODULE = pystow.module("gilda", "biocreative")
URL = "https://github.com/buzgalbraith/BioCreative-VI-Track-1/raw/refs/heads/main/data/BioIDtraining_2.tar.gz"  # used to be ('https://biocreative.bioinformatics.udel.edu/media/store/files/2017/BioIDtraining_2.tar.gz')
BIOID_PATH = "~/.data/BioIDtraining_2/gilda_dataset.tsv"

import polars as pl 
import pystow
from functools import lru_cache
from lxml import etree
import gilda
from rapidfuzz import fuzz
from typing import Callable


MODULE = pystow.module("gilda", "biocreative")
URL = "https://github.com/buzgalbraith/BioCreative-VI-Track-1/raw/refs/heads/main/data/BioIDtraining_2.tar.gz"  # used to be ('https://biocreative.bioinformatics.udel.edu/media/store/files/2017/BioIDtraining_2.tar.gz')
BIOID_PATH = "~/.data/BioIDtraining_2/gilda_dataset.tsv"


def safe_max_score(scores:list[float]|list[None])->float:
    if len(scores) > 0:
        return max(scores)
    else:
        ## if there is no list
        return 0.0
def safe_min_loss(scores:list[float]|list[None])->float:
    if len(scores) > 0:
        return min(scores)
    else:
        ## if there is no list
        return 1.0


def fuzzy_string_score_list(entity: str, candidate_names: list[str], agg_method:Callable[[list[float]], float]=safe_max_score) -> float:
    """
    fuzzy string match a list of candidate names against a entity text. Aggregate scores with a function
    """
    scores = []
    for candidate_name in candidate_names:
        score = fuzz.ratio(entity, candidate_name) / 100
        scores.append(score)
    return agg_method(scores)

def binary_miscoverage_loss_list(labels: list[str], candidate_set: list[str], agg_method:Callable[[list[float]], float]=safe_min_loss) -> float:
    """

    """
    term_losses = [float(candidate not in labels) for candidate in candidate_set]
    return agg_method(term_losses)


@lru_cache(maxsize=None)
def get_plaintext(don_article: str) -> str:
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
    directory = MODULE.ensure_untar(url=URL, directory='BioIDtraining_2')
    path = directory.joinpath('BioIDtraining_2', 'fulltext_bioc',
                                f'{don_article}.xml')
    tree = etree.parse(path.as_posix())
    paragraphs = tree.xpath('//text')
    paragraphs = [' '.join(text.itertext()) for text in paragraphs]
    return '/n'.join(paragraphs) + '/n'

from typing import TypedDict
class mini_gilda_match(TypedDict):
    name: str
    curie: str
    score: float





def get_gilda_candidates(text:str, context:str|None)->list[mini_gilda_match]:
    matches = gilda.ground(
        text = text, 
        context=context
    )
    records = []
    for match in matches:
        records.append(
            {
                'name' : match.term.entry_name, 
                'curie' : f"{match.term.db}:{match.term.id}",
                "score" : match.score
            }
        )
    return records
def get_gilda_candidates_bioid(text:str, don_article:str)->list[mini_gilda_match]:
    context = get_plaintext(don_article)
    return get_gilda_candidates(text=text, context=context)



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



if __name__ == "__main__":

    original_bio_df = pl.read_csv(BIOID_PATH, separator="\t").with_columns(
        pl.col("obj_synonyms")
        .str.strip_prefix("{")
        .str.strip_suffix("}")
        .str.split(",")
    ).with_columns(
        pl.col("obj_synonyms").list.eval(pl.element().str.strip_chars("'' "))
    )
    bio_id_df = original_bio_df.sample(100)
    print(len(bio_id_df))

    # # here we are assuming that this will always hold
    bio_id_df = bio_id_df.with_columns(
        gilda_matches = pl.struct(["text", "don_article"]).map_elements(
        lambda x: get_gilda_candidates_bioid(x["text"], x["don_article"]),
        return_dtype=pl.List(pl.Struct({
            "name": pl.String,
            "curie": pl.String,
            "score": pl.Float64,
        })),
        )
        ).with_columns(
            gilda_names=pl.col("gilda_matches").list.eval(pl.element().struct.field("name")),
            gilda_curie=pl.col("gilda_matches").list.eval(pl.element().struct.field("curie")),
            gilda_scores=pl.col("gilda_matches").list.eval(pl.element().struct.field("score")),
        )
    bio_id_df = bio_id_df.with_columns(
        score = pl.struct(['text', 'gilda_names']).map_elements(
            lambda x: fuzzy_string_score_list(x['text'], x['gilda_names'], agg_method=safe_max_score), 
            return_dtype=pl.Float64
        )
    )
    bio_id_df = bio_id_df.with_columns(
        loss = pl.struct(['obj_synonyms', 'gilda_curie']).map_elements(
            lambda x: binary_miscoverage_loss_list(x['obj_synonyms'], x['gilda_curie'], agg_method=safe_min_loss), 
            return_dtype=pl.Float64
        )
    )
    ## there are still cases where gilda is getting mappings that we are not using a custom grounder ##
    bio_id_df = bio_id_df.with_columns(
        numeric_exists = pl.col('exists_correct').cast(dtype=pl.Float64),
    )
    bio_id_df.filter(pl.col('numeric_exists').eq(pl.col('loss'))).select(
        [
            'obj_synonyms',
            'groundings',
            'gilda_curie', 
            'numeric_exists',
            'loss'
        ]
    )