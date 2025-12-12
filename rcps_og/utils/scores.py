"""
Scoring functions. Likely want to make these instances of a class.
"""

from rapidfuzz import fuzz
from transformers import AutoTokenizer, AutoModel
import numpy as np
import polars as pl
import tqdm

tokenizer = AutoTokenizer.from_pretrained(
    "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
)
model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")


def gilda_score(entity: dict, candidate: dict) -> float:
    """simple score that just takes gilda's estimate"""
    return candidate.get("gilda_score", None)


def fuzzy_string_score(entity: str, candidate: str) -> float:
    """
    fuzzy string matching score.
    """
    if (entity is None) or (candidate is None):
        return 0
    else:
        return fuzz.ratio(entity, candidate) / 100


def sapBert_score_direct(entity: dict, candidate: dict) -> float:
    """
    basic similarity score between sap bert embeddings using cosine sim
    """
    if type(entity) is dict:
        x = entity.get("entity_raw_text", None)
    else:
        x = entity
    if type(candidate) is dict:
        y = candidate.get("entry_name", None)
    else:
        y = candidate
    if (x is None) or (y is None):
        return 0
    else:
        try:
            x_token = tokenizer.encode(x, return_tensors="pt")
        except:
            print(x)
        x_embed = model(x_token)[0][:, 0, :].detach().numpy()
        y_token = tokenizer.encode(y, return_tensors="pt")
        y_embed = model(y_token)[0][:, 0, :].detach().numpy()
        cosin_sim = lambda a, b: (
            np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b))
        ).item()
        return cosin_sim(x_embed, y_embed)


def get_Sapbert_embeddings(df: pl.DataFrame) -> pl.DataFrame:
    """get embeddings for all raw text and candide names from Sapbert"""
    bs = 128
    all_names = (
        df["entity_raw_text"].unique().to_list()
        + df["entry_name"].drop_nulls().unique().to_list()
    )
    all_names = list(set(all_names))  ## ensure unique
    all_embeddings = []
    for i in tqdm.tqdm(np.arange(0, len(all_names), bs)):
        toks = tokenizer.batch_encode_plus(
            all_names[i : i + bs],
            padding="max_length",
            max_length=25,
            truncation=True,
            return_tensors="pt",
        )
        all_embeddings.append(model(**toks)[0][:, 0, :].detach().numpy())
    all_embeddings = np.vstack(all_embeddings)
    embedng_df = pl.from_records(
        data=[
            all_names,
            all_embeddings,
        ],
        schema=["name", "embedding"],
    )
    return (
        df.join(
            embedng_df,
            left_on="entity_raw_text",
            right_on="name",
            how="left",
            validate="m:1",
        )
        .join(
            embedng_df,
            left_on="entry_name",
            right_on="name",
            how="left",
            suffix="_entry_name",
            validate="m:1",
        )
        .rename({"embedding": "embedding_entity_raw_text"}),
        "embedding_entity_raw_text",
        "embedding_entry_name",
    )


def sapBert_score(entity, candidate):
    """Sapbert score which is optimized to run as part of the pipeline requires the get_Sapbert_embeddings processing pipeline"""
    if entity is None or candidate is None:
        return 0
    return np.dot(entity, np.transpose(candidate)) / (
        np.linalg.norm(entity) * np.linalg.norm(candidate)
    )
