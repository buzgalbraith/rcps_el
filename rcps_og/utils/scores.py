"""
Scoring functions. Likely want to make these instances of a class.
"""

from rapidfuzz import fuzz
from transformers import AutoTokenizer, AutoModel
import numpy as np

tokenizer = AutoTokenizer.from_pretrained(
    "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
)
model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")


def gilda_score(entity: dict, candidate: dict) -> float:
    """simple score that just takes gilda's estimate"""
    return candidate.get("gilda_score", None)


def fuzzy_string_score(entity: dict, candidate: dict) -> float:
    """
    fuzzy string matching score.
    """
    x = entity.get("entity_raw_text", None)
    y = candidate.get("entry_name", None)
    if (x is None) or (y is None):
        return 0
    else:
        return fuzz.ratio(x, y) / 100


def sapBert_score(entity: dict, candidate: dict) -> float:
    """
    basic similarity score between sap bert embeddings using cosine sim
    """
    x = entity.get("entity_raw_text", None)
    y = candidate.get("entry_name", None)
    if (x is None) or (y is None):
        return 0
    else:
        x_token = tokenizer.encode(x, return_tensors="pt")
        x_embed = model(x_token)[0][:, 0, :].detach().numpy()
        y_token = tokenizer.encode(y, return_tensors="pt")
        y_embed = model(y_token)[0][:, 0, :].detach().numpy()
        cosin_sim = lambda a, b: (
            np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b))
        ).item()
        return cosin_sim(x_embed, y_embed)
