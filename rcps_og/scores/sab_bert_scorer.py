"""
Scorer class using cosine similarity of SAPBert Embeddings
"""

from .scorer import Scorer, pl
import numpy as np
from transformers import AutoTokenizer, AutoModel
import tqdm


class sapbertScorer(Scorer):
    raw_text_col = "text"
    entity_name_col = "match_names"
    name = "SapBERT_scores"
    batch_size = 128
    tokenizer = AutoTokenizer.from_pretrained(
        "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    )
    model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    name_to_embeddings: dict = {}

    def score_sample(self, entity: str, candidates: list[str]) -> list[float]:
        scores = []
        entity_embedding = self.name_to_embeddings[entity]
        for candidate in candidates:
            if candidate:
                candidate_embedding = self.name_to_embeddings[candidate]
                normalized_sim = np.dot(
                    entity_embedding, np.transpose(candidate_embedding)
                ) / (
                    np.linalg.norm(entity_embedding)
                    * np.linalg.norm(candidate_embedding)
                )
            else:
                normalized_sim = 0.0
            scores.append(normalized_sim)
        return scores

    def processing_function(self, data_frame: pl.DataFrame):
        self.name_to_embeddings = self.embed_names(data_frame)

    def embed_names(self, data_frame: pl.DataFrame) -> dict[str, np.ndarray]:
        all_names = set(data_frame[self.raw_text_col].unique().to_list())
        all_names |= set(
            data_frame[self.entity_name_col].explode().drop_nulls().unique().to_list()
        )
        all_names = list(all_names)
        all_embeddings = []
        for i in tqdm.tqdm(np.arange(0, len(all_names), self.batch_size)):
            toks = self.tokenizer.batch_encode_plus(
                all_names[i : i + self.batch_size],
                padding="max_length",
                max_length=25,
                truncation=True,
                return_tensors="pt",
            )
            all_embeddings.append(self.model(**toks)[0][:, 0, :].detach().numpy())
        all_embeddings = np.vstack(all_embeddings)
        return {all_names[i]: all_embeddings[i] for i in range(len(all_names))}
