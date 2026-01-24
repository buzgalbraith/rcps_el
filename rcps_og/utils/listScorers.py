"""
Abstract class for scoring a list
"""
import polars as pl
import numpy as np

from transformers import AutoTokenizer, AutoModel
from rapidfuzz import fuzz


from typing import Callable, Optional
from abc import ABC, abstractmethod
import tqdm

class listScorer(ABC):
    """Abstract class for methods that score list results"""
    name:str = NotImplemented
    raw_text_col:str = NotImplemented
    entity_name_col:str = NotImplemented
    default_agg_method = NotImplemented
    def agg(self,  data_frame:pl.DataFrame,agg_method:Optional[Callable]=None)->pl.DataFrame:
        """run score list over data frame"""
        if agg_method is None:
            agg_method = self.default_agg_method
        self.processing_function(data_frame=data_frame)
        return data_frame.with_columns(
                 pl.struct([self.raw_text_col, self.entity_name_col]).map_elements(
                    lambda x: self.score_list(x[self.raw_text_col], x[self.entity_name_col], agg_method=agg_method), 
                    return_dtype=pl.Float64
                ).alias(f'{self.name}')
            )
    def processing_function(self,  data_frame:pl.DataFrame):
        """processing prior to running score function"""
        pass
    @abstractmethod
    def score_list(self,  entity, candidates, agg_method)->float:
        """Score the list its self"""
    def safe_max_score(self, scores:list[float])->float:
        if scores: ## evaluates to false if empty list  
            return max(scores)
        else: 
            return 0.0

class fuzzyStringScore(listScorer):
    raw_text_col = 'text'
    entity_name_col = 'gilda_names'
    default_agg_method = listScorer.safe_max_score
    name = 'fuzzy_string_score'
    def score_list(self,  entity:str, candidates:list[str], agg_method:Callable=default_agg_method)->float:
        scores = []
        for candidate_name in candidates:
            score = fuzz.ratio(entity, candidate_name) / 100
            scores.append(score)
        return agg_method(scores)
    def processing_function(self, data_frame: pl.DataFrame):
        return super().processing_function(data_frame)

class gildaScorer(listScorer):
    raw_text_col = 'text'
    entity_name_col = 'gilda_scores'
    default_agg_method = listScorer.safe_max_score
    name = 'gilda_score'
    def score_list(self,  entity:str, candidates:list[float], agg_method:Callable=default_agg_method)->float:
        return agg_method(candidates)
    def processing_function(self, data_frame: pl.DataFrame):
        return super().processing_function(data_frame)
   
class sapbertScorer(listScorer):
    raw_text_col = 'text'
    entity_name_col = 'gilda_names'
    default_agg_method = listScorer.safe_max_score
    name = 'SapBERT_score'
    batch_size = 128
    tokenizer = AutoTokenizer.from_pretrained(
    "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    )
    model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    name_to_embeddings:dict = {}
    def score_list(self,  entity:str, candidates:list[str], agg_method:Callable=default_agg_method)->float:
        if not candidates: ## deal with empty list case
            return 0.0
        scores = []
        entity_embedding = self.name_to_embeddings[entity]
        for candidate in candidates:
            if candidate:
                candidate_embedding = self.name_to_embeddings[candidate]
                normalized_sim = np.dot(entity_embedding, np.transpose(candidate_embedding)) / (np.linalg.norm(entity_embedding) * np.linalg.norm(candidate_embedding))
            else:
                normalized_sim = 0.0
            scores.append(normalized_sim)
        return agg_method(scores)
    
    def processing_function(self, data_frame: pl.DataFrame):
        self.name_to_embeddings = self.embed_names(data_frame)

    def embed_names(self, data_frame:pl.DataFrame)->dict[str, np.ndarray]:
        all_names = set(data_frame[self.raw_text_col].unique().to_list())
        all_names |= set(data_frame[self.entity_name_col].explode().drop_nulls().unique().to_list())
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
        return {
            all_names[i] : all_embeddings[i]
            for i in range(len(all_names))
        }

