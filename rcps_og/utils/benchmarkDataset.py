"""
Abstract class for benchmarking dataset
"""
from typing import Tuple
from abc import ABC, abstractmethod
import polars as pl
from pathlib import Path
from typing import TypedDict
import pystow
import xml.etree.ElementTree as etree
import os
from bioregistry import normalize_curie
from functools import lru_cache
import gilda
from rcps_og.utils.constants import BIOID_DIR, BIORED_DIR
class mini_gilda_match(TypedDict):
    name: str
    curie: str
    score: float
class benchmarkDataset(ABC):
    """Abstract class for methods that score list results"""
    original_dataframe_path:Path = NotImplemented
    processed_dataframe_path:Path = NotImplemented
    document_id_column:str = NotImplemented
    def __init__(self, seed:int = 100, split_size:float = 0.2) -> None:
            self.seed: int = seed
            self.split_size: float = split_size
            self.original_dataframe:pl.DataFrame = self.load_dataframe(dataframe_path=self.original_dataframe_path)
            self.full_dataframe:pl.DataFrame = self.preprocess_dataset()
            self.validation_set, self.calibration_set = self.split_dataset()
    @abstractmethod
    def preprocess_dataset(self, )->pl.DataFrame:
        """pre-process the dataset"""
    @abstractmethod
    def load_dataframe(self,dataframe_path:Path|None = None)->pl.DataFrame:
         """load a version of the dataframe"""
    def split_dataset(self)->Tuple[pl.DataFrame, pl.DataFrame]:
        """split the calibration dataset based on some document id column"""
        ## TODO: not currently shuffling split
        documents = self.full_dataframe.select(self.document_id_column).unique()
        validate_size = int(self.split_size * len(documents))
        validation_document_ids, calibration_document_ids = documents.head(
            validate_size
        ), documents.tail(-validate_size)
        validation_set = self.full_dataframe.filter(
            pl.col(self.document_id_column).is_in(
                validation_document_ids.to_numpy().flatten().tolist()
            )
        ).with_row_index()
        calibration_set = self.full_dataframe.filter(
            pl.col(self.document_id_column).is_in(
                calibration_document_ids.to_numpy().flatten().tolist()
            )
        ).with_row_index()
        return validation_set, calibration_set
class bioIDBenchmark(benchmarkDataset):
    document_id_column = 'don_article'
    mod = pystow.module("gilda", "biocreative")
    url = "https://github.com/buzgalbraith/BioCreative-VI-Track-1/raw/refs/heads/main/data/BioIDtraining_2.tar.gz"
    original_dataframe_path:Path = BIOID_DIR.joinpath("gilda_dataset.tsv")
    processed_dataframe_path:Path = BIOID_DIR.joinpath("processed_gilda_dataset.parquet")
    def __init__(self, seed: int = 100, split_size: float = 0.2) -> None:
          super().__init__(seed, split_size)
    def load_dataframe(self,dataframe_path:Path|None = None)->pl.DataFrame:
        if dataframe_path is None:
             dataframe_path = self.original_dataframe_path
        return pl.read_csv(dataframe_path, separator="\t").with_columns(
            pl.col("obj_synonyms")
            .str.strip_prefix("{")
            .str.strip_suffix("}")
            .str.split(",")
        ).with_columns(
            pl.col("obj_synonyms").list.eval(pl.element().str.strip_chars("'' "))
        )
    def preprocess_dataset(self, )->pl.DataFrame:
        """pre-process the dataset"""
        if os.path.exists(self.processed_dataframe_path):
             return pl.read_parquet(self.processed_dataframe_path) 
        df = self.original_dataframe.with_columns(
        gilda_matches = pl.struct(["text", "don_article"]).map_elements(
        lambda x: self.get_gilda_candidates_bioid(x["text"], x["don_article"]),
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
        directory = self.mod.ensure_untar(url=self.url, directory='BioIDtraining_2')
        path = directory.joinpath('BioIDtraining_2', 'fulltext_bioc',
                                    f'{don_article}.xml')
        tree = etree.parse(path.as_posix())
        paragraphs = tree.findall('//text')
        paragraphs = [' '.join(text.itertext()) for text in paragraphs]
        return '/n'.join(paragraphs) + '/n'
    def get_gilda_candidates(self, text:str, context:str|None)->list[mini_gilda_match]:
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
    def get_gilda_candidates_bioid(self, text:str, don_article:str)->list[mini_gilda_match]:
        context = self.get_plaintext(don_article)
        return self.get_gilda_candidates(text=text, context=context)

class bioRedBenchmark(benchmarkDataset):
    document_id_column = 'document_id'
    original_dataframe_path:Path = BIORED_DIR.joinpath("BioRed_calibration.tsv")
    processed_dataframe_path:Path = BIORED_DIR.joinpath("processed_BioRed_calibration.parquet")
    def __init__(self, seed: int = 100, split_size: float = 0.2) -> None:
          super().__init__(seed, split_size)
    def load_dataframe(self,dataframe_path:Path|None = None)->pl.DataFrame:
        if dataframe_path is None:
             dataframe_path = self.original_dataframe_path
        return pl.read_csv(dataframe_path, separator="\t")
    def get_gilda_candidates(self, text:str, context:str|None)->list[mini_gilda_match]:
        matches = gilda.ground(
            text = text, 
            context=context
        )
        records = []
        for match in matches:
            records.append(
                {
                    'name' : match.term.entry_name, 
                    'curie' : normalize_curie(f"{match.term.db}:{match.term.id}"),
                    "score" : match.score
                }
            )
        return records
    def preprocess_dataset(self, )->pl.DataFrame:
        """pre-process the dataset"""
        if os.path.exists(self.processed_dataframe_path):
             return pl.read_parquet(self.processed_dataframe_path) 
        df = self.original_dataframe.with_columns(
        gilda_matches = pl.struct(["entity_raw_text", "full_text"]).map_elements(
        lambda x: self.get_gilda_candidates(x["entity_raw_text"], x["full_text"]),
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

            obj_synonyms=pl.struct(["db", "identifier"]).map_elements(
                lambda x: [normalize_curie(':'.join([x["db"], x["identifier"]]))],
                return_dtype=pl.List(pl.String)  
            )
        ).rename({'entity_raw_text' : 'text'})
        df.write_parquet(self.processed_dataframe_path)
        return df
    