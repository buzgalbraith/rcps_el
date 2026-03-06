"""
Abstract class for RCPS datasets
"""

from typing import Tuple
from abc import ABC, abstractmethod
import polars as pl
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Dataset(ABC):
    """Abstract class for methods that score list results"""

    original_dataframe_path: Path = NotImplemented
    processed_dataframe_path: Path = NotImplemented
    document_id_column: str = NotImplemented
    name: str = NotImplemented
    known_methods: list = NotImplemented

    def __init__(
        self, seed: int = 100, split_size: float = 0.2, method: str = "gilda"
    ) -> None:
        self.seed: int = seed
        self.method: str = method.lower().strip()
        self.split_size: float = split_size
        logger.info("Loading original dataset...")
        self.original_dataframe: pl.DataFrame = self.load_dataframe(
            dataframe_path=self.original_dataframe_path
        )
        logger.info("Pre-processing dataset...")
        self.full_dataframe: pl.DataFrame = self.preprocess_dataset()
        logger.info("Splitting dataset... ")
        self.validation_set, self.calibration_set = self.split_dataset()

    @abstractmethod
    def preprocess_dataset(
        self,
    ) -> pl.DataFrame:
        """pre-process the dataset"""

    @abstractmethod
    def load_dataframe(self, dataframe_path: Path | None = None) -> pl.DataFrame:
        """load a version of the dataframe"""

    def split_dataset(self) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """split the calibration dataset based on some document id column"""
        ## TODO: not currently shuffling split
        documents = self.full_dataframe.select(self.document_id_column).unique(
            maintain_order=True
        )
        documents = documents.sample(fraction=1.0, seed=self.seed, shuffle=True)
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
