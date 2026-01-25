"""
Abstract class for a score function
"""

import polars as pl

from abc import ABC, abstractmethod


class Scorer(ABC):
    """Abstract class for scoring candidate sets"""

    name: str = NotImplemented
    raw_text_col: str = NotImplemented
    entity_name_col: str = NotImplemented

    def execute(self, data_frame: pl.DataFrame) -> pl.DataFrame:
        """Get scores over a dataset"""
        self.processing_function(data_frame=data_frame)
        return data_frame.with_columns(
            pl.struct([self.raw_text_col, self.entity_name_col])
            .map_elements(
                lambda x: self.score_sample(
                    x[self.raw_text_col], x[self.entity_name_col]
                ),
                return_dtype=pl.List(pl.Float64),
            )
            .alias(f"{self.name}")
        )

    def processing_function(self, data_frame: pl.DataFrame):
        """processing prior to running score function"""
        pass

    @abstractmethod
    def score_sample(self, entity, candidates) -> list[float]:
        """Score the list its self"""
