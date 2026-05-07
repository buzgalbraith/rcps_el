"""
Abstract class for getting the loss from a list
"""

from rcps_el.aggregators import Aggregator
import polars as pl

from typing import Optional
from abc import ABC, abstractmethod


class lossFunction(ABC):
    """Abstract class for loss functions, has aggregator_method for how to aggregate over samples with multiple labels"""

    name: str = NotImplemented
    label_curie_col: str = NotImplemented
    candidate_curie_col: str = NotImplemented
    default_agg_method: Aggregator = NotImplemented

    def __init__(self, agg_method: Optional[Aggregator] = None) -> None:
        self.agg_method: Aggregator = (
            agg_method if agg_method else self.default_agg_method
        )
        self.aggregator_name: str = self.agg_method.name
        self.name = f"{self.name}_{self.aggregator_name}"

    def execute(
        self,
        data_frame: pl.DataFrame,
    ) -> pl.DataFrame:
        """run score list over data frame"""
        self.processing_function(data_frame=data_frame)
        return data_frame.with_columns(
            pl.struct([self.label_curie_col, self.candidate_curie_col])
            .map_elements(
                lambda x: self.calc_loss(
                    x[self.label_curie_col], x[self.candidate_curie_col]
                ),
                return_dtype=pl.Float64,
            )
            .alias(f"{self.name}")
        )

    def processing_function(self, data_frame: pl.DataFrame):
        """processing prior to running loss function"""
        pass

    @abstractmethod
    def calc_loss(self, labels, candidate_set) -> float:
        """Loss the list its self"""
