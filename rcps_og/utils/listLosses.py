"""
Abstract class for getting the loss from a list
"""
import polars as pl

from typing import Callable, Optional
from abc import ABC, abstractmethod

class listLoss(ABC):
    """Abstract class for methods that getting loss for list results"""
    name:str = NotImplemented
    label_col:str = NotImplemented
    candidate_name_col:str = NotImplemented
    default_agg_method = NotImplemented
    def agg(self,  data_frame:pl.DataFrame,agg_method:Optional[Callable]=None)->pl.DataFrame:
        """run score list over data frame"""
        if agg_method is None:
            agg_method = self.default_agg_method
        self.processing_function(data_frame=data_frame)
        return data_frame.with_columns(
                 pl.struct([self.label_col, self.candidate_name_col]).map_elements(
                    lambda x: self.loss_list(x[self.label_col], x[self.candidate_name_col], agg_method=agg_method), 
                    return_dtype=pl.Float64
                ).alias(f'{self.name}')
            )
    def processing_function(self, data_frame:pl.DataFrame):
        """processing prior to running loss function"""
        pass
    @abstractmethod
    def loss_list(self,  labels, candidate_set, agg_method)->float:
        """Loss the list its self"""
    def safe_min_loss(self,scores:list[float])->float:
        if scores: ## evaluates to false if empty list  
            return min(scores)
        else:
            ## if there is no list
            return 1.0

class binaryMisscoverageLoss(listLoss):
    label_col = 'obj_synonyms'
    candidate_name_col = 'gilda_curie'
    default_agg_method = listLoss.safe_min_loss
    name = 'binary_misscoverage_loss'
    def loss_list(self,  labels: list[str], candidate_set: list[str], agg_method:Callable=default_agg_method)->float:
        term_losses = [float(label not in candidate_set) for label in labels]
        return agg_method(term_losses)
    def processing_function(self, data_frame: pl.DataFrame):
        return super().processing_function(data_frame)
