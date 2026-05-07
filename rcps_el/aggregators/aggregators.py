"""Class for list aggregators"""

from abc import ABC, abstractmethod


class Aggregator(ABC):
    """Abstract class for safely aggregating results"""

    name: str = NotImplemented

    @abstractmethod
    def execute(self, scores: list[float]) -> float:
        """runs the aggregation"""


class safeMaxAggregator(Aggregator):
    name = "safe_max_aggregation"

    def execute(self, scores: list[float]) -> float:
        if scores:  ## evaluates to false if empty list
            return max(scores)
        else:
            return 0.0


class safeMinAggregator(Aggregator):
    name = "safe_min_aggregation"

    def execute(self, scores: list[float]) -> float:
        if scores:  ## evaluates to false if empty list
            return min(scores)
        else:
            ## if there is no list
            return 1.0
