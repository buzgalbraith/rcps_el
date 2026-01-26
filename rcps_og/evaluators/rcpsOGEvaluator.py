"""
Class for running RCPS across a given dataset, with a specific score and loss function
"""

from rcps_og.scores import Scorer
from rcps_og.losses import lossFunction
from rcps_og.dataset import Dataset
from rcps_og.utils import safeMatch

from numpy import linspace
import polars as pl
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class rcpsOGEvaluator:
    results_summary: list[dict] = []

    def __init__(
        self,
        dataset: Dataset,
        score_function: Scorer,
        loss_function: lossFunction,
        target_proportional_risk_increase: float = 0.2,
        absolute_risk: bool = False,
        min_candidates: int = 2,
        max_q: int | float = 1.0,
        min_q: int | float = 0.0,
        num_steps: int = 100,
    ) -> None:
        """Make an evaluator
        Parameters:
            target_proportional_risk_increase : optional,float
                Max allowable % change increase in risk.
            absolute_risk : optional, bool
                If to calculate risk over all samples or just those grounded to more than min_candidates candidates
            min_candidates : optional, int
                Minimum number of candidates to try to narrow candidate set for
        """
        self.dataset = dataset
        self.score_function = score_function
        self.loss_function = loss_function
        self.target_proportional_risk_increase = target_proportional_risk_increase
        self.absolute_risk = absolute_risk
        self.min_candidates = min_candidates
        self.q_range = sorted(
            linspace(start=min_q, stop=max_q, num=num_steps), reverse=True
        )
        self.result_calibration_original = self.get_original_results(
            self.dataset.calibration_set
        )
        self.result_validation_original = self.get_original_results(
            self.dataset.validation_set
        )
        self.calibration_risk_index = self.get_risk_index(
            self.result_calibration_original
        )
        self.validation_risk_index = self.get_risk_index(
            self.result_validation_original
        )
        self.original_empirical_risk = self.calc_empirical_risk(
            self.result_calibration_original, calibration=True
        )
        self.q_star: float | None = None
        self.result_calibration_fitted: pl.DataFrame | None = None
        self.result_validation_fitted: pl.DataFrame | None = None

    def execute(
        self,
        verbose: bool = True,
    ):
        """Fit and evaluate the model"""
        logger.info("Fitting q* on calibration data...")
        self.get_q_star(verbose=verbose)
        logger.info("Evaluating q* on validation data...")
        self.evaluate_on_validation()
        logger.info("Summary:")
        self.get_results_summary()

    def get_risk_index(self, dataset: pl.DataFrame):
        if self.absolute_risk:
            return dataset["index"].unique()
        else:
            return dataset.filter(pl.col("n_candidates") >= self.min_candidates)[
                "index"
            ].unique()

    def get_results_summary(self):
        assert isinstance(self.result_calibration_fitted, pl.DataFrame) and isinstance(
            self.result_validation_fitted, pl.DataFrame
        )
        self.results_summary: list[dict] = []  ## reset just in case
        ## get calibration results
        self.results_summary.append(
            {
                "dataset": self.dataset.name,
                "split": "calibration",
                "target_proportional_risk_increase": self.target_proportional_risk_increase,
                "min_candidates": self.min_candidates,
                "evaluation_strategy": "absolute" if self.absolute_risk else "relative",
                "score_function": self.score_function.name,
                "loss_function": self.loss_function.name,
                "samples": len(self.calibration_risk_index),
                "risk_original": self.calc_empirical_risk(
                    self.result_calibration_original, calibration=True
                ),
                "risk_controlled": self.calc_empirical_risk(
                    self.result_calibration_fitted, calibration=True
                ),
                "c_set_size_original": self.get_average_candidates(
                    self.result_calibration_original, calibration=True
                ),
                "c_set_size_controlled": self.get_average_candidates(
                    self.result_calibration_fitted, calibration=True
                ),
            }
        )
        ## get validation results
        self.results_summary.append(
            {
                "dataset": self.dataset.name,
                "split": "validation",
                "target_proportional_risk_increase": self.target_proportional_risk_increase,
                "min_candidates": self.min_candidates,
                "evaluation_strategy": "absolute" if self.absolute_risk else "relative",
                "score_function": self.score_function.name,
                "loss_function": self.loss_function.name,
                "samples": len(self.validation_risk_index),
                "risk_original": self.calc_empirical_risk(
                    self.result_validation_original, calibration=False
                ),
                "risk_controlled": self.calc_empirical_risk(
                    self.result_validation_fitted, calibration=False
                ),
                "c_set_size_original": self.get_average_candidates(
                    self.result_validation_original, calibration=False
                ),
                "c_set_size_controlled": self.get_average_candidates(
                    self.result_validation_fitted, calibration=False
                ),
            }
        )
        logger.info(f"Calibration samples {len(self.result_calibration_original)}")
        if not self.absolute_risk:
            logger.info(
                f"Calibration samples with at least {self.min_candidates}: {len(self.calibration_risk_index)}"
            )
        logger.info(
            f"Calibration risk: {self.calc_empirical_risk(self.result_calibration_original, calibration=True)}->{self.calc_empirical_risk(self.result_calibration_fitted,calibration=True)}"
        )
        logger.info(
            f"Calibration average candidate set size:  {self.get_average_candidates(self.result_calibration_original,calibration=True)}->{self.get_average_candidates(self.result_calibration_fitted,calibration=True)}"
        )
        logger.info("-" * 100)
        logger.info(f"Validation samples {len(self.result_validation_original)}")
        if not self.absolute_risk:
            logger.info(
                f"Validation samples with at least {self.min_candidates}: {len(self.validation_risk_index)}"
            )
        logger.info(
            f"Validation risk: {self.calc_empirical_risk(self.result_validation_original,calibration=False)}->{self.calc_empirical_risk(self.result_validation_fitted,calibration=False)}"
        )
        logger.info(
            f"Validation average candidate set size:  {self.get_average_candidates(self.result_validation_original,calibration=False)}->{self.get_average_candidates(self.result_validation_fitted,calibration=False)}"
        )

    def get_average_candidates(self, dataset: pl.DataFrame, calibration: bool):
        if calibration:
            return (
                dataset.filter(pl.col("index").is_in(self.calibration_risk_index))
                .select(pl.mean("n_candidates"))
                .item()
            )
        return (
            dataset.filter(pl.col("index").is_in(self.validation_risk_index))
            .select(pl.mean("n_candidates"))
            .item()
        )

    def get_original_results(self, dataset: pl.DataFrame):
        """Get metrics on the original dataframe"""
        dataset = self.score_function.execute(dataset)
        dataset = self.loss_function.execute(dataset)
        return self.count_candidates(dataset)

    def count_candidates(self, dataset: pl.DataFrame):
        """
        Get the size of candidate set for each row in a dataset.
        """
        return dataset.with_columns(n_candidates=pl.col("match_curies").list.len())

    def calc_empirical_risk(self, dataset: pl.DataFrame, calibration: bool) -> float:
        if calibration:
            return (
                dataset.filter(pl.col("index").is_in(self.calibration_risk_index))
                .select(pl.mean(self.loss_function.name))
                .item()
            )
        return (
            dataset.filter(pl.col("index").is_in(self.validation_risk_index))
            .select(pl.mean(self.loss_function.name))
            .item()
        )

    def evaluate_on_validation(
        self, q: float | None = None
    ) -> tuple[float, pl.DataFrame]:
        q = q if q else self.q_star
        assert isinstance(self.q_star, float)
        result_validation = self.filter_candidates(
            q=self.q_star, dataset=self.result_validation_original
        )
        result_validation = self.loss_function.execute(result_validation)
        result_validation = self.count_candidates(result_validation)
        empirical_risk = self.calc_empirical_risk(result_validation, calibration=False)
        self.result_validation_fitted = result_validation
        return empirical_risk, result_validation

    def get_q_star(self, verbose: bool = True) -> float:
        if self.q_star:
            logger.info(f"q star loading from cache...")
            return self.q_star
        q = max(self.q_range)

        self.result_calibration_fitted = self.result_calibration_original
        for q in tqdm(
            self.q_range,
            desc=f"Finding optimal score threshold on {self.dataset.name} calibration dataset using {self.score_function.name} and {self.loss_function.name}",
            total=len(self.q_range),
        ):
            result_calibration = self.filter_candidates(
                q=q, dataset=self.result_calibration_original
            )
            result_calibration = self.loss_function.execute(result_calibration)
            result_calibration = self.count_candidates(result_calibration)
            empirical_risk = self.calc_empirical_risk(
                result_calibration, calibration=True
            )
            proportion_risk_delta = (
                empirical_risk - self.original_empirical_risk
            ) / self.original_empirical_risk  ## % change in risk compared to original
            if verbose:
                avg_candidates = result_calibration["n_candidates"].mean()
                logger.info(
                    f"q:{q}, risk:{empirical_risk},proportional risk delta {proportion_risk_delta},  average number of candidates: {avg_candidates}"
                )
            if proportion_risk_delta <= self.target_proportional_risk_increase:
                self.result_calibration_fitted = result_calibration
                break
        self.q_star = q
        return q

    def filter_candidates(self, q: float, dataset: pl.DataFrame) -> pl.DataFrame:
        """
        filter candidates above score threshold
        """
        return dataset.with_columns(
            evaluated=pl.struct(
                ["match_names", "match_curies", self.score_function.name]
            ).map_elements(
                lambda x: self._filter_candidates(
                    x["match_names"],
                    x["match_curies"],
                    x[self.score_function.name],
                    q=q,
                ),
                return_dtype=pl.List(
                    pl.Struct(
                        {
                            "name": pl.String,
                            "curie": pl.String,
                            "score": pl.Float64,
                        }
                    )
                ),
            )
        ).with_columns(
            pl.col("evaluated")
            .list.eval(pl.element().struct.field("name"))
            .alias("match_names"),
            pl.col("evaluated")
            .list.eval(pl.element().struct.field("curie"))
            .alias("match_curies"),
            pl.col("evaluated")
            .list.eval(pl.element().struct.field("score"))
            .alias(self.score_function.name),
        )

    def _filter_candidates(
        self, names: list[str], curies: list[str], scores: list[float], q: float
    ) -> list[safeMatch]:
        """
        internal method for filtering candidates above score threshold
        """
        records = []
        ## short circuit so do not filter any candidate sets smaller than our min target ##
        if len(scores) < self.min_candidates:
            q = 0.0
        max_score_index = 0
        for i, score in enumerate(scores):
            if score > scores[max_score_index]:
                max_score_index = i
            if score < q:
                continue
            records.append({"name": names[i], "curie": curies[i], "score": scores[i]})
        ## don't reduce the size of the set bellow one candidate
        if len(records) < 1 and len(scores) > 0:
            records.append(
                {
                    "name": names[max_score_index],
                    "curie": curies[max_score_index],
                    "score": scores[max_score_index],
                }
            )
        return records
