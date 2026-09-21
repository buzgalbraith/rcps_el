from .rcpsELEvaluator import rcpsELEvaluator
from rcps_el.losses import lossFunction
from rcps_el.dataset import Dataset
from rcps_el.scores import Scorer

import polars as pl

from itertools import product
from tqdm import tqdm
from pathlib import Path
import os
import logging

logger = logging.getLogger(__name__)

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent.parent
RESULTS_BASE = REPO_ROOT.joinpath("results")
DEFAULT_RESULT = RESULTS_BASE.joinpath("rcps_el_results_summary.tsv")


class rcpsELSetEvaluator:
    ## identity of a trial, used to de-duplicate against cached results. The
    ## risk-control settings belong here: the same dataset/score/loss at a
    ## different delta or bound is a different trial, not a duplicate.
    summary_cols = [
        "dataset",
        "split",
        "target_proportional_risk_increase",
        "min_candidates",
        "evaluation_strategy",
        "score_function",
        "loss_function",
        "delta",
        "bound",
        "risk_unit",
    ]

    def __init__(
        self,
        benchmarks: list[Dataset],
        scores: list[Scorer],
        losses: list[lossFunction],
        results_path: Path,
        target_proportion_risk: list[float] = [
            0.00,
            0.01,
            0.02,
            0.05,
            0.10,
            0.20,
            0.25,
        ],
        risk_types: list[bool] = [False],
        min_candidates: list[int] = [2],
        delta: float = 0.1,
        bound: str = "wsr",
        risk_unit: str = "document",
    ) -> None:
        self.evaluators: list[rcpsELEvaluator] = []
        self.benchmarks = benchmarks
        self.scores = scores
        self.losses = losses
        self.risk_types = risk_types
        self.min_candidates = min_candidates
        self.target_proportion_risk = target_proportion_risk
        ## risk-control settings, forwarded to every evaluator ##
        self.delta = delta
        self.bound = bound
        self.risk_unit = risk_unit
        self.result_set: pl.DataFrame | None = None
        self.results_path = (
            results_path if isinstance(results_path, Path) else Path(DEFAULT_RESULT)
        )
        os.makedirs(self.results_path.parent, exist_ok=True)

    def execute(self, verbose: bool = False):
        itter = product(
            self.benchmarks,
            self.scores,
            self.losses,
            self.risk_types,
            self.min_candidates,
            self.target_proportion_risk,
        )
        total = (
            len(self.benchmarks)
            * len(self.scores)
            * len(self.losses)
            * len(self.risk_types)
            * len(self.min_candidates)
            * len(self.target_proportion_risk)
        )
        records = []
        progress = tqdm(
            itter,
            total=total,
            desc="Evaluating RCPS entity-linking configurations",
            unit="config",
        )
        for dataset, score, loss, risk_type, min_candidate, target_risk in progress:
            progress.set_postfix_str(
                f"data={dataset.name} score={score.name} loss={loss.name} "
                f"risk={'abs' if risk_type else 'prop'} "
                f"min_cand={min_candidate} target_risk={target_risk}"
            )
            evaluator = rcpsELEvaluator(
                dataset=dataset,
                score_function=score,
                loss_function=loss,
                min_candidates=min_candidate,
                absolute_risk=risk_type,
                target_proportional_risk_increase=target_risk,
                delta=self.delta,
                bound=self.bound,
                risk_unit=self.risk_unit,
            )
            evaluator.execute(verbose=verbose)
            self.evaluators.append(evaluator)
            records += evaluator.results_summary
            ## cache after every trial ##
            self.result_set = pl.from_dicts(records)
            self.safe_write_results()

    def safe_write_results(self):
        assert isinstance(self.result_set, pl.DataFrame)
        write_results = self.result_set
        if self.results_path.exists():
            existing_results = pl.read_csv(self.results_path, separator="\t")
            try:
                new_rows = self.result_set.join(
                    existing_results, on=self.summary_cols, how="anti"
                )
                write_results = existing_results.vstack(new_rows)
            except pl.exceptions.ShapeError:
                raise ValueError(
                    f"Existing and new dataset schemas do not match. Consider removing existing results at {self.results_path}"
                )
        write_results.write_csv(self.results_path, separator="\t")
