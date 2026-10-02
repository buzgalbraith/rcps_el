from .rcpsELEvaluator import rcpsELEvaluator
from rcps_el.losses import lossFunction
from rcps_el.dataset import Dataset
from rcps_el.scores import Scorer
from rcps_el.utils import ensure_tqdm_logging

import polars as pl

from itertools import product
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
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
    ## different delta or bound is a different trial, not a duplicate. Likewise
    ## seed/split_size: a different split of the same dataset is a different trial.
    summary_cols = [
        "dataset",
        "split",
        "seed",
        "split_size",
        "target_risk",
        "min_candidates",
        "risk_formulation",
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
        target_risks: list[float] = [
            0.00,
            0.01,
            0.02,
            0.05,
            0.10,
            0.20,
            0.25,
        ],
        risk_formulations: list[str] = ["relative"],
        min_candidates: list[int] = [2],
        deltas: list[float] = [0.1],
        bound: str = "wsr",
        risk_unit: str = "document",
        derived_hits_at_k: list[int] = [1,2,5,10],
    ) -> None:
        self.evaluators: list[rcpsELEvaluator] = []
        self.benchmarks = benchmarks
        self.scores = scores
        self.losses = losses
        self.risk_formulations = risk_formulations
        self.min_candidates = min_candidates
        self.target_risks = target_risks
        ## risk-control settings, forwarded to every evaluator ##
        self.deltas = deltas
        self.bound = bound
        self.risk_unit = risk_unit
        ## reporting only, not part of a trial's identity ##
        self.derived_hits_at_k = derived_hits_at_k
        self.result_set: pl.DataFrame | None = None
        self.results_path = (
            results_path if isinstance(results_path, Path) else Path(DEFAULT_RESULT)
        )
        os.makedirs(self.results_path.parent, exist_ok=True)

    def execute(self, verbose: bool = False):
        """
        Run every configuration in the grid, checkpointing results after each.

        Progress is shown as one bar over configurations with each evaluator's q*
        scan nested beneath it and cleared when that configuration finishes. Log
        records are printed above the bars rather than through them.
        """
        ensure_tqdm_logging()
        ## also covers handlers a caller installed on the root logger (basicConfig) ##
        with logging_redirect_tqdm():
            self._execute(verbose=verbose)

    def _execute(self, verbose: bool):
        itter = product(
            self.benchmarks,
            self.scores,
            self.losses,
            self.risk_formulations,
            self.min_candidates,
            self.target_risks,
            self.deltas
        )
        total = (
            len(self.benchmarks)
            * len(self.scores)
            * len(self.losses)
            * len(self.risk_formulations)
            * len(self.min_candidates)
            * len(self.target_risks)
            * len(self.deltas) 
        )
        records = []
        progress = tqdm(
            itter,
            total=total,
            desc="RCPS configurations",
            unit="config",
            position=0,
            leave=True,
            dynamic_ncols=True,
        )
        for dataset, score, loss, risk_formulation, min_candidate, target_risk, delta in progress:
            split_params = dataset.split_parameters()
            config = (
                f"data={dataset.name} seed={split_params['seed']} "
                f"score={score.name} loss={loss.name} "
                f"risk={risk_formulation} min_cand={min_candidate} "
                f"alpha={target_risk} delta={delta}"
            )
            progress.set_postfix_str(config)
            evaluator = rcpsELEvaluator(
                dataset=dataset,
                score_function=score,
                loss_function=loss,
                min_candidates=min_candidate,
                risk_formulation=risk_formulation,
                target_risk=target_risk,
                delta=delta,
                bound=self.bound,
                risk_unit=self.risk_unit,
                derived_hits_at_k=self.derived_hits_at_k,
            )
            evaluator.execute(
                verbose=verbose, progress_position=1, leave_progress=False
            )
            self.evaluators.append(evaluator)
            logger.info(
                f"[{progress.n + 1}/{total}] {config} -> q*={evaluator.q_star:.4g} "
                f"certified={evaluator.q_star_certified}"
            )
            records += evaluator.results_summary
            ## cache after every trial ##
            self.result_set = pl.from_dicts(records)
            self.safe_write_results()

    def safe_write_results(self):
        assert isinstance(self.result_set, pl.DataFrame)
        write_results = self.result_set
        if self.results_path.exists():
            existing_results = pl.read_csv(
                self.results_path, separator="\t", infer_schema_length=None
            )
            existing_results = self._align_identity_columns(existing_results)
            try:
                ## nulls_equal so datasets with fixed splits (seed=None) still de-duplicate ##
                new_rows = self.result_set.join(
                    existing_results, on=self.summary_cols, how="anti", nulls_equal=True
                )
                ## diagonal so optional columns (e.g. derived hits@k) can be added to,
                ## or missing from, an existing results file; absent values are null ##
                write_results = pl.concat(
                    [existing_results, new_rows], how="diagonal_relaxed"
                )
            except (pl.exceptions.ShapeError, pl.exceptions.SchemaError):
                raise ValueError(
                    f"Existing and new dataset schemas do not match. Consider removing existing results at {self.results_path}"
                )
        write_results.write_csv(self.results_path, separator="\t")

    def _align_identity_columns(self, existing_results: pl.DataFrame) -> pl.DataFrame:
        """
        Make the identity columns of a results file joinable with the new results.

        Files written before an identity column existed (e.g. seed) get it as null,
        and all-null columns read back as strings are cast to the new dtype.
        """
        missing = [c for c in self.summary_cols if c not in existing_results.columns]
        if missing:
            logger.warning(
                f"{self.results_path} has no {missing} columns; treating them as null "
                "for existing rows, so re-running those trials will add new rows "
                "rather than being recognised as duplicates."
            )
        return existing_results.with_columns(
            [pl.lit(None).alias(c) for c in missing]
        ).with_columns(
            [
                pl.col(c).cast(self.result_set.schema[c])
                for c in self.summary_cols
                if existing_results.schema.get(c, pl.Null) != self.result_set.schema[c]
            ]
        )
