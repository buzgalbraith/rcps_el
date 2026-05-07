from .rcpsELEvaluator import rcpsELEvaluator
import polars as pl
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
    summary_cols = [
        "dataset",
        "split",
        "target_proportional_risk_increase",
        "min_candidates",
        "evaluation_strategy",
        "score_function",
        "loss_function",
    ]

    def __init__(
        self, evaluators: list[rcpsELEvaluator], results_path: Path | None = None
    ) -> None:
        self.evaluators = evaluators
        self.result_set: pl.DataFrame | None = None
        self.results_path = (
            results_path if isinstance(results_path, Path) else Path(DEFAULT_RESULT)
        )
        os.makedirs(self.results_path.parent, exist_ok=True)

    def execute(self, verbose: bool = False):
        records = []
        for evaluator in tqdm(
            self.evaluators,
            total=len(self.evaluators),
            desc="Running evaluations with different configurations",
            unit="configuration",
        ):
            evaluator.execute(verbose=verbose)
            records += evaluator.results_summary
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
