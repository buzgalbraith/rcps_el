from rcps_og import rcpsOGEvaluator
from rcps_og.dataset import bioIDBenchmark, bioRedBenchmark, BCD5, Dataset
from rcps_og.scores import fuzzyStringScore, gildaScorer, sapbertScorer, krissbertScorer, Scorer
from rcps_og.losses import binaryMisscoverageLoss, lossFunction
from rcps_og.dataset.bioIDGilda import bioIDGildaBenchmark


from itertools import product
from tqdm import tqdm
import os
import polars as pl
from pandas import DataFrame

BENCHMARKS: list[Dataset] = [BCD5(), bioIDBenchmark(), bioRedBenchmark()]
SCORES: list[Scorer] = [fuzzyStringScore(), gildaScorer(), sapbertScorer(), krissbertScorer()]
LOSSES: list[lossFunction] = [binaryMisscoverageLoss()]
RISK_TYPES = [True, False]
MIN_CANDIDATES = [2, 5, 10]
TARGET_PROPORTIONAL_RISKS = [0.00, 0.01, 0.02, 0.05, 0.10, 0.20, 0.25]


RISK_TYPES = [False]
# SCORES: list[Scorer] = [fuzzyStringScore(), sapbertScorer()]
SCORES : list[Scorer] = [krissbertScorer()]
BENCHMARKS: list[Dataset] = [BCD5(method="krissbert")]
MIN_CANDIDATES = [
    20,
]


if __name__ == "__main__":
    itter = product(
        BENCHMARKS,
        SCORES,
        LOSSES,
        RISK_TYPES,
        MIN_CANDIDATES,
        TARGET_PROPORTIONAL_RISKS,
    )
    os.makedirs("./figs", exist_ok=True)
    records = []
    if os.path.exists("trials.tsv"):
        df = pl.read_csv("trials.tsv", separator="\t")
    else:
        df = None
    for dataset, score, loss, risk_type, min_candidate, target_risk in tqdm(
        itter, desc="Running trials"
    ):
        evaluator = rcpsOGEvaluator(
            dataset=dataset,
            score_function=score,
            loss_function=loss,
            min_candidates=min_candidate,
            absolute_risk=risk_type,
            target_proportional_risk_increase=target_risk,
        )
        evaluator.execute()

        calibration_summary, validation_summary = evaluator.results_summary
        ## calibration
        original_risk_calibration = calibration_summary.get("risk_original")
        risk_controlled_calibration = calibration_summary.get("risk_controlled")
        c_set_size_original_calibration = calibration_summary.get("c_set_size_original")
        c_set_size_controlled_calibration = calibration_summary.get(
            "c_set_size_controlled"
        )
        samples_calibration = calibration_summary.get("samples")
        ## validation
        original_risk_validation = validation_summary.get("risk_original")
        risk_controlled_validation = validation_summary.get("risk_controlled")
        c_set_size_original_validation = validation_summary.get("c_set_size_original")
        c_set_size_controlled_validation = validation_summary.get(
            "c_set_size_controlled"
        )
        samples_validation = validation_summary.get("samples")

        orig = evaluator.result_validation_original
        fitted = evaluator.result_validation_fitted
        orig_candidates = orig["n_candidates"].to_numpy()
        fitted_candidates = fitted["n_candidates"].to_numpy()

        risk_name = "absolute" if risk_type else "relative"
        ## add training information
        records.append(
            {
                "dataset_name": f"{dataset.name}_{dataset.method}",
                "split": "calibration",
                "score": score.name,
                "min_candidates": min_candidate,
                "target_risk": target_risk,
                "risk_type": risk_name,
                "original_risk": original_risk_calibration,
                "controlled_risk": risk_controlled_calibration,
                "original_c_set_size": c_set_size_original_calibration,
                "controlled_c_set_size": c_set_size_controlled_calibration,
                "samples": samples_calibration,
            }
        )
        records.append(
            {
                "dataset_name": f"{dataset.name}_{dataset.method}",
                "split": "validation",
                "score": score.name,
                "min_candidates": min_candidate,
                "target_risk": target_risk,
                "risk_type": risk_name,
                "original_risk": original_risk_validation,
                "controlled_risk": risk_controlled_validation,
                "original_c_set_size": c_set_size_original_validation,
                "controlled_c_set_size": c_set_size_controlled_validation,
                "samples": samples_validation,
            }
        )
        ## incremental updates for the dataset
        if df is not None:
            update = pl.from_dicts(records, schema=df.schema)
            df = df.vstack(update).unique()
        else:
            df = pl.from_dicts(records)
        df.write_csv("trials.tsv", separator="\t")
