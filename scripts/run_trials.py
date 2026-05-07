from rcps_el import rcpsELEvaluator
from rcps_el.dataset import bioIDBenchmark, bioRedBenchmark, BCD5, Dataset
from rcps_el.scores import fuzzyStringScore, gildaScorer, sapbertScorer, krissbertScorer, llmScorer, Scorer
from rcps_el.losses import binaryMisscoverageLoss, hitsAtK, lossFunction
# from rcps_el.dataset.bioIDGilda import bioIDGildaBenchmark


from itertools import product
from tqdm import tqdm 
import os
import polars as pl

BENCHMARKS: list[Dataset] = [BCD5(), bioIDBenchmark(), bioRedBenchmark()]

SCORES: list[Scorer] = [fuzzyStringScore(), gildaScorer(), sapbertScorer()]
# SCORES: list[Scorer] = [llmScorer(batch_size=1)]
# SCORES: list[Scorer] = [krissbertScorer()]
# LOSSES: list[lossFunction] = [binaryMisscoverageLoss(), hitsAtK(k_size=1)]
# RISK_TYPES = [True, False]
# MIN_CANDIDATES = [2, 5, 10]

TARGET_PROPORTIONAL_RISKS = [0.00, 0.01, 0.02, 0.05, 0.10, 0.20, 0.25]


RISK_TYPES = [False]
LOSSES: list[lossFunction] = [hitsAtK(k_size=1), hitsAtK(k_size=2), hitsAtK(k_size=5), hitsAtK(k_size=10)]
# LOSSES: list[lossFunction] = [binaryMisscoverageLoss()]
# SCORES: list[Scorer] = [fuzzyStringScore(), sapbertScorer(), gildaScorer()]
# SCORES: list[Scorer] = [llmScorer(batch_size=1)]
# SCORES : list[Scorer] = [krissbertScorer()]
# BENCHMARKS: list[Dataset] = [bioIDBenchmark(method='gilda')]
# BENCHMARKS: list[Dataset] = [bioIDBenchmark(method='gilda')]
# BENCHMARKS: list[Dataset] = [BCD5(method='gilda')]
BENCHMARKS: list[Dataset] = [BCD5(method='krissbert')]
# SCORES : list[Scorer] = [fuzzyStringScore(), sapbertScorer(), llmScorer()]
MIN_CANDIDATES = [
    2
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
        evaluator = rcpsELEvaluator(
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
        loss_type = loss.name
        ## add training information
        records.append(
            {
                "dataset_name": f"{dataset.name}_{dataset.method}",
                "split": "calibration",
                "score": score.name,
                "min_candidates": min_candidate,
                "target_risk": target_risk,
                "risk_type": risk_name,
                "loss_name": loss_type,
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
                "loss_name": loss_type,
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
