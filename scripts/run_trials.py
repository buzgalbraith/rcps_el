from rcps_el import rcpsELEvaluator
from rcps_el.dataset import (
    bioIDBenchmark,
    bioRedBenchmark,
    BCD5,
    medCodERBenchmark,
    Dataset,
)
from rcps_el.scores import (
    fuzzyStringScore,
    gildaScorer,
    sapbertScorer,
    krissbertScorer,
    llmScorer,
    MedCodErScorer,
    Scorer,
)
from rcps_el.losses import binaryMisscoverageLoss, hitsAtK, lossFunction

# from rcps_el.dataset.bioIDGilda import bioIDGildaBenchmark


from itertools import product
from tqdm import tqdm
import os
import polars as pl


def run_trials(
    benchmarks: list[Dataset],
    scores: list[Scorer],
    losses: list[lossFunction],
    target_proportion_risk: list[float] = [0.00, 0.01, 0.02, 0.05, 0.10, 0.20, 0.25],
    risk_types: list[bool] = [False],
    min_candidates: list[int] = [2],
):
    itter = product(
        benchmarks,
        scores,
        losses,
        risk_types,
        min_candidates,
        target_proportion_risk,
    )
    if os.path.exists("trials.tsv"):
        results_df = pl.read_csv("trials.tsv", separator="\t")
    else:
        results_df = None
    records = []
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
        records += evaluator.results_summary
        # incremental updates for the dataset
        if results_df is not None:
            update = pl.from_dicts(records, schema=results_df.schema)
            results_df = results_df.vstack(update).unique()
        else:
            results_df = pl.from_dicts(records)
        results_df.write_csv("trials.tsv", separator="\t")


if __name__ == "__main__":
    scores = [MedCodErScorer(), sapbertScorer()]

    losses = [binaryMisscoverageLoss(),]
    # losses = [hitsAtK(1), hitsAtK(2), hitsAtK(5), hitsAtK(10)]
    ## run trial for currently used method ##
    # run trials for new method with risk control ##
    run_trials(
        benchmarks=[medCodERBenchmark(n_retrieved=20, billable=True, resplit=True,  method='medcoder-rerank')],
        target_proportion_risk=[0.00, 0.01, 0.02, 0.05, 0.10, 0.20, 0.25],
        scores=scores,
        losses=losses,
    )
    # run_trials(
    #     benchmarks=[medCodERBenchmark(n_retrieved=10, billable=True, resplit=False,  method='medcoder-rerank')],
    #     target_proportion_risk=[0.00, 0.01, 0.02, 0.05, 0.10, 0.20, 0.25],
    #     scores=scores,
    #     losses=losses,
    # )

    # run_trials(
    #     benchmarks=[medCodERBenchmark(n_retrieved=10, billable=True, resplit=True, method='medcoder-retrieve')],
    #     target_proportion_risk=[0.00, 0.01, 0.02, 0.05, 0.10, 0.20, 0.25],
    #     scores=scores,
    #     losses=losses,
    # )

    # run_trials(
    #     benchmarks=[medCodERBenchmark(n_retrieved=20, billable=True, resplit=False,  method='medcoder-rerank')],
    #     target_proportion_risk=[0.00, 0.01, 0.02, 0.05, 0.10, 0.20, 0.25],
    #     scores=scores,
    #     losses=losses,
    # )

    # run_trials(
    #     benchmarks=[medCodERBenchmark(n_retrieved=20, billable=True, resplit=True, method='medcoder-retrieve')],
    #     target_proportion_risk=[0.00, 0.01, 0.02, 0.05, 0.10, 0.20, 0.25],
    #     scores=scores,
    #     losses=losses,
    # )