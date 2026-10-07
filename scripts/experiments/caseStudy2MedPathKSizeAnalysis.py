"""
K size ablation for the MedPath case study: how certification, set size and
validity change when only the top k_size retrieved candidates are kept.
"""
from pystow import module
from numpy.random import default_rng

from rcps_el.dataset import medPathBenchmark
from rcps_el.scores import retrievalScorer, cumulativeRetrievalScorer, Scorer
from rcps_el.losses import descendantsAtK, binaryMisscoverageLoss, lossFunction
from rcps_el.evaluators import rcpsELSetEvaluator

METHOD_NAME = "MedPath Score"
RESPLIT: bool = True
TRIAL_PATH = module("rcps_el", "experiments").base.joinpath("trials_medpath_KSize.tsv")

DELTAS = [0.1]
TARGET_RISKS = [0.10 ,0.15, 0.20, 0.25]
RISK_FORMULATIONS: list[str] = ["absolute", "relative"]
SCORES: list[Scorer] = [retrievalScorer(name=METHOD_NAME), cumulativeRetrievalScorer(name=METHOD_NAME)]
DERIVED_HITS_AT_K: list[int] = [10]
LOSSES: list[lossFunction] = [descendantsAtK(k_size=1, k_candidates=False), binaryMisscoverageLoss()]

## each repeat draws a new calibration/validation split; the split is document-level,
## so for a given seed every k_size sees the same documents and only the candidate
## lists are truncated ##
SEED = 10022
SPLIT_SIZE = 0.3
K_SIZES: list[int] = [2, 5, 10, 15, 20]
NUM_REPEATS = 10
rng = default_rng(seed=SEED)
split_seeds = rng.integers(10**4, size=NUM_REPEATS)

## (k_size, split_seed); repeats outermost so a partial run already covers every k_size ##
CONFIGS = [(k_size, int(seed)) for seed in split_seeds for k_size in K_SIZES]

if __name__ == "__main__":
    ## one benchmark at a time: each holds its scored frames, so building them all up front
    ## does not fit in memory. Finished configurations already in TRIAL_PATH are skipped ##
    for k_size, seed in CONFIGS:
        benchmark = medPathBenchmark(
            resplit=RESPLIT, seed=seed, split_size=SPLIT_SIZE, k_size=k_size
        )
        evaluator = rcpsELSetEvaluator(
            benchmarks=[benchmark],
            scores=SCORES,
            losses=LOSSES,
            results_path=TRIAL_PATH,
            target_risks=TARGET_RISKS,
            risk_formulations=RISK_FORMULATIONS,
            deltas=DELTAS,
            derived_hits_at_k=DERIVED_HITS_AT_K,
        )
        evaluator.execute(verbose=True)
