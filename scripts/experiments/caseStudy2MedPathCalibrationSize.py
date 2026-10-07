"""
Calibration size analysis for the MedPath with SapBERT case study: how certification,
set size and validity change with the number of calibration documents, against a
validation set that stays fixed.

One seeded document-level calibration/validation split; calibration_size documents
are then drawn from the calibration side, stratified by corpus. For each repeat the
draws are nested (a smaller size is a subset of a larger one), so differences across
sizes come from the size and not from which documents were drawn. The full
calibration split (calibration_size=None) is the same for every repeat, so it runs once.
"""
from pystow import module
from numpy.random import default_rng

from rcps_el.dataset import medPathBenchmark
from rcps_el.scores import retrievalScorer, cumulativeRetrievalScorer, Scorer
from rcps_el.losses import descendantsAtK, binaryMisscoverageLoss, lossFunction
from rcps_el.evaluators import rcpsELSetEvaluator

METHOD_NAME = "MedPath Score"
RESPLIT: bool = True
TRIAL_PATH = module("rcps_el", "experiments").base.joinpath("trials_medpath_CalibrationSize.tsv")

DELTAS = [0.1]
TARGET_RISKS = [0.10, 0.25]
RISK_FORMULATIONS: list[str] = ["absolute", "relative"]
SCORES: list[Scorer] = [retrievalScorer(name=METHOD_NAME), cumulativeRetrievalScorer(name=METHOD_NAME)]
DERIVED_HITS_AT_K: list[int] = [10]
LOSSES: list[lossFunction] = [descendantsAtK(k_size=1, k_candidates=False), binaryMisscoverageLoss()]

## the validation split is fixed by SEED; only the calibration draw varies ##
SEED = 10022
SPLIT_SIZE = 0.3
## documents; the calibration split at SEED/SPLIT_SIZE holds 1594 ##
CALIBRATION_SIZES: list[int] = [25, 50, 100, 200, 400, 800]
NUM_REPEATS = 50
rng = default_rng(seed=SEED)
calibration_seeds = rng.integers(10**4, size=NUM_REPEATS)

## (calibration_size, calibration_seed); repeats outermost so a partial run already covers every size ##
CONFIGS = [(None, None)] + [(size, int(seed)) for seed in calibration_seeds for size in CALIBRATION_SIZES]

if __name__ == "__main__":
    ## one benchmark at a time: each holds its scored frames, so building them all up front
    ## does not fit in memory. Finished configurations already in TRIAL_PATH are skipped ##
    for size, seed in CONFIGS:
        benchmark = medPathBenchmark(
            resplit=RESPLIT, seed=SEED, split_size=SPLIT_SIZE, calibration_size=size, calibration_seed=seed
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
