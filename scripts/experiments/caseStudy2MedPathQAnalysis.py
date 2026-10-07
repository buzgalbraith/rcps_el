
"""
Code for reproducing Q analysis on MedCoder dataset
"""
from pystow import module
from numpy.random import default_rng

from rcps_el.dataset import medPathBenchmark, Dataset
from rcps_el.scores import retrievalScorer, cumulativeRetrievalScorer, Scorer
from rcps_el.losses import ancestorsAtK, descendantsAtK, hierarchyAtK, binaryMisscoverageLoss,  lossFunction
from rcps_el.evaluators import rcpsELSetEvaluator

METHOD_NAME = "MedPath Score"
RESPLIT: bool = True
TRIAL_PATH = module("rcps_el", "experiments").base.joinpath("trials_medpath_QAnalysis.tsv" )

DELTAS = [0.01, 0.05, 0.10, 0.20, 0.25]
# DELTAS = [0.1]
# TARGET_RISKS = [0.0, 0.05, 0.10, 0.20,  0.25]
TARGET_RISKS = [ 0.10,]
RISK_FORMULATIONS: list[str] = ['relative', ]
SCORES: list[Scorer] = [retrievalScorer(name = METHOD_NAME), cumulativeRetrievalScorer(name = METHOD_NAME)]
DERIVED_HITS_AT_K: list[int] = [10]
LOSSES: list[lossFunction] = [ descendantsAtK(k_size = 1, k_candidates = False), binaryMisscoverageLoss() ]

SEED = 10022
NUM_TRIALS = 101
rng = default_rng(seed = SEED)
seeds = rng.integers(10**4, size = NUM_TRIALS)
SPLIT_SIZE = 0.3
BENCHMARKS = []
val_sizes = []
cal_sizes = []
for seed in seeds:
    BENCHMARKS.append(medPathBenchmark(resplit= RESPLIT, seed = seed, split_size = SPLIT_SIZE))
    val_sizes.append(len(BENCHMARKS[-1].validation_set))
    cal_sizes.append(len(BENCHMARKS[-1].calibration_set))

if __name__ == "__main__":
    evaluator = rcpsELSetEvaluator(
        benchmarks=BENCHMARKS,
            scores=SCORES,
            losses=LOSSES,
            results_path= TRIAL_PATH,
            target_risks=TARGET_RISKS,
            risk_formulations=RISK_FORMULATIONS,
            deltas=DELTAS, 
            derived_hits_at_k=DERIVED_HITS_AT_K,
    )
    evaluator.execute(verbose=True)

#
