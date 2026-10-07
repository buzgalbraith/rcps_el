
"""
Code for running the MedPath with SapBert Case study
"""
from pystow import module

from rcps_el.dataset import medPathBenchmark, Dataset
from rcps_el.scores import retrievalScorer, cumulativeRetrievalScorer, Scorer
from rcps_el.losses import ancestorsAtK, descendantsAtK, hierarchyAtK, binaryMisscoverageLoss,  lossFunction
from rcps_el.evaluators import rcpsELSetEvaluator

METHOD_NAME = "MedPath Score"
RESPLIT: bool = True
TRIAL_PATH = module("rcps_el", "experiments").base.joinpath("trials_medpath_resplit.tsv" if RESPLIT else "trials_medpath.tsv")

DELTAS = [0.1]
TARGET_RISKS = [0.0, 0.01,  0.05, 0.10, 0.20,  0.25]
RISK_FORMULATIONS: list[str] = ['relative', ]
SCORES: list[Scorer] = [retrievalScorer(name = METHOD_NAME), cumulativeRetrievalScorer(name = METHOD_NAME)]
K_VALUES: list[int] = [1,2,5]
DERIVED_HITS_AT_K: list[int] = [10]

SEED = 10022
SPLIT_SIZE = 0.3

BENCHMARKS: list[Dataset] = [medPathBenchmark(resplit=RESPLIT, seed = SEED, split_size= SPLIT_SIZE,)]
LOSSES: list[lossFunction] = [binaryMisscoverageLoss()] + [ancestorsAtK(k, k_candidates = False) for k in K_VALUES] + [descendantsAtK(k , k_candidates = False) for k in K_VALUES] + [hierarchyAtK(k, k_candidates = False) for k in K_VALUES]

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


