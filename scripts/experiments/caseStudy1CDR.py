"""
Code for running the BDC with KrissBERT Case study
"""
from pystow import module

from rcps_el.dataset import BCD5, Dataset
from rcps_el.scores import retrievalScorer, cumulativeRetrievalScorer, Scorer
from rcps_el.losses import binaryMisscoverageLoss, hitsAtK, lossFunction
from rcps_el.evaluators import rcpsELSetEvaluator

METHOD_NAME = "KRISSBERT Score"
TRIAL_PATH = module("rcps_el", "experiments").base.joinpath("trials_cdr.tsv")

DELTAS = [0.1]
TARGET_RISKS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.25]
RISK_FORMULATIONS: list[str] = ['relative', 'absolute']
SCORES: list[Scorer] = [retrievalScorer(name = METHOD_NAME), cumulativeRetrievalScorer(name = METHOD_NAME)]
K_VALUES: list[int] = [1,2,3,5,10]
DERIVED_HITS_AT_K: list[int] = [1, 3, 5, 10]

BENCHMARKS: list[Dataset] = [BCD5(method='krissbert')]
LOSSES: list[lossFunction] = [binaryMisscoverageLoss()] + [hitsAtK(k) for k in K_VALUES]

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


