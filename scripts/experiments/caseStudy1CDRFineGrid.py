"""
Fine alpha grid for the one CDR configuration plotted in scripts/plot_cdr_table.py:
relative risk control, cumulative KRISSBERT score (S2), binary miss-coverage loss.

Writes into the same results file as caseStudy1CDR.py. The set evaluator only appends
rows not already present, so the alphas that run already covered are skipped here.
"""
from pystow import module

from rcps_el.dataset import BCD5, Dataset
from rcps_el.scores import cumulativeRetrievalScorer, Scorer
from rcps_el.losses import binaryMisscoverageLoss, lossFunction
from rcps_el.evaluators import rcpsELSetEvaluator

METHOD_NAME = "KRISSBERT Score"
TRIAL_PATH = module("rcps_el", "experiments").base.joinpath("trials_cdr.tsv")

DELTAS = [0.1]
## 0.00 to 0.30 in steps of 0.01, rounded so keys match the coarse run's alphas ##
ALREADY_RUN = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.25]
TARGET_RISKS = [
    alpha for alpha in (round(i / 100, 2) for i in range(31))
    if alpha not in ALREADY_RUN
]
RISK_FORMULATIONS: list[str] = ['relative']
SCORES: list[Scorer] = [cumulativeRetrievalScorer(name = METHOD_NAME)]
DERIVED_HITS_AT_K: list[int] = [1, 3, 5, 10]

BENCHMARKS: list[Dataset] = [BCD5(method='krissbert')]
LOSSES: list[lossFunction] = [binaryMisscoverageLoss()]

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
    evaluator.execute(verbose=False)
