"""
MedCodER case study on the 50/50 document-level resplit (213 validation mentions),
with exact-code and ICD-10 category losses under relative and absolute control.
Absolute runs are included as a check against trials_medcoder_story.tsv.
"""
import polars as pl
from pystow import module

from rcps_el.dataset import medCodERResplitBenchmark, Dataset
from rcps_el.dataset.medCodERBenchmark import fix_labels, MEDCODER_DIR as medcoder_module
from rcps_el.scores import retrievalScorer, cumulativeRetrievalScorer, Scorer
from rcps_el.losses import binaryMisscoverageLoss, commonAncestorsAtKLoss, lossFunction
from rcps_el.evaluators import rcpsELSetEvaluator

METHOD_NAME = "MedCodER Score"
TRIAL_PATH = module("rcps_el", "experiments").base.joinpath("trials_medcoder.tsv")

DELTAS = [0.1]
TARGET_RISKS = [0.0, 0.01, 0.05, 0.10, 0.20, 0.25]
RISK_FORMULATIONS: list[str] = ['relative', 'absolute']
SCORES: list[Scorer] = [retrievalScorer(name = METHOD_NAME), cumulativeRetrievalScorer(name = METHOD_NAME)]
LOSSES: list[lossFunction] = [binaryMisscoverageLoss(), commonAncestorsAtKLoss(prefix_len=3)]



BENCHMARKS: list[Dataset] = [medCodERResplitBenchmark()]

if __name__ == "__main__":
    evaluator = rcpsELSetEvaluator(
        benchmarks=BENCHMARKS,
            scores=SCORES,
            losses=LOSSES,
            results_path= TRIAL_PATH,
            target_risks=TARGET_RISKS,
            risk_formulations=RISK_FORMULATIONS,
            deltas=DELTAS,
            risk_unit='mention'
    )
    evaluator.execute(verbose=True)
