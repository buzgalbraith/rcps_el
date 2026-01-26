from rcps_og import rcpsOGEvaluator, rcpsOGSetEvaluator
from rcps_og.dataset import bioIDBenchmark, bioRedBenchmark, Dataset
from rcps_og.scores import fuzzyStringScore, gildaScorer, sapbertScorer, Scorer
from rcps_og.losses import binaryMisscoverageLoss, lossFunction

BENCHMARKS: list[Dataset] = [bioIDBenchmark(), bioRedBenchmark()]
SCORES: list[Scorer] = [fuzzyStringScore(), gildaScorer(), sapbertScorer()]
LOSSES: list[lossFunction] = [binaryMisscoverageLoss()]
EVALUATIONS = [True, False]


def main():
    evaluators = []
    for dataset in BENCHMARKS:
        for loss_function in LOSSES:
            for score_function in SCORES:
                for is_absolute in EVALUATIONS:
                    evaluators.append(
                        rcpsOGEvaluator(
                            dataset=dataset,
                            score_function=score_function,
                            loss_function=loss_function,
                            min_candidates=5,
                            absolute_risk=is_absolute,
                            target_proportional_risk_increase=0.05,
                        )
                    )
    set_evaluator = rcpsOGSetEvaluator(evaluators=evaluators)
    set_evaluator.execute()


if __name__ == "__main__":
    main()
