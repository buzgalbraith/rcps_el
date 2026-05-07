from rcps_el import rcpsELEvaluator, rcpsELSetEvaluator
from rcps_el.dataset import bioIDBenchmark, bioRedBenchmark, BCD5, Dataset
from rcps_el.dataset.bioIDGilda import bioIDGildaBenchmark
from rcps_el.scores import fuzzyStringScore, gildaScorer, sapbertScorer, krissbertScorer, Scorer
from rcps_el.losses import binaryMisscoverageLoss, lossFunction

# BENCHMARKS: list[Dataset] = [bioIDBenchmark(), bioRedBenchmark(), BCD5()]
# SCORES: list[Scorer] = [fuzzyStringScore(), gildaScorer(), sapbertScorer()]
# LOSSES: list[lossFunction] = [binaryMisscoverageLoss()]
# EVALUATIONS = [True, False]

## TODO Remove ##
# BENCHMARKS: list[Dataset] = [bioIDBenchmark()]
BENCHMARKS: list[Dataset] = [BCD5(method='krissbert')]
SCORES: list[Scorer] = [krissbertScorer()]
LOSSES: list[lossFunction] = [binaryMisscoverageLoss()]
EVALUATIONS = [False]
## end REMOVE ##

def main():
    evaluators = []
    for dataset in BENCHMARKS:
        for loss_function in LOSSES:
            for score_function in SCORES:
                for is_absolute in EVALUATIONS:
                    # dataset.calibration_set = dataset.validation_set
                    # dataset.validation_set = dataset.test_set
                    evaluators.append(
                        rcpsELEvaluator(
                            dataset=dataset,
                            score_function=score_function,
                            loss_function=loss_function,
                            min_candidates=10,
                            absolute_risk=is_absolute,
                            target_proportional_risk_increase=0.20,
                        )
                    )
    set_evaluator = rcpsELSetEvaluator(evaluators=evaluators)
    set_evaluator.execute()
    evaluator = set_evaluator.evaluators[0]
    evaluator.results_summary[1]

    # evaluator.result_validation_original.select(pl.col('n_candidates')).write_csv('validation.csv', separator='\t')
    # evaluator.result_calibration_original.select(pl.col('n_candidates')).write_csv('calibration.csv', separator='\t')
    evaluator.result_validation_original['n_candidates'].value_counts().write_csv('og_validation.csv', separator='\t')
    evaluator.result_calibration_original['n_candidates'].value_counts().write_csv('og_calibration.csv', separator='\t')
    # tmp = evaluator.result_validation_original['gilda_scores'].explode().value_counts()
    # import polars as pl
    # tmp.with_columns(
    #     proportion = pl.col('count') / len( evaluator.result_validation_original['gilda_scores'].explode())
    # ).sort(by='count', descending=True).write_csv('here.csv')
    # x = evaluator.result_validation_original['gilda_scores'].explode()
    # y = x.filter(x> evaluator.q_star)
    # len(y)/len(x)
    # tmp = evaluator.result_calibration_original['gilda_scores'].explode().value_counts()
    # import polars as pl
    # tmp.with_columns(
    #     proportion = pl.col('count') / len( evaluator.result_calibration_original['gilda_scores'].explode())
    # ).sort(by='count', descending=True).write_csv('calibration.csv')
    # x = evaluator.result_calibration_original['gilda_scores'].explode()
    # y = x.filter(x> evaluator.q_star)
    # len(y)/len(x)
if __name__ == "__main__":
    main()
    
