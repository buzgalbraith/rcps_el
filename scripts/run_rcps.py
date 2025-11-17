from rcps_og.utils.constants import CALIBRATION_DATA_PATH
from rcps_og.utils.scores import gilda_score, fuzzy_string_score
from rcps_og.utils.losses import (
    binary_miscoverage_loss,
)
from rcps_og.utils.utils import (
    load_calibration_and_validation,
    get_gilda_predictions,
    get_gilda_prediction_stats,
    filter_calibration_data,
    calibration_evaluation_generator,
)
import polars as pl
import numpy as np


if __name__ == "__main__":
    alpha = 0.78
    q_range = [0, 1.0, 100]  # [min q val, max q val, # to search between]
    merge_score = (
        lambda entity, candidate: (
            fuzzy_string_score(entity, candidate) + gilda_score(entity, candidate)
        )
        / 2
    )
    score_func = merge_score
    loss_func = binary_miscoverage_loss
    candidate_cutoff = 1  # min number of candidates to consider
    ## run
    calibration_df, validation_df = load_calibration_and_validation(validate_prop=0.2)
    # calibration_df =

    index_to_candidates_map = get_gilda_predictions(calibration_df=calibration_df)

    no_hits, one_hits, above_i = get_gilda_prediction_stats(
        index_to_candidates_map=index_to_candidates_map,
        candidate_cutoff=candidate_cutoff,
    )
    # After filtering
    above_i_calibration_data = filter_calibration_data(
        calibration_df=calibration_df, above_i=above_i
    )
    # get a function to evaluate the calibration set as a function of q only.
    calibration_evaluator = calibration_evaluation_generator(
        above_i_calibration_data=above_i_calibration_data,
        index_to_candidates_map=index_to_candidates_map,
        loss_function=loss_func,
        score_function=score_func,
    )
    ##  raise q -> smaller set. so walk from most to least strict sets
    for q in sorted(
        np.linspace(start=q_range[0], stop=q_range[1], num=q_range[2]), reverse=True
    ):
        _, loss_df, empirical_risk = calibration_evaluator(q=q)
        candidates = loss_df.select(pl.mean("n_candidates")).item()
        print(empirical_risk, candidates, q)
        if empirical_risk <= alpha:
            break

    ## evaluate on validation dataset
    validation_index_to_candidates_map = get_gilda_predictions(
        calibration_df=validation_df
    )
    validation_no_hits, validation_one_hits, validation_above_i = (
        get_gilda_prediction_stats(
            index_to_candidates_map=validation_index_to_candidates_map,
            candidate_cutoff=candidate_cutoff,
        )
    )
    above_i_validation_data = filter_calibration_data(
        calibration_df=validation_df, above_i=validation_above_i
    )
    validation_evaluator = calibration_evaluation_generator(
        above_i_calibration_data=above_i_validation_data,
        index_to_candidates_map=validation_index_to_candidates_map,
        loss_function=loss_func,
        score_function=score_func,
    )
    ## evaluate with risk control
    score_df, loss_df, empirical_risk = validation_evaluator(q=q)
    ## evaluate with out risk control
    _, loss_df_orig_gilda, _ = validation_evaluator(q=0.0)

    ## merge
    merged_loss = loss_df.join(
        loss_df_orig_gilda.select([loss_func.__name__, "sample_index"]),
        on=pl.col("sample_index"),
        suffix="_gilda",
    )
    print(merged_loss.mean())
    merged_loss.filter(
        pl.col("binary_miscoverage_loss").eq(1)
        & pl.col("binary_miscoverage_loss_gilda").eq(0)
    )
