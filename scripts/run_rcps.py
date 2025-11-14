from rcps_og.utils.constants import CALIBRATION_DATA_PATH
from rcps_og.utils.scores import gilda_score, fuzzy_string_score
from rcps_og.utils.losses import (
    binary_miscoverage_loss,
    weighted_binary_miscoverage_loss,
)
from rcps_og.utils.utils import load_calibration_and_validation
from typing import Tuple, Callable
import gilda
import polars as pl
import tqdm
import numpy as np


def load_calibration_and_validation() -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Reads in the calibration DF and splits it based on document id to prevent contamination"""
    calibration_df = pl.read_csv(CALIBRATION_DATA_PATH, separator="\t").filter(
        pl.col("db").eq("mesh")
    )
    documents = calibration_df.select("document_id").unique()
    validate_size = int(0.2 * len(documents))
    validation_document_ids, calibration_document_ids = documents.head(
        validate_size
    ), documents.tail(-validate_size)
    validation_df = calibration_df.filter(
        pl.col("document_id").is_in(
            validation_document_ids.to_numpy().flatten().tolist()
        )
    ).with_row_index()
    calibration_df = calibration_df.filter(
        pl.col("document_id").is_in(
            calibration_document_ids.to_numpy().flatten().tolist()
        )
    ).with_row_index()
    validation_df.write_csv("here_1.csv")
    return validation_df.sort(by="index"), calibration_df.sort(by="index")


def get_gilda_predictions(calibration_df: pl.DataFrame) -> dict:
    """Use gilda to ground each example"""
    index_to_candidates_map = {}
    for k, x in enumerate(calibration_df.iter_rows(named=True)):
        candidate_matches = gilda.ground(
            text=x["entity_raw_text"],
            context=x["full_text"],
        )
        if len(candidate_matches) > 0:
            index_to_candidates_map[x["index"]] = [
                {"entry_name": y.term.entry_name, "gilda_score": y.score}
                for y in candidate_matches
            ]
        else:
            index_to_candidates_map[x["index"]] = [
                {"entry_name": None, "gilda_score": None}
            ]
    return index_to_candidates_map


def get_gilda_prediction_stats(
    index_to_candidates_map: dict, candidate_cutoff: int = 5
) -> Tuple[list[int], list[int], list[Tuple[int, int]]]:
    """
    Return the number of cases where gilda found no matches, were it found one match and where it found above i matches
    """
    ## get just cases with more than i maps
    # counts = [len(index_to_candidates_map[x]) for x in index_to_candidates_map]
    counts = {
        idx: len(candidates) for idx, candidates in index_to_candidates_map.items()
    }

    no_hits = [x for x in counts if x == 0]
    one_hits = [x for x in counts if x == 1]
    above_i = [
        (idx, count) for idx, count in counts.items() if count > candidate_cutoff
    ]

    return no_hits, one_hits, above_i


def filter_calibration_data(
    calibration_df: pl.DataFrame, above_i: list[Tuple[int, int]]
) -> pl.DataFrame:
    """
    Returns a filtered data frame for just rows above a given cut off with a row index
    """
    above_i_indices = [x[0] for x in above_i]
    return calibration_df.filter(pl.col("index").is_in(above_i_indices))


def calc_empirical_risk(loss_df: pl.DataFrame, loss_name: str):
    return loss_df.select(pl.mean(loss_name)).item()


def evaluate_on_calibration_data(
    q: float,
    above_i_calibration_data: pl.DataFrame,
    index_to_candidates_map: dict,
    loss_function: Callable,
    score_function: Callable,
):
    scores = []
    losses = []
    for row in above_i_calibration_data.iter_rows(named=True):
        sample_idx = row["index"]
        candidate_terms = index_to_candidates_map[sample_idx]
        local_scores = []
        for candidate in candidate_terms:
            local_scores.append(
                {
                    "sample_index": sample_idx,
                    "entity_raw_text": row["entity_raw_text"],
                    "entry_name": candidate["entry_name"],
                    score_function.__name__: score_function(
                        entity=row, candidate=candidate
                    ),
                }
            )
        scores += local_scores
        ## weight threshold depending on the size of the candidate set
        q_star = adaptive_q(n_candidates=len(candidate_terms), base_q=q)
        ## check the loss for the candidate set.
        candidate_set = [
            x
            for x, y in zip(candidate_terms, local_scores)
            if y[score_function.__name__] >= q_star
        ]
        candidate_names = [x.get("entry_name", "") for x in candidate_set]
        losses.append(
            {
                "sample_index": sample_idx,
                "entity_raw_text": row.get("entity_raw_text", None),
                "entity_label": row.get("normalized_name", None),
                "candidates": candidate_names,
                "n_candidates": len(candidate_set),
                "n_total_candidates": len(candidate_terms),
                loss_function.__name__: loss_function(
                    entity=row, candidate=candidate_set
                ),
            }
        )
    score_df = pl.from_records(scores)
    loss_df = pl.from_records(losses)
    empirical_risk = calc_empirical_risk(
        loss_df=loss_df, loss_name=loss_function.__name__
    )
    return score_df, loss_df, empirical_risk


def calibration_evaluation_generator(
    above_i_calibration_data: pl.DataFrame,
    index_to_candidates_map: dict,
    loss_function: Callable,
    score_function: Callable,
):
    return lambda q: evaluate_on_calibration_data(
        q=q,
        above_i_calibration_data=above_i_calibration_data,
        index_to_candidates_map=index_to_candidates_map,
        loss_function=loss_function,
        score_function=score_function,
    )


def adaptive_q(n_candidates, base_q):
    if n_candidates > 5:
        return base_q  # Aggressive filtering is safe
    elif n_candidates > 3:
        return base_q * 0.8  # Be more conservative
    else:
        return base_q * 0.5  # Very conservative, or just return all


if __name__ == "__main__":
    alpha = 0.25
    q_range = [0, 1.0, 100]  # [min q val, max q val, # to search between]
    merge_score = (
        lambda entity, candidate: (
            fuzzy_string_score(entity, candidate) + gilda_score(entity, candidate)
        )
        / 2
    )
    score_func = merge_score
    loss_func = binary_miscoverage_loss
    candidate_cutoff = 5  # min number of candidates to consider
    ## run
    calibration_df, validation_df = load_calibration_and_validation()

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
    ## find cases where gilda and rcps prediction are not equal
    merged_loss.filter(
        ~(pl.col("binary_miscoverage_loss").eq(pl.col("binary_miscoverage_loss_gilda")))
    )  # 37
    print(merged_loss.mean())
    # by_entity = merged_loss.group_by('entity_raw_text').first()
    # by_entity.filter(~(pl.col('binary_miscoverage_loss').eq(pl.col('binary_miscoverage_loss_gilda'))))# 37
    # doc_loss = merged_loss.join(
    #     above_i_validation_data.select(['index', 'db', 'document_id']),
    #     left_on='sample_index',
    #     right_on='index'
    # )
    # merged_loss.join(
    #     above_i_validation_data.select(['index', 'db', 'document_id']),
    #     left_on='sample_index',
    #     right_on='index'
    # ).group_by(
    #     'db'
    # ).mean()

#     0.5076923076923077 12.307692307692308 0.33333333333333337
# 0.47692307692307695 13.923076923076923 0.32323232323232326
