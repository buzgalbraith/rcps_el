from rcps_og.utils.constants import CALIBRATION_DATA_PATH
from rcps_og.utils.scores import gilda_score, fuzzy_string_score
from rcps_og.utils.losses import binary_miscoverage_loss
from typing import Tuple
import gilda
import polars as pl
import tqdm
import numpy as np

## read in calibration df and get an example
calibration_df = pl.read_csv(CALIBRATION_DATA_PATH, separator="\t").filter(
    pl.col("db").eq("mesh")
)


def get_gilda_predictions(calibration_df: pl.DataFrame) -> dict:
    """Use gilda to ground each example"""
    index_to_candidates_map = {}
    for k, x in enumerate(calibration_df.iter_rows(named=True)):
        candidate_matches = gilda.ground(
            text=x["entity_raw_text"],
            context=x["full_text"],
        )
        if len(candidate_matches) > 0:
            index_to_candidates_map[k] = [
                {"entry_name": y.term.entry_name, "gilda_score": y.score}
                for y in candidate_matches
            ]
        else:
            index_to_candidates_map[k] = [{"entry_name": None, "gilda_score": None}]
    return index_to_candidates_map


def get_gilda_prediction_stats(
    index_to_candidates_map: dict, candidate_cutoff: int = 5
) -> Tuple[list[int], list[int], list[Tuple[int, int]]]:
    """
    Return the number of cases where gilda found no matches, were it found one match and where it found above i matches
    """
    ## get just cases with more than i maps
    counts = [len(index_to_candidates_map[x]) for x in index_to_candidates_map]
    no_hits = [x for x in counts if x == 0]
    one_hits = [x for x in counts if x == 1]
    above_i = [(k, x) for k, x in enumerate(counts) if x > candidate_cutoff]
    return no_hits, one_hits, above_i


def filter_calibration_data(
    calibration_df: pl.DataFrame, above_i: list[Tuple[int, int]]
) -> pl.DataFrame:
    """
    Returns a filtered data frame for just rows above a given cut off with a row index
    """
    above_i_mask = [x[0] for x in above_i]
    index_calibration_df = calibration_df.with_row_index()
    return index_calibration_df[above_i_mask]


def calc_empirical_risk(loss_df: pl.DataFrame, loss_cols: list):
    return {
        loss_type: loss_df.select(pl.mean(loss_type)).item() for loss_type in loss_cols
    }


if __name__ == "__main__":
    alpha = 0.1
    q_initial = 1.0
    ## run
    index_to_candidates_map = get_gilda_predictions(calibration_df=calibration_df)
    no_hits, one_hits, above_i = get_gilda_prediction_stats(
        index_to_candidates_map=index_to_candidates_map, candidate_cutoff=5
    )
    above_i_calibration_data = filter_calibration_data(
        calibration_df=calibration_df, above_i=above_i
    )
    ## write all this as a function of q.
    scores = []
    losses = []
    for row in tqdm.tqdm(above_i_calibration_data.iter_rows(named=True)):
        sample_idx = row["index"]
        candidate_terms = index_to_candidates_map[sample_idx]
        local_scores = []
        for candidate in candidate_terms:
            local_scores.append(
                {
                    "sample_index": sample_idx,
                    "entity_raw_text": row["entity_raw_text"],
                    "entry_name": candidate["entry_name"],
                    "gilda_score": gilda_score(entity=row, candidate=candidate),
                    "fuzzy_string_score": fuzzy_string_score(
                        entity=row, candidate=candidate
                    ),
                }
            )
        scores += local_scores
        ## check the loss for the candidate set.
        candidate_set = [
            x
            for x, y in zip(candidate_terms, local_scores)
            if y["gilda_score"] < q_initial
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
                "binary_miscoverage": binary_miscoverage_loss(
                    entity=row, candidate=candidate_set, size_weighted=False
                ),
                "weighted_binary_miscoverage": binary_miscoverage_loss(
                    entity=row, candidate=candidate_set, size_weighted=True
                ),
            }
        )
    score_df = pl.from_records(scores)
    loss_df = pl.from_records(losses)
    empirical_risk = calc_empirical_risk(
        loss_df=loss_df, loss_cols=["binary_miscoverage", "weighted_binary_miscoverage"]
    )

    ## Total of 80 misses. out of 178 rows
    loss_df.filter(pl.col("binary_miscoverage").eq(1))
