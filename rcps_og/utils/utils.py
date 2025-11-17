"""
Utility functions
"""

from rcps_og.utils.constants import CALIBRATION_DATA_PATH
import polars as pl
from typing import Tuple, Callable
import gilda


def load_calibration_and_validation(
    validate_prop: float = 0.2,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Reads in the calibration DF and splits it based on document id to prevent contamination"""
    calibration_df = pl.read_csv(CALIBRATION_DATA_PATH, separator="\t")
    documents = calibration_df.select("document_id").unique()
    validate_size = int(validate_prop * len(documents))
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
    ## testing
    calibration_df = pl.read_csv(
        "/Users/buzgalbraith/.data/BioRED/phenotype_train.tsv", separator="\t"
    )
    validate_size = int(validate_prop * len(calibration_df))
    validation_df, calibration_df = (
        calibration_df.head(validate_size).with_row_index(),
        calibration_df.tail(-validate_size).with_row_index(),
    )
    return calibration_df.sort(by="index"), validation_df.sort(by="index")


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
                    "db": row["db"],
                    score_function.__name__: score_function(
                        entity=row, candidate=candidate
                    ),
                }
            )
        scores += local_scores
        ## weight threshold depending on the size of the candidate set
        # q_star = adaptive_q(n_candidates=len(candidate_terms), base_q=q)
        q_star = q
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
                "db": row["db"],
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
