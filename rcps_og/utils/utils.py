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


def get_loss_df(
    score_df, score_function, q, loss_function, above_i_calibration_data
) -> pl.DataFrame:
    ## filter out scores bellow threshold
    scores_over_threshold = score_df.filter(pl.col(score_function.__name__) >= q)
    ## do some
    label_candidate_df = scores_over_threshold.group_by("index").agg(
        [
            pl.col("normalized_name").first().alias("normalized_name"),
            pl.col("entry_name").unique().alias("candidates"),
        ]
    )
    losses = label_candidate_df.with_columns(
        pl.struct(["normalized_name", "candidates"])
        .map_elements(
            return_dtype=pl.Float64,
            function=lambda x: loss_function(
                x.get("normalized_name"), x.get("candidates")
            ),
            skip_nulls=True,
        )
        .alias(loss_function.__name__),
        n_candidates=pl.col("candidates").list.len(),
    )
    total_candidates = score_df.group_by("index").len("n_total_candidates")
    return (
        above_i_calibration_data.join(
            losses.drop("normalized_name"), on="index", how="left", validate="1:m"
        )
        .with_columns(
            [
                pl.col("n_candidates").fill_null(0),
                pl.col(loss_function.__name__).fill_null(1),
            ]
        )
        .join(total_candidates, on="index", validate="1:m")
    )


def get_candidates_df(index_to_candidates_map: dict) -> pl.DataFrame:
    """process a candidates dictionary to a polars data frame"""
    records = []
    for sample_idx in index_to_candidates_map:
        entry = index_to_candidates_map[sample_idx]
        for record in entry:
            record.update({"index": sample_idx})
            records.append(record)
    return pl.from_dicts(records)


def evaluate_on_calibration_data(
    q: float,
    above_i_calibration_data: pl.DataFrame,
    index_to_candidates_map: dict,
    loss_function: Callable,
    score_function: Callable,
    processing_function: Callable = None,
):
    raw_text_col = "entity_raw_text"
    entity_name_col = "entry_name"
    ## index to candidate map to dataframe
    candidate_df = above_i_calibration_data.join(
        get_candidates_df(index_to_candidates_map=index_to_candidates_map), on="index"
    )
    ## if score function requires additional processing run that here
    if processing_function is not None:
        candidate_df, raw_text_col, entity_name_col = processing_function(candidate_df)
    ## apply score function to df
    if score_function.__name__ != "gilda_score":
        score_df = candidate_df.with_columns(
            pl.struct([raw_text_col, entity_name_col])
            .map_elements(
                return_dtype=pl.Float64,
                function=lambda x: score_function(
                    entity=x.get(raw_text_col), candidate=x.get(entity_name_col)
                ),
            )
            .alias(score_function.__name__)
        )
    else:  ## gilda scores computed in step 1
        score_df = candidate_df
    ## get losses df
    loss_df = get_loss_df(
        score_df=score_df,
        score_function=score_function,
        q=q,
        loss_function=loss_function,
        above_i_calibration_data=above_i_calibration_data,
    )
    ## calculate empirical risk
    empirical_risk = calc_empirical_risk(
        loss_df=loss_df, loss_name=loss_function.__name__
    )
    return score_df, loss_df, empirical_risk


def calibration_evaluation_generator(
    above_i_calibration_data: pl.DataFrame,
    index_to_candidates_map: dict,
    loss_function: Callable,
    score_function: Callable,
    processing_function: Callable = None,
):
    return lambda q: evaluate_on_calibration_data(
        q=q,
        above_i_calibration_data=above_i_calibration_data,
        index_to_candidates_map=index_to_candidates_map,
        loss_function=loss_function,
        score_function=score_function,
        processing_function=processing_function,
    )


def adaptive_q(n_candidates, base_q):
    if n_candidates > 5:
        return base_q  # Aggressive filtering is safe
    elif n_candidates > 3:
        return base_q * 0.8  # Be more conservative
    else:
        return base_q * 0.5  # Very conservative, or just return all
