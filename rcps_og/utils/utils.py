from rcps_og.utils.constants import CALIBRATION_DATA_PATH
import polars as pl
from typing import Tuple


def load_calibration_and_validation(
    seed: int = 151873,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Reads in the calibration DF and splits it based on document id to prevent contamination"""
    calibration_df = pl.read_csv(CALIBRATION_DATA_PATH, separator="\t")
    documents = calibration_df.select("document_id").unique()
    documents = documents.sample(fraction=1, shuffle=True, seed=52)
    validate_size = int(0.2 * len(documents))
    validation_document_ids, calibration_document_ids = documents.head(
        validate_size
    ), documents.tail(-validate_size)
    validation_df = calibration_df.filter(
        pl.col("document_id").is_in(
            validation_document_ids.to_numpy().flatten().tolist()
        )
    )
    calibration_df = calibration_df.filter(
        pl.col("document_id").is_in(
            calibration_document_ids.to_numpy().flatten().tolist()
        )
    )
    return validation_df, calibration_df
