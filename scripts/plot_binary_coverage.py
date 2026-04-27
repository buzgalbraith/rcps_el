import matplotlib.pyplot as plt
import polars as pl

DATASET_NAME = "bioID_gilda"
# DATASET_NAME = "BCD5_krissbert"
# DATASET_NAME = 'bioRED'
# DATASET_NAME = 'BCD5'
RISK_TYPE = "relative"
MIN_SAMPLES = 10


if __name__ == "__main__":
    df = pl.read_csv("trials.tsv", separator="\t").unique()
    df = (
        df.filter(pl.col("dataset_name").eq(DATASET_NAME))
        .filter(pl.col("min_candidates").eq(MIN_SAMPLES))
        .filter(pl.col("risk_type").eq(RISK_TYPE))
        .filter(pl.col('loss_name').eq('binary_misscoverage_loss_safe_min_aggregation'))
    )
    fig, ax = plt.subplots(2, 2)
    calibration_res = df.filter(pl.col("split").eq("calibration")).sort(
        by="target_risk"
    )
    validation_res = df.filter(pl.col("split").eq("validation")).sort(by="target_risk")
    calibration_samples = calibration_res["samples"][0]
    validation_samples = validation_res["samples"][0]
    risk_targets = calibration_res["target_risk"].unique()
    ## calibration risk
    orig_risk_calibration = calibration_res["original_risk"][0]

    
    ax[0][0].plot(
        risk_targets,
        calibration_res.filter(pl.col("score").eq("fuzzy_string_scores"))[
            "controlled_risk"
        ],
        label="Fuzzy string score",
        alpha=0.5,
    )
    # ax[0][0].plot(risk_targets, calibration_res.filter(pl.col('score').eq('gilda_scores'))['controlled_risk'], label = 'Gilda score', alpha = 0.5)
    ax[0][0].plot(
        risk_targets,
        calibration_res.filter(pl.col("score").eq("SapBERT_scores"))["controlled_risk"],
        label="SapBERT score",
        alpha=0.5,
    )
    # ax[0][0].plot(
    #     risk_targets,
    #     calibration_res.filter(pl.col("score").eq("KrissBERT_scores"))["controlled_risk"],
    #     label="KrissBert score",
    #     alpha=0.5,
    # )
    ax[0][0].plot(
        risk_targets,
        calibration_res.filter(pl.col("score").eq("LLM_scorer"))["controlled_risk"],
        label="LLM score",
        alpha=0.5,
    )
    ax[0][0].plot(
        risk_targets,
        [orig_risk_calibration for _ in risk_targets],
        "--",
        label="Original risk",
    )
    ax[0][0].plot(
        risk_targets,
        [orig_risk_calibration * (1 + tgt) for tgt in risk_targets],
        ":",
        label="expected risk",
    )
    ax[0][0].set_xlabel("Target proportional risk increase")
    ax[0][0].set_ylabel("Observed risk")
    ax[0][0].set_title(
        f"{DATASET_NAME} calibration set {RISK_TYPE} risk with min candidate set size = {MIN_SAMPLES}, Samples = {calibration_samples}"
    )
    ## validation risk
    orig_risk_validation = validation_res["original_risk"][0]
    ax[0][1].plot(
        risk_targets,
        validation_res.filter(pl.col("score").eq("fuzzy_string_scores"))[
            "controlled_risk"
        ],
        alpha=0.5,
    )
    # ax[0][1].plot(risk_targets, validation_res.filter(pl.col('score').eq('gilda_scores'))['controlled_risk'], alpha = 0.5)
    ax[0][1].plot(
        risk_targets,
        validation_res.filter(pl.col("score").eq("SapBERT_scores"))["controlled_risk"],
        alpha=0.5,
    )
    # ax[0][1].plot(
    #     risk_targets,
    #     validation_res.filter(pl.col("score").eq("KrissBERT_scores"))["controlled_risk"],
    #     alpha=0.5,
    # )
    ax[0][1].plot(
        risk_targets,
        validation_res.filter(pl.col("score").eq("LLM_scorer"))["controlled_risk"],
        alpha=0.5,
    )
    ax[0][1].plot(
        risk_targets,
        [orig_risk_validation for _ in risk_targets],
        "--",
    )
    ax[0][1].plot(
        risk_targets,
        [orig_risk_validation * (1 + tgt) for tgt in risk_targets],
        ":",
    )
    ax[0][1].set_xlabel("Target proportional risk increase")
    ax[0][1].set_ylabel("Observed risk")
    ax[0][1].set_title(
        f"{DATASET_NAME} validation set {RISK_TYPE} risk with min candidate set size = {MIN_SAMPLES}, Samples = {validation_samples}"
    )
    ## calibration candidate set size
    orig_candidate_set_calibration = calibration_res["original_c_set_size"][0]
    ax[1][0].plot(
        risk_targets,
        calibration_res.filter(pl.col("score").eq("fuzzy_string_scores"))[
            "controlled_c_set_size"
        ],
        alpha=0.5,
    )
    # ax[1][0].plot(risk_targets, calibration_res.filter(pl.col('score').eq('gilda_scores'))['controlled_c_set_size'],  alpha = 0.5)
    ax[1][0].plot(
        risk_targets,
        calibration_res.filter(pl.col("score").eq("SapBERT_scores"))[
            "controlled_c_set_size"
        ],
        alpha=0.5,
    )
    # ax[1][0].plot(
    #     risk_targets,
    #     calibration_res.filter(pl.col("score").eq("KrissBERT_scores"))[
    #         "controlled_c_set_size"
    #     ],
    #     alpha=0.5,
    # )
    ax[1][0].plot(
        risk_targets,
        calibration_res.filter(pl.col("score").eq("LLM_scorer"))[
            "controlled_c_set_size"
        ],
        alpha=0.5,
    )
    ax[1][0].plot(
        risk_targets,
        calibration_res.filter(pl.col("score").eq("LLM_scorer"))[
            "controlled_c_set_size"
        ],
        alpha=0.5,
    )
    ax[1][0].plot(
        risk_targets,
        [orig_candidate_set_calibration for _ in risk_targets],
        linestyle="dashdot",
        label="Original candidate set size",
    )
    ax[1][0].set_xlabel("Target proportional risk increase")
    ax[1][0].set_ylabel("Observed candidate set size")
    ax[1][0].set_title(
        f"{DATASET_NAME} calibration set {RISK_TYPE} control with min candidate set size = {MIN_SAMPLES}, Samples = {calibration_samples}"
    )
    ## validation candidate set size
    orig_candidate_set_validation = validation_res["original_c_set_size"][0]
    ax[1][1].plot(
        risk_targets,
        validation_res.filter(pl.col("score").eq("fuzzy_string_scores"))[
            "controlled_c_set_size"
        ],
        alpha=0.5,
    )
    # ax[1][1].plot(risk_targets, validation_res.filter(pl.col('score').eq('gilda_scores'))['controlled_c_set_size'], alpha = 0.5)
    ax[1][1].plot(
        risk_targets,
        validation_res.filter(pl.col("score").eq("SapBERT_scores"))[
            "controlled_c_set_size"
        ],
        alpha=0.5,
    )
    # ax[1][1].plot(
    #     risk_targets,
    #     validation_res.filter(pl.col("score").eq("KrissBERT_scores"))[
    #         "controlled_c_set_size"
    #     ],
    #     alpha=0.5,
    # )
    ax[1][1].plot(
        risk_targets,
        validation_res.filter(pl.col("score").eq("LLM_scorer"))[
            "controlled_c_set_size"
        ],
        alpha=0.5,
    )
    ax[1][1].plot(
        risk_targets,
        [orig_candidate_set_validation for _ in risk_targets],
        linestyle="dashdot",
    )
    ax[1][1].set_xlabel("Target proportional risk increase")
    ax[1][1].set_ylabel("Observed candidate set size")
    ax[1][1].set_title(
        f"{DATASET_NAME} validation set {RISK_TYPE} control with min candidate set size = {MIN_SAMPLES}, Samples = {validation_samples}"
    )
    fig.legend()
    fig.show()
