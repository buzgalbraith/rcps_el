import matplotlib.pyplot as plt
import polars as pl

DATASET_NAME = "BCD5_krissbert"
# DATASET_NAME = 'bioRED'
# DATASET_NAME = 'BCD5'
RISK_TYPE = "relative"
MIN_SAMPLES = 20
K_VALUES = [1, 2, 5, 10]

SCORES = [
    ("fuzzy_string_scores", "Fuzzy string score"),
    ("SapBERT_scores",      "SapBERT score"),
    ("KrissBERT_scores",    "KrissBERT score"),
]


def plot_k(subfig, k, calibration_res, validation_res, calibration_samples, validation_samples):
    loss = f"Hits@{k} loss"
    subfig.suptitle(f"Hits @ {k}", fontsize=13, fontweight="bold")
    axes = subfig.subplots(2, 2)

    cal = calibration_res.filter(pl.col("loss_name").eq(loss))
    val = validation_res.filter(pl.col("loss_name").eq(loss))
    risk_targets = cal["target_risk"].unique().sort()

    orig_risk_cal  = 1 - cal["original_risk"][0]
    orig_risk_val  = 1 - val["original_risk"][0]
    orig_c_set_cal = cal["original_c_set_size"][0]
    orig_c_set_val = val["original_c_set_size"][0]

    # ------------------------------------------------------------------
    # Row 0  –  Hits@k risk  (calibration | validation)
    # ------------------------------------------------------------------
    for ax, split, orig_risk, n in (
        (axes[0][0], cal, orig_risk_cal, calibration_samples),
        (axes[0][1], val, orig_risk_val, validation_samples),
    ):
        for score_col, score_label in SCORES:
            score_rows = split.filter(pl.col("score").eq(score_col)).sort("target_risk")
            if score_rows.is_empty():
                continue
            x = score_rows["target_risk"]
            ax.plot(x, 1 - score_rows["controlled_risk"], label=score_label, alpha=0.7)
        ax.plot(risk_targets, [orig_risk for _ in risk_targets],
                "--", label="Original Hits@k")
        ax.plot(risk_targets, [1 - (1 - orig_risk) * (1 + tgt) for tgt in risk_targets],
                ":", label="Expected Hits@k")
        ax.set_xlabel("Target proportional risk increase")
        ax.set_ylabel(f"Hits@{k}")

    axes[0][0].set_title(f"Calibration  (n = {calibration_samples})", fontsize=9)
    axes[0][1].set_title(f"Validation   (n = {validation_samples})",  fontsize=9)

    # ------------------------------------------------------------------
    # Row 1  –  Candidate set size  (calibration | validation)
    # ------------------------------------------------------------------
    for ax, split, orig_c_set in (
        (axes[1][0], cal, orig_c_set_cal),
        (axes[1][1], val, orig_c_set_val),
    ):
        for score_col, score_label in SCORES:
            score_rows = split.filter(pl.col("score").eq(score_col)).sort("target_risk")
            if score_rows.is_empty():
                continue
            x = score_rows["target_risk"]
            ax.plot(x, score_rows["controlled_c_set_size"], label=score_label, alpha=0.7)
        ax.plot(risk_targets, [orig_c_set for _ in risk_targets],
                linestyle="dashdot", label="Original c-set size")
        ax.set_xlabel("Target proportional risk increase")
        ax.set_ylabel("Candidate set size")

    axes[1][0].set_title(f"Calibration  (n = {calibration_samples})", fontsize=9)
    axes[1][1].set_title(f"Validation   (n = {validation_samples})",  fontsize=9)

    return axes[0][0]   # return one ax for shared legend extraction


if __name__ == "__main__":
    df = pl.read_csv("trials.tsv", separator="\t")
    df = (
        df.filter(pl.col("dataset_name").eq(DATASET_NAME))
          .filter(pl.col("min_candidates").eq(MIN_SAMPLES))
          .filter(pl.col("risk_type").eq(RISK_TYPE))
            .filter(pl.col('loss_name').str.contains('Hits'))
    )

    calibration_res = df.filter(pl.col("split").eq("calibration")).sort("target_risk")
    validation_res  = df.filter(pl.col("split").eq("validation")).sort("target_risk")
    calibration_samples = calibration_res["samples"][0]
    validation_samples  = validation_res["samples"][0]

    fig = plt.figure(figsize=(28, 24))
    fig.suptitle(
        f"{DATASET_NAME}  |  {RISK_TYPE} risk  |  min candidates = {MIN_SAMPLES}",
        fontsize=15, fontweight="bold", y=1.01,
    )
    subfigs = fig.subfigures(2, 2, hspace=0.08, wspace=0.06)

    legend_ax = None
    for idx, k in enumerate(K_VALUES):
        sf = subfigs[idx // 2][idx % 2]
        legend_ax = plot_k(sf, k, calibration_res, validation_res,
                           calibration_samples, validation_samples)

    handles, labels = legend_ax.get_legend_handles_labels()
    fig.legend(
        handles, labels,
    )

    plt.savefig(f"{DATASET_NAME}_hits_at_k.png", dpi=150, bbox_inches="tight")
    fig.show()