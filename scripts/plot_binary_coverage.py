import matplotlib.pyplot as plt
import polars as pl

DATASET_NAME = "bioID_gilda"
# DATASET_NAME = "BCD5_krissbert"
# DATASET_NAME = 'bioRED'
# DATASET_NAME = 'BCD5'
RISK_TYPE = "relative"
DATASET_PLOT_NAME = "Bio-ID" if DATASET_NAME.lower().startswith("bioid") else DATASET_NAME
METHOD_NAME = "Gilda" if DATASET_NAME.lower().endswith("gilda") else "KRISSBERT"
MIN_SAMPLES = 2

if METHOD_NAME == "Gilda":
    SCORES = [
        ("fuzzy_string_scores",     "Fuzzy string score"),
        ("gilda_scores",            "Gilda score"),
        ("SapBERT_scores",          "SapBERT score"),
        ("LLM_scorer_batch_size1",  "LLM score"),
    ]
else:
    SCORES = [
        ("fuzzy_string_scores",     "Fuzzy string score"),
        ("KrissBERT_scores",        "KrissBERT score"),
        ("SapBERT_scores",          "SapBERT score"),
        ("LLM_scorer_batch_size1",  "LLM score"),
    ]

COLORS = ["#4CAF50", "#2196F3", "#FF9800", "#E91E63"]
BASELINE_COLORS = {"original": "#555555", "expected": "#888888"}


def plot_risk(ax, val, risk_targets, orig_risk):
    for (score_col, score_label), color in zip(SCORES, COLORS):
        score_rows = val.filter(pl.col("score").eq(score_col)).sort("target_risk")
        if score_rows.is_empty():
            continue
        ax.plot(score_rows["target_risk"], score_rows["controlled_risk"],
                label=score_label, color=color, linewidth=1.8)
    ax.plot(risk_targets, [orig_risk] * len(risk_targets),
            "--", color=BASELINE_COLORS["original"], linewidth=1.2, label="Original risk")
    ax.plot(risk_targets, [orig_risk * (1 + t) for t in risk_targets],
            ":", color=BASELINE_COLORS["expected"], linewidth=1.2, label="Expected risk")
    ax.set_title("Binary miscoverage risk", fontsize=15, fontweight="bold", )
    ax.set_xlabel("Target risk increase", fontsize=12,fontweight="bold", )
    ax.set_ylabel("Observed risk", fontsize=12,fontweight="bold",)
    ax.tick_params(labelsize=12)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def plot_cset(ax, val, risk_targets, orig_c_set):
    for (score_col, score_label), color in zip(SCORES, COLORS):
        score_rows = val.filter(pl.col("score").eq(score_col)).sort("target_risk")
        if score_rows.is_empty():
            continue
        ax.plot(score_rows["target_risk"], score_rows["controlled_c_set_size"],
                label=score_label, color=color, linewidth=1.8)
    ax.plot(risk_targets, [orig_c_set] * len(risk_targets),
            linestyle="dashdot", color=BASELINE_COLORS["original"], linewidth=1.2,
            label="Original c-set size")
    ax.set_title("Binary miscoverage candidate set sizes", fontsize=15, fontweight="bold")
    ax.set_xlabel("Target risk increase", fontsize=12,fontweight="bold",)
    ax.set_ylabel("Mean candidate set size", fontsize=12,fontweight="bold",)
    ax.tick_params(labelsize=12)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)


if __name__ == "__main__":
    df = pl.read_csv("trials.tsv", separator="\t").unique()
    df = (
        df.filter(pl.col("dataset_name").eq(DATASET_NAME))
          .filter(pl.col("min_candidates").eq(MIN_SAMPLES))
          .filter(pl.col("risk_type").eq(RISK_TYPE))
          .filter(pl.col("loss_name").eq("binary_misscoverage_loss_safe_min_aggregation"))
          .sort("target_risk")
    )
    calibration_samples = df.filter(pl.col("split").eq("calibration"))["samples"][0]
    val = df.filter(pl.col("split").eq("validation"))
    validation_samples = val["samples"][0]

    risk_targets = val["target_risk"].unique().sort()
    orig_risk = val["original_risk"][0]
    orig_c_set = val["original_c_set_size"][0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle(
        f"{METHOD_NAME} risk control on {DATASET_PLOT_NAME} Benchmark validation set (n = {validation_samples}) \n calibrated on {calibration_samples} samples.",
        fontsize=15, fontweight="bold",
        y=0.8,
    )

    plot_risk(axes[0], val, risk_targets, orig_risk)
    plot_cset(axes[1], val, risk_targets, orig_c_set)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper right",
        fontsize=11,
        bbox_to_anchor=(0.90, 0.85),
    )

    fig.tight_layout(rect=[0, 0.04, 0.88, 0.9])
    plt.savefig(f"{DATASET_NAME}_binary_coverage.png", dpi=150, bbox_inches="tight")
    fig.show()
