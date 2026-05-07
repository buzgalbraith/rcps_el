import matplotlib.pyplot as plt
import polars as pl

DATASET_NAME = "BCD5_krissbert"
# DATASET_NAME = 'bioRED'
# DATASET_NAME = 'BCD5'
RISK_TYPE = "relative"
MIN_SAMPLES = 2
K_VALUES = [1, 2, 5, 10]

SCORES = [
    ("fuzzy_string_scores", "Fuzzy string score"),
    ("SapBERT_scores",      "SapBERT score"),
    ("KrissBERT_scores",    "KrissBERT score"),
]
SCORES = [
    ("KrissBERT_scores",    "KrissBERT score"),
]

COLORS = ["#4CAF50", "#2196F3", "#FF9800"]
BASELINE_COLORS = {"original": "#555555", "expected": "#888888"}


def plot_hits(ax, k, val, risk_targets, orig_risk):
    for (score_col, score_label), color in zip(SCORES, COLORS):
        score_rows = val.filter(pl.col("score").eq(score_col)).sort("target_risk")
        if score_rows.is_empty():
            continue
        ax.plot(score_rows["target_risk"], 1 - score_rows["controlled_risk"],
                label=score_label, color=color, linewidth=1.8)
    ax.plot(risk_targets, [orig_risk] * len(risk_targets),
            "--", color=BASELINE_COLORS["original"], linewidth=1.2, label="Original Hits@k")
    ax.plot(risk_targets, [1 - (1 - orig_risk) * (1 + t) for t in risk_targets],
            ":", color=BASELINE_COLORS["expected"], linewidth=1.2, label="Expected Hits@k")
    ax.set_title(f"Hits @ {k}", fontsize=15, fontweight="bold")
    ax.set_xlabel("Target risk increase", fontsize=12,fontweight="bold",)
    ax.set_ylabel(f"Hits@{k}", fontsize=12,fontweight="bold",)
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
    ax.set_xlabel("Target risk increase",fontsize=12,fontweight="bold",)
    ax.set_ylabel("Candidate set size",fontsize=12,fontweight="bold",)
    ax.tick_params(labelsize=10)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)


if __name__ == "__main__":
    df = pl.read_csv("trials.tsv", separator="\t")
    df = (
        df.filter(pl.col("dataset_name").eq(DATASET_NAME))
          .filter(pl.col("min_candidates").eq(MIN_SAMPLES))
          .filter(pl.col("risk_type").eq(RISK_TYPE))
          .filter(pl.col("loss_name").str.contains("Hits"))
        #   .filter(pl.col("split").eq("validation"))
          .sort("target_risk")
    )
    calibration_samples = df.filter(pl.col("split").eq("calibration"))["samples"][0]
    df = df.filter(pl.col("split").eq("validation"))
    validation_samples = df["samples"][0]

    fig, axes = plt.subplots(2, len(K_VALUES), figsize=(4 * len(K_VALUES), 6.5),
                             sharex=True)
    fig.suptitle(
        f"KRISSBERT risk control on BC5CDR Benchmark validation set (n = {validation_samples}) \n calibrated on {calibration_samples} samples.",
        fontsize=15, fontweight="bold",
        y=0.91,
    )

    legend_ax = None
    for col, k in enumerate(K_VALUES):
        val = df.filter(pl.col("loss_name").eq(f"Hits@{k} loss"))
        risk_targets = val["target_risk"].unique().sort()
        orig_risk = 1 - val["original_risk"][0]
        orig_c_set = val["original_c_set_size"][0]

        plot_hits(axes[0][col], k, val, risk_targets, orig_risk)
        plot_cset(axes[1][col], val, risk_targets, orig_c_set)
        legend_ax = axes[0][col]

    handles, labels = legend_ax.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper right",
        fontsize=12,
        bbox_to_anchor=(0.90, 0.95),
    )

    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    plt.savefig(f"{DATASET_NAME}_hits_at_k.png", dpi=150, bbox_inches="tight")
    fig.show()
