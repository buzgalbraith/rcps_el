from rcps_el.dataset import Dataset, medCodERBenchmark
from rcps_el.scores import Scorer, MedCodErScorer, sapbertScorer
from rcps_el.losses import lossFunction, binaryMisscoverageLoss
from rcps_el.evaluators import rcpsELEvaluator

import matplotlib.pyplot as plt
import polars as pl

from re import sub

DATASET_NAME = "bioID_gilda"
# DATASET_NAME = "BCD5_krissbert"
# DATASET_NAME = 'bioRED'
# DATASET_NAME = 'BCD5'
RISK_TYPE = "relative"
DATASET_PLOT_NAME = (
    "Bio-ID" if DATASET_NAME.lower().startswith("bioid") else DATASET_NAME
)
METHOD_NAME = "Gilda" if DATASET_NAME.lower().endswith("gilda") else "KRISSBERT"
MIN_SAMPLES = 2

if METHOD_NAME == "Gilda":
    SCORES = [
        ("fuzzy_string_scores", "Fuzzy string score"),
        ("gilda_scores", "Gilda score"),
        ("SapBERT_scores", "SapBERT score"),
        ("LLM_scorer_batch_size1", "LLM score"),
    ]
else:
    SCORES = [
        ("fuzzy_string_scores", "Fuzzy string score"),
        ("KrissBERT_scores", "KrissBERT score"),
        ("SapBERT_scores", "SapBERT score"),
        ("LLM_scorer_batch_size1", "LLM score"),
    ]

COLORS = ["#4CAF50", "#2196F3", "#FF9800", "#E91E63"]
BASELINE_COLORS = {"original": "#555555", "expected": "#888888"}


def plot_risk(ax, risk_results, score_idx: int | None):
    risk_targets = risk_results["target_proportional_risk_increase"].unique().sort()
    orig_risk = risk_results["risk_original"][0]
    score_name = risk_results["score_function"][0]
    risk_name = risk_results["loss_function"][0]
    split_name = risk_results["split"][0]
    n_samples = risk_results["samples"][0]
    ax.plot(
        risk_results["target_proportional_risk_increase"],
        risk_results["risk_controlled"],
        label=score_name,
        color=COLORS[score_idx],
        linewidth=1.8,
    )
    ax.plot(
        risk_targets,
        [orig_risk] * len(risk_targets),
        "--",
        color=BASELINE_COLORS["original"],
        linewidth=1.2,
        label="Original risk",
    )
    ax.plot(
        risk_targets,
        [orig_risk * (1 + t) for t in risk_targets],
        ":",
        color=BASELINE_COLORS["expected"],
        linewidth=1.2,
        label="Expected risk",
    )
    raw_title = f"{split_name} set (n={n_samples}) {risk_name}"
    intermediate_title = sub("_loss.*", "_loss", raw_title)
    final_title = sub("_", " ", intermediate_title)
    ax.set_title(
        final_title,
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xlabel(
        "Target risk increase",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylabel(
        "Observed risk",
        fontsize=12,
        fontweight="bold",
    )
    ax.tick_params(labelsize=12)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def plot_cset(ax, cset_results, score_idx: int | None):
    risk_targets = cset_results["target_proportional_risk_increase"].unique().sort()
    score_name = cset_results["score_function"][0]
    risk_name = cset_results["loss_function"][0]
    orig_c_set = cset_results["c_set_size_original"]
    split_name = cset_results["split"][0]
    n_samples = cset_results["samples"][0]
    ax.plot(
        cset_results["target_proportional_risk_increase"],
        cset_results["c_set_size_controlled"],
        label=score_name,
        color=COLORS[score_idx],
        linewidth=1.8,
    )
    ax.plot(
        risk_targets,
        [orig_c_set] * len(risk_targets),
        linestyle="dashdot",
        color=BASELINE_COLORS["original"],
        linewidth=1.2,
        label="Original c-set size",
    )
    raw_title = f"{split_name} set (n={n_samples}) {risk_name}"
    intermediate_title = sub("_loss.*", "_loss", raw_title)
    final_title = sub("_", " ", intermediate_title)
    ax.set_title(
        final_title,
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xlabel(
        "Target risk increase",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylabel(
        "Mean candidate set size",
        fontsize=12,
        fontweight="bold",
    )
    ax.tick_params(labelsize=12)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def get_trail_subsets(
    trail_results: pl.DataFrame,
    benchmark: Dataset,
    scores: list[Scorer],
    loss: lossFunction,
    splits: list[str] = ["calibration", "validation"],
    absolute_risk: bool = False,
    min_candidates: int = 2,
    subplot_args: dict = {"figsize": (10, 4.5)},
):
    fig, axes = plt.subplots(len(splits), 2, **subplot_args)
    risk_type_str = "absolute" if absolute_risk else "relative"
    raw_title = f"{benchmark.method} risk control on {benchmark.name} Benchmark"
    fig.suptitle(
        sub("_", " ", raw_title),
        fontsize=15,
        fontweight="bold",
        y=0.8,
    )

    for score_index, score in enumerate(scores):
        for split_index, split in enumerate(splits):
            trail_subset = (
                trail_results.filter(pl.col("dataset").eq(benchmark.name))
                .filter(pl.col("source_method").eq(benchmark.method))
                .filter(pl.col("score_function").eq(score.name))
                .filter(pl.col("loss_function").eq(loss.name))
                .filter(pl.col("evaluation_strategy").eq(risk_type_str))
                .filter(pl.col("min_candidates").eq(min_candidates))
                .filter(pl.col("split").eq(split))
            ).sort("target_proportional_risk_increase")

            plot_risk(axes[split_index, 0], trail_subset, score_index)
            plot_cset(axes[split_index, 1], trail_subset, score_index)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper right",
        fontsize=11,
        bbox_to_anchor=(0.90, 0.85),
    )

    fig.tight_layout(rect=[0, 0.04, 0.88, 0.9])
    plt.savefig(f"{DATASET_NAME}_binary_coverage.png", dpi=150, bbox_inches="tight")
    fig.show()


if __name__ == "__main__":
    df = pl.read_csv("trials.tsv", separator="\t")
    ## need: dataset name ,method name, min samples, risk type, loss name, scores as a list.
    dataset = medCodERBenchmark(n_retrieved=10, billable=True, resplit=True)
    scores = [MedCodErScorer(), sapbertScorer()]
    # scores = [ sapbertScorer()]
    loss = binaryMisscoverageLoss()
    target_proportion_risks: list[float] = [0.00, 0.01, 0.02, 0.05, 0.10, 0.20, 0.25]
    min_candidates = 2
    absolute = True
    get_trail_subsets(
        trail_results=df,
        benchmark=dataset,
        scores=scores,
        loss=loss,
        min_candidates=2,
    )
