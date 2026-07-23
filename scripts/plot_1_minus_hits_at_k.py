from rcps_el.dataset import Dataset, medCodERBenchmark
from rcps_el.scores import Scorer, MedCodErScorer, sapbertScorer

import matplotlib.pyplot as plt
import polars as pl


SCORES = [
    ("fuzzy_string_scores", "Fuzzy string score"),
    ("SapBERT_scores",      "SapBERT score"),
    ("KrissBERT_scores",    "KrissBERT score"),
    ("MedCodER_scores",    "MedCodER score"),
]

# reference-line styling, kept consistent across every subplot
ORIGINAL_RISK_STYLE = dict(color="0.3",  linestyle="--",      linewidth=1.5)
EXPECTED_RISK_STYLE = dict(color="crimson", linestyle=":",    linewidth=1.5)
ORIGINAL_CSET_STYLE = dict(color="0.3",  linestyle="dashdot", linewidth=1.5)


def _style_ax(ax):
    """Light grid + de-cluttered spines for a cleaner look."""
    ax.grid(True, which="major", linewidth=0.5, alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


def plot_k(subfig, k, calibration_res, validation_res, calibration_samples,
           validation_samples, scores, color_map):
    loss = f"Hits@{k} loss"
    subfig.suptitle(f"1-Hits @ {k}", fontsize=13, fontweight="bold")
    axes = subfig.subplots(2, 2)

    cal = calibration_res.filter(pl.col("loss_function").eq(loss))
    val = validation_res.filter(pl.col("loss_function").eq(loss))
    risk_targets = cal["target_proportional_risk_increase"].unique().sort()

    orig_risk_cal  = cal["risk_original"][0]
    orig_risk_val  = val["risk_original"][0]
    orig_c_set_cal = cal["c_set_size_original"][0]
    orig_c_set_val = val["c_set_size_original"][0]

    # ------------------------------------------------------------------
    # Row 0  –  Hits@k risk  (calibration | validation)
    # ------------------------------------------------------------------
    for ax, split, orig_risk, n in (
        (axes[0][0], cal, orig_risk_cal, calibration_samples),
        (axes[0][1], val, orig_risk_val, validation_samples),
    ):
        for score in scores:
            score_col = score.name
            score_rows = split.filter(pl.col("score_function").eq(score_col)).sort("target_proportional_risk_increase")
            if score_rows.is_empty():
                continue
            x = score_rows["target_proportional_risk_increase"]
            ax.plot(x, score_rows["risk_controlled"], label=score_col,
                    color=color_map[score_col], marker="o", markersize=4, linewidth=2)
        ax.plot(risk_targets, [orig_risk for _ in risk_targets],
                label="Original risk", **ORIGINAL_RISK_STYLE)
        # hits@k loss: risk_controlled should stay below orig * (1 + tgt)
        ax.plot(risk_targets, [orig_risk * (1 + tgt) for tgt in risk_targets],
                label="Expected risk", **EXPECTED_RISK_STYLE)
        ax.set_xlabel("Target proportional risk increase")
        ax.set_ylabel(f"Hits@{k} risk")
        _style_ax(ax)

    axes[0][0].set_title(f"Calibration  (n = {calibration_samples})", fontsize=9)
    axes[0][1].set_title(f"Validation   (n = {validation_samples})",  fontsize=9)

    # ------------------------------------------------------------------
    # Row 1  –  Candidate set size  (calibration | validation)
    # ------------------------------------------------------------------
    for ax, split, orig_c_set in (
        (axes[1][0], cal, orig_c_set_cal),
        (axes[1][1], val, orig_c_set_val),
    ):
        for score in scores:
            score_col = score.name
            score_rows = split.filter(pl.col("score_function").eq(score_col)).sort("target_proportional_risk_increase")
            if score_rows.is_empty():
                continue
            x = score_rows["target_proportional_risk_increase"]
            ax.plot(x, score_rows["c_set_size_controlled"], label=score_col,
                    color=color_map[score_col], marker="o", markersize=4, linewidth=2)
        ax.plot(risk_targets, [orig_c_set for _ in risk_targets],
                label="Original c-set size", **ORIGINAL_CSET_STYLE)
        ax.set_xlabel("Target proportional risk increase")
        ax.set_ylabel("Candidate set size")
        _style_ax(ax)

    axes[1][0].set_title(f"Calibration  (n = {calibration_samples})", fontsize=9)
    axes[1][1].set_title(f"Validation   (n = {validation_samples})",  fontsize=9)

    return axes   # return all axes for complete legend extraction


def plot_1_minus_hits_at_k(
    dataset: Dataset,
    scores: list[Scorer],
    risk_type: str = "relative",
    min_samples: int = 20,
    k_values: list[int] = [1, 2, 5, 10],
    trials_path: str = "trials.tsv",
):
    df = pl.read_csv(trials_path, separator="\t")
    df = (
        df.filter(pl.col("dataset").eq(dataset.name))
            .filter(pl.col("min_candidates").eq(min_samples))
          .filter(pl.col("source_method").eq(dataset.method))
          .filter(pl.col("evaluation_strategy").eq(risk_type))
    )

    calibration_res = df.filter(pl.col("split").eq("calibration")).sort("target_proportional_risk_increase")
    validation_res  = df.filter(pl.col("split").eq("validation")).sort("target_proportional_risk_increase")
    calibration_samples = calibration_res["samples"][0]
    validation_samples  = validation_res["samples"][0]

    # one stable color per score, shared across every subplot
    palette = plt.get_cmap("tab10").colors
    color_map = {score.name: palette[i % len(palette)] for i, score in enumerate(scores)}

    fig = plt.figure(figsize=(28, 24))
    fig.suptitle(
        f"{dataset.name}  |  {risk_type} risk  |  min candidates = {min_samples}",
        fontsize=18, fontweight="bold", y=0.99,
    )
    subfigs = fig.subfigures(2, 2, hspace=0.08, wspace=0.06)

    legend_axes = None
    for idx, k in enumerate(k_values):
        sf = subfigs[idx // 2][idx % 2]
        legend_axes = plot_k(sf, k, calibration_res, validation_res,
                             calibration_samples, validation_samples, scores, color_map)

    # collect a complete, de-duplicated set of legend entries across all axes
    handles, labels = [], []
    seen = set()
    for ax in legend_axes.flatten():
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in seen:
                seen.add(label)
                handles.append(handle)
                labels.append(label)
    fig.legend(
        handles, labels,
        loc="lower center", ncol=len(labels), frameon=False, fontsize=12,
        bbox_to_anchor=(0.5, -0.01),
    )
    plt.savefig(f"{dataset.name}_hits_at_k.png", dpi=150, bbox_inches="tight")
    fig.show()


if __name__ == "__main__":
    dataset = medCodERBenchmark(n_retrieved=20, resplit=False, billable=True)
    scores = [MedCodErScorer(), sapbertScorer()]
    plot_1_minus_hits_at_k(
        dataset = dataset,
        scores = scores, 
        risk_type="relative",
        min_samples=2,
        k_values=[1,2,5,10],
    )
