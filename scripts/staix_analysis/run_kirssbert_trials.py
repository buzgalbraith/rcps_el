"""
Re-run the KrissBERT trials on the BCD5 dataset and produce the corresponding visualization, output written to `~/.data/rcps_el/trials_krissbert.tsv`

Notes:
    - Uses BCD5 with method = 'krissbert' this means directly use the krissbert results

"""
from rcps_el.dataset import BCD5, Dataset
from rcps_el.scores import krissbertScorer, Scorer
from rcps_el.losses import hitsAtK, lossFunction
from rcps_el.evaluators import rcpsELSetEvaluator

import matplotlib
import matplotlib.pyplot as plt
import polars as pl
from pystow import module

from pathlib import Path

## keep vector output editable in Illustrator: text stays as text (not paths),
## nothing is rasterized, and fonts embed as editable type in SVG/PDF ##
plt.rcParams['pdf.fonttype'] = 42 
# matplotlib.rcParams["image.composite_image"] = False

## write to figs dir ##
FIGS_DIR = Path(__file__).resolve().parent.parent.parent / "figs"

## evaluation configuration ##
BENCHMARK: Dataset = BCD5(method="krissbert")
SCORES: list[Scorer] = [krissbertScorer()]
K_VALUES = [1, 2, 5, 10]
LOSSES: list[lossFunction] = [hitsAtK(k) for k in K_VALUES]

TRIAL_PATH = module("rcps_el").base.joinpath("trials_krissbert.tsv")
FORCE_RERUN = False  ## set True to recompute even when TRIAL_PATH exists

RISK_TYPE = "relative"
MIN_CANDIDATES = 2
OUTPUT_FORMAT = "pdf"

## display labels ##
DATASET_LABEL = "BC5CDR"
METHOD_LABEL = "KRISSBERT"
SCORE_LABELS = {"KrissBERT_scores": "KRISSBERT score"}

## green score line, black dashed original baseline, magenta dotted tolerated ##
COLORS = ["#4CAF50", "#2196F3", "#FF9800"]
BASELINE_COLORS = {"original": "#000000", "expected": "#E91E63"}

## line widths / font sizes tuned to match KRISSBERT_Fig.png ##
LW = 6
SCORE_LW = LW
BASELINE_LW = LW + 0.5
text_mod = 3
SUPTITLE_SIZE = 20 + text_mod
TITLE_SIZE = 18 + text_mod
LABEL_SIZE = 17 + text_mod
TICK_SIZE = 13 + text_mod
LEGEND_SIZE = 15 + text_mod
## shared y-axis range for the Hits@k (risk) row; set to None for auto-scaling ##
HITS_YLIM = (0.22, 0.36)


def score_label(name: str) -> str:
    """Map a raw score_function name to a display label."""
    return SCORE_LABELS.get(name, name)


def plot_hits(ax, k, val, risk_targets, orig_hits, col_number):
    for score_index, score in enumerate(SCORES):
        score_rows = val.filter(pl.col("score_function").eq(score.name)).sort(
            "target_proportional_risk_increase"
        )
        if score_rows.is_empty():
            continue
        ax.plot(
            score_rows["target_proportional_risk_increase"],
            # 1 - score_rows["risk_controlled"],
            score_rows["risk_controlled"],
            label=score_label(score.name),
            color=COLORS[score_index],
            linewidth=SCORE_LW,
        )
    ax.plot(
        risk_targets,
        [orig_hits] * len(risk_targets),
        "--",
        color=BASELINE_COLORS["original"],
        linewidth=BASELINE_LW,
        label="Original model risk ( R(0) )",
    )
    ax.plot(
        risk_targets,
        [(orig_hits) * (1 + t) for t in risk_targets],
        # [1 - (1 - orig_hits) * (1 + t) for t in risk_targets],
        ":",
        color=BASELINE_COLORS["expected"],
        linewidth=BASELINE_LW,
        label="Max tolerated risk ( R(0) * (1 + α) )",
    )
    if HITS_YLIM is not None:
        ax.set_ylim(*HITS_YLIM)
    # ax.set_title(f"Hits @ {k}", fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_title(f"Loss = 1 - Hits @ {k}", fontsize=TITLE_SIZE, fontweight="bold")
    if col_number == 0:
        # ax.set_ylabel(f"Hits@k", fontsize=LABEL_SIZE, fontweight="bold")
        ax.set_ylabel(f"Observed risk", fontsize=LABEL_SIZE, fontweight="bold")
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def plot_cset(ax, val, risk_targets, orig_c_set, col_number):
    for score_index, score in enumerate(SCORES):
        score_rows = val.filter(pl.col("score_function").eq(score.name)).sort(
            "target_proportional_risk_increase"
        )
        if score_rows.is_empty():
            continue
        ax.plot(
            score_rows["target_proportional_risk_increase"],
            score_rows["c_set_size_controlled"],
            label=score_label(score.name),
            color=COLORS[score_index],
            linewidth=SCORE_LW,
        )
    ax.plot(
        risk_targets,
        [orig_c_set] * len(risk_targets),
        linestyle="dashdot",
        color=BASELINE_COLORS["original"],
        linewidth=BASELINE_LW,
        label="Original CS size",
    )
    ax.set_xlabel("Tolerated risk increase (α)", fontsize=LABEL_SIZE-2, fontweight="bold")
    if col_number == 0:
        ax.set_ylabel("Mean CS size", fontsize=LABEL_SIZE, fontweight="bold")
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def plot_hits_at_k(trial_results: pl.DataFrame, output_format: str = "png"):
    df = (
        trial_results.filter(pl.col("dataset").eq(BENCHMARK.name))
        .filter(pl.col("source_method").eq(BENCHMARK.method))
        .filter(pl.col("min_candidates").eq(MIN_CANDIDATES))
        .filter(pl.col("evaluation_strategy").eq(RISK_TYPE))
        .filter(pl.col("loss_function").str.contains("Hits"))
        .sort("target_proportional_risk_increase")
    )
    df = df.filter(pl.col("split").eq("validation"))

    fig, axes = plt.subplots(2, len(LOSSES), figsize=(4 * len(LOSSES), 6.5), sharex=True)
    fig.suptitle(
        f"{METHOD_LABEL} risk control on {DATASET_LABEL} benchmark validation\n"
        f"set across 1 - Hits@K loss targets",
        fontsize=SUPTITLE_SIZE,
        fontweight="bold",
    )

    legend_ax = None
    for col, loss in enumerate(LOSSES):
        val = df.filter(pl.col("loss_function").eq(loss.name))
        risk_targets = val["target_proportional_risk_increase"].unique().sort()
        # orig_hits = 1 - val["risk_original"][0]
        orig_hits = val["risk_original"][0]
        orig_c_set = val["c_set_size_original"][0]

        plot_hits(axes[0][col], loss.k_size, val, risk_targets, orig_hits, col)
        plot_cset(axes[1][col], val, risk_targets, orig_c_set, col)
        legend_ax = axes[0][col]

    ## single framed legend, centered below all panels (dedupe by label) ##
    handles, labels = legend_ax.get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="lower center",
        ncol=3,
        fontsize=LEGEND_SIZE,
        frameon=True,
        bbox_to_anchor=(0.5, -0.04),
    )

    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    ## pdf keeps text/vectors editable in Illustrator without the per-character
    ## <tspan> and deep group nesting that bloat matplotlib's svg output ##
    if output_format not in ("png", "svg", "pdf"):
        raise ValueError(f"{output_format} not recognized")

    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGS_DIR / f"{BENCHMARK.name}_hits_at_k.{output_format}"
    ## dpi only affects png rasters; svg is vector and ignores it ##
    plt.savefig(out_path, dpi=150, bbox_inches="tight", format=output_format)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    ## run the set evaluator (or load cached results) ##
    if FORCE_RERUN or not TRIAL_PATH.exists():
        evaluator = rcpsELSetEvaluator(
            benchmarks=[BENCHMARK],
            scores=SCORES,
            losses=LOSSES,
            results_path=TRIAL_PATH,
        )
        evaluator.execute()
        trial_results = evaluator.result_set
    else:
        trial_results = pl.read_csv(TRIAL_PATH, separator="\t")

    plot_hits_at_k(trial_results, output_format=OUTPUT_FORMAT)
