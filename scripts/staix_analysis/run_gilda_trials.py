"""
Re-run the Gilda trials on the BioID dataset and produce the corresponding visualization, output written to `~/.data/rcps_el/trials_gilda.tsv`

Notes:
    - Uses BioID with method = 'gilda' this means directly use the Gilda results from the original paper (https://academic.oup.com/bioinformaticsadvances/article/2/1/vbac034/6584365?login=true)
    - llmScorer ran with gpt-oss:20b OpenAI model
    - Uses llmScorer with batch_size = 1 ie LLM looks at one (term, candidate_entity) pair at a time 
    - Uses llmScorer with use_titles = False ie LLM does not receive additional context from paper title.
"""
from rcps_el.dataset import bioIDBenchmark, Dataset
from rcps_el.scores import gildaScorer, llmScorer, sapbertScorer, fuzzyStringScore, Scorer
from rcps_el.losses import binaryMisscoverageLoss, lossFunction
from rcps_el.evaluators import rcpsELSetEvaluator

import matplotlib.pyplot as plt
import polars as pl
from pystow import module
import matplotlib

from pathlib import Path
from re import sub


## keep vector output editable in Illustrator: text stays as text (not paths),
## nothing is rasterized, and fonts embed as editable type in SVG/PDF ##
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["image.composite_image"] = False

## write to figs dir ##
FIGS_DIR = Path(__file__).resolve().parent.parent.parent / "figs"
SCORES : list[Scorer] = [fuzzyStringScore(), gildaScorer(), sapbertScorer(), llmScorer(use_titles=False, batch_size=1)]
LOSS : lossFunction = binaryMisscoverageLoss()
BENCHMARK : Dataset = bioIDBenchmark(method='gilda')

TRIAL_PATH = module("rcps_el").base.joinpath("trials_gilda.tsv")
FORCE_RERUN = False  ## set True to recompute even when RESULTS_PATH exists



## per-score line colors (LLM is brown, matching gilda.png) ##
COLORS = ["#4CAF50", "#2196F3", "#FF9800", "#8D6E63"]
## black dashed original baseline, magenta dotted expected baseline ##
BASELINE_COLORS = {"original": "#000000", "expected": "#E91E63"}

## human-friendly display labels ##
SCORE_LABELS = {
    "fuzzy_string_scores": "Fuzzy string score",
    "gilda_scores": "Gilda score",
    "SapBERT_scores": "SapBERT score",
    "KrissBERT_scores": "KrissBERT score",
    "MedCodER_scores": "MedCodER score",
}
LOSS_LABELS = {"binary_misscoverage_loss_safe_min_aggregation": "Binary miscoverage"}
BENCHMARK_LABELS = {"bioID": "Bio-ID"}

## font sizes tuned to match gilda.png ##
text_mod = 3 
SUPTITLE_SIZE = 20 + text_mod
TITLE_SIZE = 18 + text_mod
LABEL_SIZE = 17 + text_mod
TICK_SIZE = 13 + text_mod
LEGEND_SIZE = 13 + text_mod
## Line widths ##
LW = 6
lw_risk = LW 
lw_bl = LW + 0.5

def score_label(name: str) -> str:
    """Map a raw score_function name to a display label."""
    if name in SCORE_LABELS:
        return SCORE_LABELS[name]
    if name.startswith("LLM_scorer"):
        return "LLM score"
    return sub("_", " ", name)


def loss_label(name: str) -> str:
    """Map a raw loss_function name to a display label."""
    return LOSS_LABELS.get(name, sub("_loss.*", "", sub("_", " ", name)).title())


def _style_axis(ax, title: str, ylabel: str):
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_xlabel("Tolerated risk increase (α)", fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE, fontweight="bold")
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def plot_risk(ax, risk_results, score_idx: int, label: str):
    """Plot a single score's controlled-risk curve."""
    ax.plot(
        risk_results["target_proportional_risk_increase"],
        risk_results["risk_controlled"],
        label=label,
        color=COLORS[score_idx],
        linewidth=lw_risk,
    )


def plot_risk_baselines(ax, risk_results):
    """Draw the original / expected risk baselines once."""
    risk_targets = risk_results["target_proportional_risk_increase"]
    orig_risk = risk_results["risk_original"][0]
    ax.plot(
        risk_targets,
        [orig_risk] * len(risk_targets),
        "--",
        color=BASELINE_COLORS["original"],
        linewidth=lw_bl,
        label="Original model risk ( R(0) ) ",
    )
    ax.plot(
        risk_targets,
        [orig_risk * (1 + t) for t in risk_targets],
        ":",
        color=BASELINE_COLORS["expected"],
        linewidth=lw_bl,
        label="Max tolerated risk ( R(0) * (1 + α) )",
    )


def plot_cset(ax, cset_results, score_idx: int, label: str):
    """Plot a single score's controlled candidate-set-size curve."""
    ax.plot(
        cset_results["target_proportional_risk_increase"],
        cset_results["c_set_size_controlled"],
        label=label,
        color=COLORS[score_idx],
        linewidth=lw_risk,
    )


def plot_cset_baseline(ax, cset_results):
    """Draw the original candidate-set-size baseline once."""
    risk_targets = cset_results["target_proportional_risk_increase"]
    orig_c_set = cset_results["c_set_size_original"][0]
    ax.plot(
        risk_targets,
        [orig_c_set] * len(risk_targets),
        linestyle="dashdot",
        color=BASELINE_COLORS["original"],
        linewidth=lw_bl,
        label="Original CS size",
    )


def get_trail_subsets(
    trail_results: pl.DataFrame,
    benchmark: Dataset,
    scores: list[Scorer],
    loss: lossFunction,
    split: str = "validation",
    absolute_risk: bool = False,
    min_candidates: int = 2,
    subplot_args: dict = {"figsize": (11, 4.8)},
    output_format: str = 'png'
):
    fig, axes = plt.subplots(1, 2, **subplot_args)
    risk_type_str = "absolute" if absolute_risk else "relative"

    ## shared filter for this benchmark / method / loss / strategy ##
    base = (
        trail_results.filter(pl.col("dataset").eq(benchmark.name))
        .filter(pl.col("source_method").eq(benchmark.method))
        .filter(pl.col("loss_function").eq(loss.name))
        .filter(pl.col("evaluation_strategy").eq(risk_type_str))
        .filter(pl.col("min_candidates").eq(min_candidates))
    )

    ref_subset = None
    for score_index, score in enumerate(scores):
        subset = (
            base.filter(pl.col("score_function").eq(score.name))
            .filter(pl.col("split").eq(split))
            .sort("target_proportional_risk_increase")
        )
        if subset.is_empty():
            continue
        label = score_label(score.name)
        plot_risk(axes[0], subset, score_index, label)
        plot_cset(axes[1], subset, score_index, label)
        ref_subset = subset

    ## baselines drawn once (identical across scores) so the legend stays clean ##
    if ref_subset is not None:
        plot_risk_baselines(axes[0], ref_subset)
        plot_cset_baseline(axes[1], ref_subset)

    _style_axis(axes[0], f"{loss_label(loss.name)} risk", "Observed risk")
    _style_axis(axes[1], "Mean candidate set (CS) size", "Mean CS size")

    bench_label = BENCHMARK_LABELS.get(benchmark.name, benchmark.name)
    suptitle = (
        f"{benchmark.method.title()} risk control on {bench_label} benchmark across score functions"
    )
    fig.suptitle(suptitle, fontsize=SUPTITLE_SIZE, fontweight="bold")

    ## single framed legend, centered below both panels (dedupe by label) ##
    handles, labels = axes[0].get_legend_handles_labels()
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

    fig.tight_layout(rect=[0, 0.10, 1, 0.92])
    ## pdf keeps text/vectors editable in Illustrator without the per-character
    ## <tspan> and deep group nesting that bloat matplotlib's svg output ##
    if output_format not in ("png", "svg", "pdf"):
        raise ValueError(f"{output_format} not recognized")

    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGS_DIR / f"{benchmark.name}_binary_coverage.{output_format}"
    ## dpi only affects png rasters; svg is vector and ignores it ##
    plt.savefig(out_path, dpi=150, bbox_inches="tight", format=output_format)
    plt.close(fig)
    print(f"wrote {out_path}")
    # fig.show()


if __name__ == "__main__":

    ## run the set evaluator (or load cached results) ##
    if FORCE_RERUN or not TRIAL_PATH.exists():
        evaluator = rcpsELSetEvaluator(
            benchmarks=[BENCHMARK],
            scores=SCORES,
            losses=[LOSS],
            results_path=TRIAL_PATH
        )
        evaluator.execute()
        trail_results = evaluator.result_set
    else:
        trail_results = pl.read_csv(TRIAL_PATH, separator="\t")

    ## plot results ## 
    get_trail_subsets(
        trail_results=trail_results,
        benchmark=BENCHMARK,
        scores=SCORES,
        loss=LOSS,
        output_format='pdf'
    )
