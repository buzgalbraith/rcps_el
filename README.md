# Risk Controlled Prediction Sets for Biomedical Entity Linking (RCPS-EL)

RCPS-EL is a modular framework for constructing calibrated candidate sets for biomedical entity linking (EL). Building on Risk-Controlling Prediction Sets (RCPS), it converts the ranked candidate output of a pre-trained retrieval-based EL model into a smaller set that carries a finite-sample guarantee on the risk of dropping the correct grounding.

## Overview

Biomedical EL models return ranked candidate lists with no formal guarantee that the correct grounding is among the candidates retained. RCPS-EL calibrates a score threshold $\hat{q}$ on held-out data so that a user-chosen risk is controlled at level $\alpha$ with probability at least $1 - \delta$, while making $\mathbb{E}[|T_q(m)|]$ as small as that constraint allows.

Two formulations are available, selected with `risk_formulation`:

**Absolute** — control the risk directly.

$$\hat{q} = \sup\\{q : \hat{R}^{+}(q) \le \alpha\\}$$

**Relative** (default) — control the *increase* in risk over the model's uncalibrated output $T_{q_0}(m)$.

$$\frac{R(q) - R(q_0)}{R(q_0)} \le \alpha$$

The relative formulation exists because the true label is often absent from the retrieved candidates to begin with. That irreducible risk $R(q_0)$ lower-bounds $R(q)$ for every threshold, so when $R(q_0) > \alpha$ no threshold can satisfy absolute control at all. The relative formulation still narrows sets in that regime, at the cost of a guarantee stated relative to the base model rather than in absolute terms.

### How the guarantee is obtained

The constraint is enforced against a finite-sample **upper confidence bound** on the risk, not the empirical risk. Selecting on the empirical risk stops exactly where sampling noise happens to be favourable, which lets the realised risk exceed the target roughly half the time; bounding it is what buys the $1 - \delta$ statement.

For the **absolute** formulation the loss is already normalised to $[0,1]$, so the bound applies to it directly.

For the **relative** formulation, $R(q_0)$ is itself estimated on the calibration set, so the constraint has a random right-hand side. Rather than bound the numerator alone, the per-unit paired quantity

$$Z_i = L_i(q) - (1 + \alpha) L_i(q_0) \in [-(1+\alpha),\, 1]$$

is formed, making the constraint exactly $\mathbb{E}[Z] \le 0$ — a single bounded mean with no random denominator left in it. $Z$ is rescaled onto $[0,1]$ and bounded with either the Waudby-Smith–Ramdas betting bound (default; variance-adaptive and tightest here) or the Hoeffding–Bentkus bound of Bates et al. (2021).

Two further details matter for validity:

- **Monotonicity and multiplicity.** RCPS may spend the full $\delta$ at every threshold only when the risk is monotone in $q$, which makes the violating thresholds a suffix of the grid and reduces the bad event to a single deterministic threshold. The evaluator sorts every candidate list into descending score order before computing the loss (`get_original_results`), so `score >= q` always retains a prefix, the top $k$ of a prefix is a prefix of the top $k$, and a top-$k$ loss can only grow as $q$ rises. Monotonicity is therefore structural rather than checked per dataset, and the search is a fixed-sequence procedure: walk from the loosest threshold toward the strictest and stop at the first failure. Continuing past a failure would require a Bonferroni correction.
- **Unit of concentration.** Splits are by document and mentions within a document are correlated, so pooling mentions overstates the effective sample size. `risk_unit` defaults to `"document"`, averaging the loss within each document before bounding. `"mention"` is available but voids the guarantee under within-document correlation.

### When RCPS-EL declines to narrow

If no threshold can be certified at the requested $\alpha$ and $\delta$, the uncalibrated candidate set is returned — it satisfies the constraint trivially — `q_star_certified` is set to `False`, and a warning is logged. This happens at both ends of the $\alpha$ range, for different reasons:

- **$\alpha$ below the irreducible risk** (absolute formulation only). If $R(q_0) > \alpha$ then $R(q) \ge R(q_0) > \alpha$ for every $q$, so no threshold exists. Use the relative formulation, or raise $\alpha$ above $R(q_0)$.
- **$\alpha$ near zero** (either formulation). At $q_0$ the margin between the constraint and its boundary is $\alpha R(q_0) / (2 + \alpha)$, which vanishes as $\alpha \to 0$, while the width of the confidence bound does not. Below a dataset-dependent floor the bound cannot detect that the constraint holds even when it holds deterministically. Raising $\delta$ or calibrating on more documents lowers the floor.

**Always check `q_star_certified` before reading `risk_controlled`.** A declined run reports the uncalibrated risk, which will sit above the target — that is the method abstaining, not a breach of the guarantee.

## Components

**Score functions** (`rcps_el.scores`). The framework is built around scores the retrieval model already produces, so that the score defining the threshold is the same one that produced the ranking:

- `retrievalScorer` — the base model's own retrieval similarity, optionally softmax-normalised over the retrieved candidates (`normalized_score=True`, with `softmax_temperature`).
- `cumulativeRetrievalScorer` — an adaptive, APS-style score: softmax-normalise the retrieved similarities, then score each candidate by the probability mass at or beyond its rank. Confident mentions exhaust the mass quickly and get small sets; ambiguous ones accumulate slowly and get larger ones.

Both read the `match_scores` column. Further scores can be added by extending `Scorer`.

**Loss functions** (`rcps_el.losses`). `binaryMisscoverageLoss`, `hitsAtK`, and three ontology-aware losses that credit a prediction for landing near the true concept in the hierarchy — `ancestorsAtK`, `descendantsAtK`, `hierarchyAtK`. Extend `lossFunction` to add more.

**Datasets** (`rcps_el.dataset`). `bioIDBenchmark`, `bioRedBenchmark`, `BCD5`, `medCodERBenchmark`, `medPathBenchmark`. Extend `Dataset` to add your own predictions.

> **Note on splits.** `bioIDBenchmark` and `bioRedBenchmark` partition documents at random under a seed, so calibration and validation are exchangeable. `medPathBenchmark`, `medCodERBenchmark`, and `BCD5` instead use the benchmark's own curated splits (for MedPath, `train` as calibration and `dev` as validation). Those are not exchangeable — on MedPath the two splits differ in baseline risk by 0.03–0.05 before any calibration runs — so validation numbers from them measure a shifted population rather than verifying the guarantee.

## Installation

```bash
uv sync          # or: pip install -e .
```

Requires Python 3.10.

## Quick start

```python
from rcps_el.dataset import medPathBenchmark
from rcps_el.scores import cumulativeRetrievalScorer
from rcps_el.losses import hitsAtK
from rcps_el.evaluators import rcpsELEvaluator

dataset = medPathBenchmark()
score = cumulativeRetrievalScorer(name="MedPath Score", softmax_temperature=0.05)
loss = hitsAtK(k_size=5)

evaluator = rcpsELEvaluator(
    dataset=dataset,
    score_function=score,
    loss_function=loss,
    target_risk=0.05,              ## alpha
    risk_formulation="relative",   ## or "absolute"
    delta=0.1,                     ## guarantee holds w.p. >= 1 - delta
    bound="wsr",                   ## or "hoeffding_bentkus"
    risk_unit="document",          ## unit treated as i.i.d. by the bound
)
evaluator.execute()

calibration, validation = evaluator.results_summary
if validation["q_star_certified"]:
    print(
        f"q* = {validation['q_star']:.4f}  "
        f"risk {validation['risk_original']:.3f} -> {validation['risk_controlled']:.3f}  "
        f"set size {validation['c_set_size_original']:.1f} -> "
        f"{validation['c_set_size_controlled']:.1f}"
    )
else:
    print("no threshold certified; the uncalibrated set was returned")
```

`results_summary` is a two-element list (calibration, validation). Alongside the risk and set-size fields it carries `q_star`, `q_star_certified`, `risk_bound_at_q_star`, the settings the guarantee depends on (`delta`, `bound`, `risk_unit`), and the `*_unit` risks, which are the quantities the $1 - \delta$ statement is actually about.

## Reproducing the experiments

```bash
python scripts/run_trials_certified.py      # sweep datasets x scores x losses x alpha -> trials_certified.tsv
python scripts/plot_risk_calibration.py     # calibration curves as small multiples -> figs/
```

`run_trials_certified.py` writes the evaluator's full summary, certification fields included, and resumes from an existing output file rather than re-running completed trials. `plot_risk_calibration.py` draws realized risk and mean set size against the target, with uncertified points marked hollow so an abstention is never mistaken for a result; configuration is a block of constants at the top.

`rcpsELSetEvaluator` runs the same sweep as a library object with caching, if you would rather drive it from Python than from a script.
