# Risk Controlled Prediction Sets for Biomedical Entity Linking (RCPS-EL)
RCPS-EL is a modular and extensible framework for constructing conformal candidate sets for biomedical entity linking (EL). Building on Risk-Controlling Prediction Sets (RCPS), it converts the ranked candidate output of pre-trained EL models into calibrated sets that provide user-specified coverage guarantees at a controlled error rate.
## Overview
Existing biomedical EL models return ranked candidate lists without formal statistical guarantees that the correct grounding is included. RCPS-EL addresses this by calibrating a score threshold $\hat{q}$ on a held-out calibration set such that the relative increase in risk over the default model output is controlled at a user-specified level $\alpha$ with probability at least $1 - \delta$.

Formally, RCPS-EL solves:

$$q^* \in \arg\min_{q} \mathbb{E}[|T_q(X)|] \quad \text{subject to} \quad \frac{R(q) - R(q_0)}{R(q_0)} \leq \alpha$$

producing compact candidate sets while preserving coverage guarantees.

## Features
- **Model-agnostic:** Works as a post-processing wrapper over predictions from any pre-trained EL model. User-provided model predictions can be added by extending the `Dataset` base-class which provides a unified API for risk-control.  
- **Multiple score functions:** Currently implements lexical edit distance, Gilda internal score, KRISSBERT internal score, SapBERT cosine similarity, LLM-based scoring functions. Further conformal score functions can be implemented by extending the `Scorer` base class, which provides a unified API for conformal scoring.
- **Multiple loss functions:** Method currently implements binary misscoverage and hits@K loss functions. Further loss functions can be implemented by extending the `lossFunction` base class, which provides a unified API for loss calculation.

## Installation 
- The package can be installed locally with `pip install -e .` 
## Quick start
Here is a basic example of using the RCPS-EL framework on a example dataset.
```
from rcps_el.dataset import bioIDBenchmark
from rcps_el.scores import fuzzyStringScore
from rcps_el.losses import binaryMisscoverageLoss
from rcps_el.evaluators import rcpsELEvaluator

## define score and loss functions ##
score = fuzzyStringScore()
loss = binaryMisscoverageLoss()

## load an example dataset ## 
dataset = bioIDBenchmark(
    original_dataframe_path="rcps_el/example_dataset/gilda_dataset.tsv", method="gilda"
)

## define and run evaluation ##
evaluator = rcpsELEvaluator(
    dataset=dataset,
    score_function=score,
    loss_function=loss,
    target_proportional_risk_increase=0.05,
)
evaluator.execute()
## view results summary ##
evaluator.get_results_summary()

```