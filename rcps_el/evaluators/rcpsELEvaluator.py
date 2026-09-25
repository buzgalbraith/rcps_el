"""
Class for running RCPS across a given dataset, with a specific score and loss function
"""

from rcps_el.scores import Scorer
from rcps_el.losses import lossFunction, hitsAtK
from rcps_el.dataset import Dataset
from rcps_el.utils import safeMatch
from rcps_el.bounds import BOUND_METHODS, mean_ucb

from numpy import linspace
import numpy as np
import polars as pl
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

## uncalibrated candidate threshold retraining every candidate in the original prediction set ##
Q_0: float = float("-inf")


def _or_nan(value: float | None) -> float:
    """Helper keeps optional columns numeric in Polars schema"""
    return float("nan") if value is None else float(value)


class rcpsELEvaluator:
    results_summary: list[dict] = []

    def __init__(
        self,
        dataset: Dataset,
        score_function: Scorer,
        loss_function: lossFunction,
        target_risk: float = 0.2,
        risk_formulation: str = 'relative',
        min_candidates: int = 2,
        max_q: int | float | None = None,
        min_q: int | float | None = None,
        num_steps: int = 100,
        delta: float = 0.1,
        bound: str = "wsr",
        risk_unit: str = "document",
        derived_hits_at_k: list[int] = [1,3,5,10],
    ) -> None:
        """Make an evaluator
        Parameters:
            target_risk : optional,float
                Max allowable % change increase in risk in relative formualation.
                Max allowable overall risk in the absolute formulation. This is 
                the alpha of the RCPS guarantee.
            risk_formulation : optional, str
                either to control absolute or relative risk
            min_candidates : optional, int
                Minimum number of candidates to try to narrow candidate set for
            min_q, max_q : optional, float
                Endpoints of the threshold grid. Left as None they are derived from
                the observed range of the score function on the calibration set,
                since score functions are not all supported on [0, 1].
            delta : optional, float
                The threshold is chosen so the proportional risk constraint holds
                with probability at least 1 - delta over draws of the calibration
                set. Selection is against an upper confidence bound on the risk,
                not the empirical risk.
            bound : optional, str
                Which upper confidence bound to select against. "wsr" (default,
                tightest here), "hoeffding_bentkus" (the bound in the RCPS paper),
                or "empirical" that is just empirical risk increase which will
                not provide finite sample coverage guarantees.
            risk_unit : optional, str
                The unit the concentration bound treats as i.i.d. Splits are by
                document, and mentions inside one document are correlated, so
                "document" (default) aggregates to a mean loss per document before
                bounding. "mention" pools mentions, which matches the older
                reported numbers but overstates the effective sample size and so
                voids the guarantee under within-document correlation.
            derived_hits_at_k : optional, list[int]
                Also report Hits@k (hit rate, pooled over mentions) on the original
                and the controlled sets for each k, as hits_at_{k}_original /
                hits_at_{k}_controlled. These are derived metrics: q* is chosen for
                the loss function only. Because a controlled set is always a top-m
                prefix of the original ranking, the per-mention drop in Hits@k is at
                most the increase in exact-code miss-coverage, so the bound carries
                over when the loss is binary_misscoverage_loss, and not otherwise.
        """
        self.dataset = dataset
        self.score_function = score_function
        self.loss_function = loss_function
        self.target_risk = target_risk
        self.risk_formulation = risk_formulation.lower()
        assert self.risk_formulation in ['absolute', 'relative'], AssertionError("Risk formulation must either be absolute or relative")
        
        self.min_candidates = min_candidates
        if not 0.0 < delta < 1.0:
            raise ValueError(f"delta must be in (0, 1); got {delta}.")
        if bound not in BOUND_METHODS:
            raise ValueError(
                f"Unknown bound {bound!r}; choose from {sorted(BOUND_METHODS)}."
            )
        if risk_unit not in ("document", "mention"):
            raise ValueError(
                f"risk_unit must be 'document' or 'mention'; got {risk_unit!r}."
            )
        if not all(
            isinstance(k, int) and k > 0 for k in derived_hits_at_k
        ):
            raise ValueError(
                f"derived_hits_at_k must be positive integers; got {derived_hits_at_k!r}."
            )
        self.derived_hits_at_k: list[int] = sorted(set(derived_hits_at_k or []))
        self.delta = float(delta)
        self.bound = bound
        self.risk_unit = risk_unit
        if bound == "empirical":
            logger.warning(
                "bound='empirical' selects on the empirical risk and provides no "
                "1 - delta guarantee; results are not risk controlled."
            )
        ## get original scores losses and candidate set sizes on calibration and validation set ##
        self.result_calibration_original = self.get_original_results(
            self.dataset.calibration_set
        )
        self.result_validation_original = self.get_original_results(
            self.dataset.validation_set
        )

        ## find range of possible score threshold (q), with granularity based on original risk ##
        self.q_range = self.build_q_range(
            min_q=min_q, max_q=max_q, num_steps=num_steps
        )
        ## keep only rows that have above a fixed number of possible candidates ##
        self.calibration_risk_index = self.get_risk_index(
            self.result_calibration_original
        )
        self.validation_risk_index = self.get_risk_index(
            self.result_validation_original
        )
        ## get the uncalibrated risk for the calibration set R(q_{0}) ##
        self.original_empirical_risk = self.calc_empirical_risk(
            self.result_calibration_original, calibration=True
        )
        self.q_star: float | None = None
        self.result_calibration_fitted: pl.DataFrame | None = None
        self.result_validation_fitted: pl.DataFrame | None = None
        self.q_star_certified: bool | None = None
        self.q_star_risk_ucb: float | None = None

    def execute(
        self,
        verbose: bool = True,
    ):
        """Fit and evaluate the model"""
        logger.info("Fitting q* on calibration data...")
        self.get_q_star(verbose=verbose)
        logger.info("Evaluating q* on validation data...")
        self.evaluate_on_validation()
        logger.info("Summary:")
        self.get_results_summary()

    def observed_score_range(self, dataset: pl.DataFrame) -> tuple[float, float] | None:
        """Observed min and max of the score function over a scored dataset"""
        scores = dataset[self.score_function.name].explode().drop_nulls()
        if len(scores) == 0:
            return None
        return float(scores.min()), float(scores.max())

    def build_q_range(
        self,
        min_q: int | float | None,
        max_q: int | float | None,
        num_steps: int,
    ) -> list[float]:
        """
        Build the descending grid of candidate thresholds.

        Endpoints default to the score range observed on the calibration set rather
        than to [0, 1]: a grid that stops short of the largest score never filters
        anything, and a grid that stops above the smallest score can never back off
        to the unfiltered set. Q_0 is always appended as the loosest point so
        the fallback is exact even if validation scores undershoot calibration ones.
        """
        observed = self.observed_score_range(self.result_calibration_original)
        if observed is None:
            logger.warning(
                f"No non-null {self.score_function.name} scores on the calibration set; "
                "falling back to a [0, 1] threshold grid."
            )
            observed = (0.0, 1.0)
        observed_min, observed_max = observed
        grid_min = float(min_q) if min_q is not None else observed_min
        grid_max = float(max_q) if max_q is not None else observed_max
        if grid_max <= grid_min:
            raise ValueError(
                f"Empty threshold grid for {self.score_function.name}: "
                f"max_q={grid_max} must exceed min_q={grid_min} "
                f"(observed score range {observed_min} to {observed_max})."
            )
        logger.info(
            f"Threshold grid for {self.score_function.name}: {num_steps} steps over "
            f"[{grid_min}, {grid_max}] (observed range [{observed_min}, {observed_max}])."
        )
        finite_grid = sorted(
            (float(q) for q in linspace(start=grid_min, stop=grid_max, num=num_steps)),
            reverse=True,
        )
        ## return grid as well as our uncalibrated value threshold q_0 ## 
        return finite_grid + [Q_0]

    def get_risk_index(self, dataset: pl.DataFrame):
        """
        Which mentions the risk is averaged over, and so which ones the bound is
        computed on.

        Every mention counts. R(q) in both 4.1 and 4.2 is a population
        expectation with no conditioning, so restricting this to mentions with at
        least min_candidates candidates would certify a different quantity than
        the one the guarantee is stated over. min_candidates governs narrowing
        only, in _filter_candidates.
        """
        return dataset["index"].unique()

    def guarantee_summary(self, calibration: bool) -> dict:
        """
        The risk-control settings and the realized risk on the unit the bound was
        computed over.

        `risk_original`/`risk_controlled` stay pooled over mentions so existing
        downstream scripts keep working; the `*_unit` fields are the quantities the
        1 - delta statement is actually about.
        """
        original = self.result_calibration_original if calibration else self.result_validation_original
        fitted = self.result_calibration_fitted if calibration else self.result_validation_fitted
        base = self.risk_sample(original, calibration=calibration)
        controlled = self.risk_sample(fitted, calibration=calibration)
        realised = (
            float((controlled.mean() - base.mean()) / base.mean())
            if base.mean() > 0
            else float("nan")
        )
        return {
            "delta": self.delta,
            "bound": self.bound,
            "risk_unit": self.risk_unit,
            "q_star": self.q_star,
            "q_star_certified": self.q_star_certified,
            "risk_bound_at_q_star": _or_nan(self.q_star_risk_ucb),
            "n_units": len(base),
            "risk_original_unit": float(base.mean()),
            "risk_controlled_unit": float(controlled.mean()),
            "proportional_risk_increase_realised": realised,
        }

    def derived_hits_summary(self, calibration: bool) -> dict:
        """Hits@k on the original and controlled sets; empty unless derived_hits_at_k is set"""
        risk_index = (
            self.calibration_risk_index if calibration else self.validation_risk_index
        )
        sets = {
            "original": self.result_calibration_original if calibration else self.result_validation_original,
            "controlled": self.result_calibration_fitted if calibration else self.result_validation_fitted,
        }
        summary = {}
        for k in self.derived_hits_at_k:
            hits = hitsAtK(k)
            for label, dataset in sets.items():
                losses = hits.execute(dataset.filter(pl.col("index").is_in(risk_index)))
                summary[f"hits_at_{k}_{label}"] = 1.0 - float(losses[hits.name].mean())
        return summary

    def get_results_summary(self):
        assert isinstance(self.result_calibration_fitted, pl.DataFrame) and isinstance(
            self.result_validation_fitted, pl.DataFrame
        )
        self.results_summary: list[dict] = []  ## reset just in case
        ## get calibration results
        self.results_summary.append(
            {
                "dataset": self.dataset.name,
                "split": "calibration",
                "target_risk": self.target_risk,
                "min_candidates": self.min_candidates,
                "risk_formulation" : self.risk_formulation,
                "score_function": self.score_function.name,
                "loss_function": self.loss_function.name,
                "samples": len(self.calibration_risk_index),
                "risk_original": self.calc_empirical_risk(
                    self.result_calibration_original, calibration=True
                ),
                "risk_controlled": self.calc_empirical_risk(
                    self.result_calibration_fitted, calibration=True
                ),
                "c_set_size_original": self.get_average_candidates(
                    self.result_calibration_original, calibration=True
                ),
                "c_set_size_controlled": self.get_average_candidates(
                    self.result_calibration_fitted, calibration=True
                ),
                **self.guarantee_summary(calibration=True),
                **self.derived_hits_summary(calibration=True),
            }
        )
        ## get validation results
        self.results_summary.append(
            {
                "dataset": self.dataset.name,
                "split": "validation",
                "target_risk": self.target_risk,
                "min_candidates": self.min_candidates,
                "risk_formulation" : self.risk_formulation,
                "score_function": self.score_function.name,
                "loss_function": self.loss_function.name,
                "samples": len(self.validation_risk_index),
                "risk_original": self.calc_empirical_risk(
                    self.result_validation_original, calibration=False
                ),
                "risk_controlled": self.calc_empirical_risk(
                    self.result_validation_fitted, calibration=False
                ),
                "c_set_size_original": self.get_average_candidates(
                    self.result_validation_original, calibration=False
                ),
                "c_set_size_controlled": self.get_average_candidates(
                    self.result_validation_fitted, calibration=False
                ),
                **self.guarantee_summary(calibration=False),
                **self.derived_hits_summary(calibration=False),
            }
        )
        logger.info(f"Calibration samples {len(self.result_calibration_original)}")
        logger.info(
            f"Calibration risk: {self.calc_empirical_risk(self.result_calibration_original, calibration=True)}->{self.calc_empirical_risk(self.result_calibration_fitted,calibration=True)}"
        )
        logger.info(
            f"Calibration average candidate set size:  {self.get_average_candidates(self.result_calibration_original,calibration=True)}->{self.get_average_candidates(self.result_calibration_fitted,calibration=True)}"
        )
        logger.info("-" * 100)
        logger.info(f"Validation samples {len(self.result_validation_original)}")
        logger.info(
            f"Validation risk: {self.calc_empirical_risk(self.result_validation_original,calibration=False)}->{self.calc_empirical_risk(self.result_validation_fitted,calibration=False)}"
        )
        logger.info(
            f"Validation average candidate set size:  {self.get_average_candidates(self.result_validation_original,calibration=False)}->{self.get_average_candidates(self.result_validation_fitted,calibration=False)}"
        )

    def get_average_candidates(self, dataset: pl.DataFrame, calibration: bool):
        if calibration:
            return (
                dataset.filter(pl.col("index").is_in(self.calibration_risk_index))
                .select(pl.mean("n_candidates"))
                .item()
            )
        return (
            dataset.filter(pl.col("index").is_in(self.validation_risk_index))
            .select(pl.mean("n_candidates"))
            .item()
        )

    def get_original_results(self, dataset: pl.DataFrame):
        """Get metrics on the original dataframe and sort by score"""
        dataset = self.score_function.execute(dataset)
        dataset = self.sort_by_scores(dataset)
        dataset = self.loss_function.execute(dataset)
        return self.count_candidates(dataset)

    def count_candidates(self, dataset: pl.DataFrame):
        """
        Get the size of candidate set for each row in a dataset.
        """
        return dataset.with_columns(n_candidates=pl.col("match_curies").list.len())

    def calc_empirical_risk(self, dataset: pl.DataFrame, calibration: bool) -> float:
        if calibration:
            return (
                dataset.filter(pl.col("index").is_in(self.calibration_risk_index))
                .select(pl.mean(self.loss_function.name))
                .item()
            )
        return (
            dataset.filter(pl.col("index").is_in(self.validation_risk_index))
            .select(pl.mean(self.loss_function.name))
            .item()
        )

    def risk_sample(self, dataset: pl.DataFrame, calibration: bool) -> np.ndarray:
        """
        The per-unit loss values the concentration bound is applied to, in [0, 1].

        Restricted to the risk index and returned in a deterministic order so that
        samples taken at two different thresholds are aligned unit by unit, which
        the paired constraint below relies on.
        """
        risk_index = (
            self.calibration_risk_index if calibration else self.validation_risk_index
        )
        loss_col = self.loss_function.name
        rows = dataset.filter(pl.col("index").is_in(risk_index))
        if self.risk_unit == "mention":
            rows = rows.sort("index")
            return rows[loss_col].to_numpy().astype(float)
        ## one i.i.d. unit per document: the document's mean loss, still in [0, 1] ##
        doc_col = self.dataset.document_id_column
        rows = (
            rows.group_by(doc_col)
            .agg(pl.mean(loss_col).alias(loss_col))
            .sort(doc_col)
        )
        return rows[loss_col].to_numpy().astype(float)

    def sort_by_scores(self, dataset: pl.DataFrame) -> pl.DataFrame:
        """
        sort match names, curies and scores in descending order according to the score value.
        """
        target = self.score_function.name
        ## get sort index and make sure null safe ## 
        sort_idx = pl.col(target).list.eval(
            pl.element().arg_sort(descending=True, nulls_last=True)
        )

        return dataset.with_columns([
            pl.col(target).list.gather(sort_idx).alias(target),
            pl.col("match_scores").list.gather(sort_idx).alias("match_scores"),
            pl.col("match_names").list.gather(sort_idx).alias("match_names"),
            pl.col("match_curies").list.gather(sort_idx).alias("match_curies"),
        ])

    def fit_at(self, q: float, dataset: pl.DataFrame) -> pl.DataFrame:
        """Apply threshold q then recompute the loss and candidate counts"""
        out = self.filter_candidates(q=q, dataset=dataset)
        out = self.loss_function.execute(out)
        return self.count_candidates(out)

    def certify(
        self,
        fitted: pl.DataFrame,
        base_losses: np.ndarray,
    ) -> tuple[bool, float]:
        """
        Test the proportional risk constraint at one threshold against a
        (1 - delta) upper confidence bound.

        The constraint R(q) <= (1 + alpha) R(q0) has a *random* right hand side,
        since R(q0) is estimated on the same calibration set. Bounding the
        numerator alone would ignore that. Instead form the paired per-unit
        quantity

            Z_i = L_i(q) - (1 + alpha) L_i(q0)      in [-alpha, 1]

        so the constraint is exactly E[Z] <= 0, a single bounded mean with no
        random denominator left in it. The lower end is -alpha rather than
        -(1 + alpha) because the sets are nested: filtering only removes
        candidates, so L_i(q) >= L_i(q0) for every unit. Rescaling Z onto
        [0, 1] as

            U_i = (Z_i + alpha) / (1 + alpha)

        turns the constraint into E[U] <= alpha / (1 + alpha), which the
        bounds in rcps_el.bounds can certify directly.

        Returns (certified, ucb) where ucb is on the E[U] scale.
        """
        fitted_losses = self.risk_sample(fitted, calibration=True)
        if fitted_losses.shape != base_losses.shape:
            raise RuntimeError(
                "Fitted and baseline risk samples are misaligned "
                f"({fitted_losses.shape} vs {base_losses.shape})."
            )
        alpha = self.target_risk
        ## if in absolute formulation do normal RCPS ## 
        if self.risk_formulation == "absolute":
            u,u_max = fitted_losses, alpha
        ## otherwise control relative risk increase ## 
        else:
            z = fitted_losses - (1.0 + alpha) * base_losses
            ## the tighter range relies on nesting, so fail loudly if it is broken ##
            if np.any(fitted_losses < base_losses - 1e-12):
                raise RuntimeError(
                    "Filtered loss fell below the unfiltered loss for some unit; "
                    "the sets are not nested, so Z is not bounded below by -alpha."
                )
            u = (z + alpha) / (1.0 + alpha)
            u_max = alpha / (1.0 + alpha)
        ucb = mean_ucb(u, self.delta, method=self.bound)
        return bool(ucb <= u_max), float(ucb)

    def evaluate_on_validation(
        self, q: float | None = None
    ) -> tuple[float, pl.DataFrame]:
        q = q if q is not None else self.q_star
        assert isinstance(q, float)
        result_validation = self.filter_candidates(
            q=q, dataset=self.result_validation_original
        )
        result_validation = self.loss_function.execute(result_validation)
        result_validation = self.count_candidates(result_validation)
        empirical_risk = self.calc_empirical_risk(result_validation, calibration=False)
        self.result_validation_fitted = result_validation
        return empirical_risk, result_validation

    def get_q_star(self, verbose: bool = True) -> float:
        """core procedure of finding score threshold q*"""
        if self.q_star is not None:
            logger.info(f"q star loading from cache...")
            return self.q_star


        base_losses = self.risk_sample(
            self.result_calibration_original, calibration=True
        )
        finite_grid = [q for q in self.q_range if q > Q_0]
        logger.info(
            f"Selecting q* against a {1 - self.delta:.5g} upper confidence bound "
            f"({self.bound}) on the {self.risk_unit}-level risk; "
            f"delta={self.delta}; "
            f"n_{self.risk_unit}s={len(base_losses)}."
        )

        self.q_star = Q_0
        self.q_star_certified = False
        self.q_star_risk_ucb = None
        self.result_calibration_fitted = self.result_calibration_original


        scan = list(reversed(finite_grid))

        progress = tqdm(
            scan,
            desc=(
                f"Calibrating q* on {self.dataset.name} with "
                f"{self.score_function.name} / {self.loss_function.name}"
            ),
            total=len(scan),
        )
        for q in progress:
            fitted = self.fit_at(q=q, dataset=self.result_calibration_original)
        
            certified, ucb = self.certify(
                fitted=fitted,
                base_losses=base_losses,
            )
            if verbose:
                empirical_risk = self.calc_empirical_risk(fitted, calibration=True)
                logger.info(
                    f"q:{q}, empirical risk:{empirical_risk}, "
                    f"risk bound:{ucb}, certified:{certified}, "
                    f"average number of candidates: {fitted['n_candidates'].mean()}"
                )
            if certified:
                self.q_star = float(q)
                self.q_star_certified = True
                self.q_star_risk_ucb = ucb
                self.result_calibration_fitted = fitted
            else:
                break

        if not self.q_star_certified:
            logger.warning(
                f"No threshold could be certified at delta={self.delta} for "
                f"target proportional risk increase "
                f"{self.target_risk} with "
                f"{len(base_losses)} {self.risk_unit}s. Falling back to the "
                "unfiltered candidate set, which satisfies the constraint "
                "trivially. Raise the target, raise delta, or calibrate on more "
                "data to narrow the sets."
            )
        return self.q_star

    def filter_candidates(self, q: float, dataset: pl.DataFrame) -> pl.DataFrame:
        """
        filter candidates above score threshold
        """
        return dataset.with_columns(
            evaluated=pl.struct(
                ["match_names", "match_curies", self.score_function.name]
            ).map_elements(
                lambda x: self._filter_candidates(
                    x["match_names"],
                    x["match_curies"],
                    x[self.score_function.name],
                    q=q,
                ),
                return_dtype=pl.List(
                    pl.Struct(
                        {
                            "name": pl.String,
                            "curie": pl.String,
                            "score": pl.Float64,
                        }
                    )
                ),
            )
        ).with_columns(
            pl.col("evaluated")
            .list.eval(pl.element().struct.field("name"))
            .alias("match_names"),
            pl.col("evaluated")
            .list.eval(pl.element().struct.field("curie"))
            .alias("match_curies"),
            pl.col("evaluated")
            .list.eval(pl.element().struct.field("score"))
            .alias(self.score_function.name),
        )

    def _filter_candidates(
        self, names: list[str], curies: list[str], scores: list[float], q: float
    ) -> list[safeMatch]:
        """
        internal method for filtering candidates above score threshold
        """
        records = []
        ## short circuit so do not filter any candidate sets smaller than our min target ##
        if len(scores) < self.min_candidates:
            q = Q_0
        for i, score in enumerate(scores):
            if score < q:
                continue
            records.append({"name": names[i], "curie": curies[i], "score": scores[i]})
        ## don't reduce the size of the set bellow one candidate
        if len(records) < 1 and len(scores) > 0:
            records.append(
                {
                    "name": names[0],
                    "curie": curies[0],
                    "score": scores[0],
                }
            )
        return records
