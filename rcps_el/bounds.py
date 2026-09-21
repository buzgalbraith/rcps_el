"""
Finite sample (1 - delta) upper confidence bounds on the mean of a [0, 1] valued sample.

These are the ingredients RCPS needs that a plain empirical mean does not provide.
Selecting a threshold by comparing the *empirical* risk to a target stops exactly
where sampling noise happens to be favourable, so the realised risk exceeds the
target roughly half the time. Selecting against an upper confidence bound instead
is what buys the "with probability at least 1 - delta" statement.

All bounds here take a sample ``x`` of values in [0, 1] and return ``r_plus`` with
``P(E[X] > r_plus) <= delta``.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binom

__all__ = [
    "BOUND_METHODS",
    "mean_ucb",
    "hoeffding_bentkus_ucb",
    "wsr_ucb",
    "empirical_mean_ucb",
]

## bisection settings shared by the bounds that invert a p-value ##
_TOL = 1e-10
_MAX_ITER = 200


def _check_sample(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("Cannot bound the mean of an empty sample.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Sample contains non-finite values.")
    ## a little slack for float error coming out of polars means ##
    if x.min() < -1e-9 or x.max() > 1 + 1e-9:
        raise ValueError(
            f"Sample must lie in [0, 1]; got range [{x.min()}, {x.max()}]. "
            "Rescale before calling."
        )
    return np.clip(x, 0.0, 1.0)


def _check_delta(delta: float) -> float:
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1); got {delta}.")
    return float(delta)


def _bisect_upper(certifies: callable, lo: float, hi: float) -> float:
    """
    Smallest m in [lo, hi] with ``certifies(m)`` true, assuming ``certifies`` is
    monotone (false below the crossing, true above it).
    """
    if certifies(lo):
        return lo
    if not certifies(hi):
        return hi
    for _ in range(_MAX_ITER):
        mid = 0.5 * (lo + hi)
        if certifies(mid):
            hi = mid
        else:
            lo = mid
        if hi - lo < _TOL:
            break
    return hi


def empirical_mean_ucb(x, delta: float) -> float:
    """
    The empirical mean, ignoring ``delta``. Provides NO coverage guarantee.

    Only here so the old (unbounded) behaviour stays reachable for comparison; do
    not use it for a result that claims risk control.
    """
    _check_delta(delta)
    return float(_check_sample(x).mean())


def hoeffding_bentkus_ucb(x, delta: float) -> float:
    """
    Hoeffding-Bentkus upper confidence bound, as used in Bates et al. (2021),
    "Distribution-Free, Risk-Controlling Prediction Sets", Section 3.

    Inverts the tighter of a Chernoff/KL tail bound and the Bentkus binomial
    bound. Tight when the mean is near 0 or 1, and close to plain Hoeffding in
    the middle of the range.
    """
    x = _check_sample(x)
    delta = _check_delta(delta)
    n = x.size
    r_hat = float(x.mean())
    if r_hat >= 1.0:
        return 1.0

    def h1(a: float, b: float) -> float:
        """KL between Bernoulli(a) and Bernoulli(b), for a <= b."""
        if b >= 1.0:
            return np.inf
        if b <= 0.0:
            return np.inf if a > 0 else 0.0
        out = (1.0 - a) * np.log((1.0 - a) / (1.0 - b))
        if a > 0.0:
            out += a * np.log(a / b)
        return float(out)

    def p_value(r: float) -> float:
        """Bound on P(empirical mean <= r_hat) when the true mean is r >= r_hat."""
        hoeffding = np.exp(-n * h1(r_hat, r))
        bentkus = np.e * binom.cdf(np.ceil(n * r_hat), n, r)
        return float(min(hoeffding, bentkus))

    ## p_value decreases as r moves up away from r_hat, so the UCB is the last r
    ## that is still plausible: sup{r : p_value(r) >= delta}.
    return _bisect_upper(lambda r: p_value(r) < delta, r_hat, 1.0)


def _wsr_lcb(y: np.ndarray, delta: float, c: float = 0.5) -> float:
    """
    Waudby-Smith-Ramdas betting *lower* confidence bound on E[Y] for Y in [0, 1].

    Builds the capital process K_i(m) = prod_j (1 + lam_j (Y_j - m)), a test
    supermartingale for H0: E[Y] <= m when lam_j >= 0 is predictable. The process
    growing past 1/delta is evidence that E[Y] > m, so the bound is the smallest m
    the data cannot argue against.
    """
    n = y.size
    idx = np.arange(1, n + 1)

    ## running mean/variance estimates; index i uses y_1..y_i ##
    mu_hat = (0.5 + np.cumsum(y)) / (1.0 + idx)
    sigma2_hat = (0.25 + np.cumsum((y - mu_hat) ** 2)) / (1.0 + idx)

    ## shift by one so the bet placed on y_i only sees y_1..y_{i-1} ##
    sigma2_pred = np.concatenate(([0.25], sigma2_hat[:-1]))
    lam = np.minimum(np.sqrt(2.0 * np.log(1.0 / delta) / (n * sigma2_pred)), c)

    log_threshold = np.log(1.0 / delta)

    def max_log_capital(m: float) -> float:
        ## lam <= c = 0.5 and |y - m| <= 1 keep every factor >= 0.5, so log1p is safe
        return float(np.max(np.cumsum(np.log1p(lam * (y - m)))))

    ## max_log_capital is decreasing in m, so the accepted set is an upper interval
    return _bisect_upper(lambda m: max_log_capital(m) <= log_threshold, 0.0, 1.0)


def wsr_ucb(x, delta: float, seed: int = 0) -> float:
    """
    Waudby-Smith-Ramdas betting upper confidence bound.

    Variance adaptive, so it is much tighter than Hoeffding-Bentkus for the low
    variance differences that show up in the proportional-risk constraint. This is
    the recommended default.

    Obtained as ``1 - lcb(1 - x)``: the capital process with non-negative bets
    tests a lower bound, and reflecting the sample turns it into an upper bound.

    The sample is permuted first. Unlike Hoeffding-Bentkus, this bound is
    *sequential*: the bets are placed one observation at a time from a running
    variance estimate, so its guarantee assumes the sample arrives in an order
    that carries no information about the values. Callers do not generally supply
    one -- the evaluator sorts by document id, and document ids often encode the
    source corpus, which correlates with difficulty. On a sample ordered by its
    own values the bound silently drops below the sample mean and covers nothing.
    Permuting with a fixed seed removes the dependence while keeping results
    reproducible. Do not permute more than once and take the best bound: the
    minimum over several permutations is not a valid 1 - delta bound.
    """
    x = _check_sample(x)
    delta = _check_delta(delta)
    x = np.random.default_rng(seed).permutation(x)
    return float(1.0 - _wsr_lcb(1.0 - x, delta))


BOUND_METHODS: dict[str, callable] = {
    "wsr": wsr_ucb,
    "hoeffding_bentkus": hoeffding_bentkus_ucb,
    "empirical": empirical_mean_ucb,
}


def mean_ucb(x, delta: float, method: str = "wsr") -> float:
    """(1 - delta) upper confidence bound on E[X] for a sample of X in [0, 1]"""
    if method not in BOUND_METHODS:
        raise ValueError(
            f"Unknown bound method {method!r}; choose from {sorted(BOUND_METHODS)}."
        )
    return BOUND_METHODS[method](x, delta)
