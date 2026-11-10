"""Selection-bias-aware Sharpe ratio statistics. Standard library only.

All Sharpe ratios in this module are *per period* (not annualized) unless a
name says otherwise. Annualize by multiplying by sqrt(periods_per_year).

References
----------
Bailey & Lopez de Prado (2012), "The Sharpe Ratio Efficient Frontier".
Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio: Correcting for
    Selection Bias, Backtest Overfitting and Non-Normality".
Harvey & Liu (2015), "Backtesting".
Nyholt (2004), "A Simple Correction for Multiple Testing for Single-Nucleotide
    Polymorphisms in Linkage Disequilibrium with Each Other".
"""
from __future__ import annotations

import math
import random
from operator import mul
from statistics import NormalDist
from typing import List, NamedTuple, Optional, Sequence

_NORMAL = NormalDist()
EULER_GAMMA = 0.5772156649015329


class Moments(NamedTuple):
    n: int
    mean: float
    sd: float      # sample standard deviation (ddof=1)
    skew: float    # third standardized moment
    kurt: float    # fourth standardized moment (a normal distribution is 3)

    @property
    def sharpe(self) -> float:
        return self.mean / self.sd if self.sd > 0 else 0.0

    @property
    def flat(self) -> bool:
        return self.sd == 0


def moments(returns: Sequence[float]) -> Moments:
    """Mean, volatility, skewness and kurtosis of a return series."""
    n = len(returns)
    if n < 3:
        raise ValueError("need at least 3 returns, got %d" % n)
    mean = math.fsum(returns) / n
    dev = [r - mean for r in returns]
    m2 = math.fsum(d * d for d in dev) / n
    # A strategy that never trades has constant returns. Treat float noise
    # around a constant as exactly flat instead of dividing by ~0.
    if m2 == 0 or m2 <= (abs(mean) * 1e-12) ** 2:
        return Moments(n, mean, 0.0, 0.0, 3.0)
    m3 = math.fsum(d ** 3 for d in dev) / n
    m4 = math.fsum(d ** 4 for d in dev) / n
    return Moments(n, mean, math.sqrt(m2 * n / (n - 1)), m3 / m2 ** 1.5, m4 / m2 ** 2)


def _sharpe_variance_term(sharpe: float, skew: float, kurt: float) -> float:
    """Numerator of Var[SR_hat] * (n - 1) for non-normal returns (Mertens)."""
    return max(1.0 - skew * sharpe + (kurt - 1.0) / 4.0 * sharpe ** 2, 1e-12)


def probabilistic_sharpe(sharpe: float, n: int, skew: float = 0.0, kurt: float = 3.0,
                         benchmark: float = 0.0) -> float:
    """P(true Sharpe > benchmark) given one observed Sharpe. Bailey & LdP (2012)."""
    if n < 2:
        raise ValueError("need at least 2 observations")
    z = (sharpe - benchmark) * math.sqrt(n - 1) / math.sqrt(_sharpe_variance_term(sharpe, skew, kurt))
    return _NORMAL.cdf(z)


def expected_max_sharpe(n_trials: float, sharpe_variance: float) -> float:
    """Expected best Sharpe among `n_trials` strategies that all have zero edge.

    This is the "luck threshold": the Sharpe you should expect to see from the
    winner of a search even if nothing you tried actually works.
    `sharpe_variance` is the variance of the Sharpe ratios across the trials.
    """
    if n_trials <= 1 or sharpe_variance <= 0:
        return 0.0
    a = _NORMAL.inv_cdf(1.0 - 1.0 / n_trials)
    b = _NORMAL.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return max(0.0, math.sqrt(sharpe_variance) * ((1.0 - EULER_GAMMA) * a + EULER_GAMMA * b))


def deflated_sharpe(sharpe: float, n: int, skew: float, kurt: float,
                    n_trials: float, sharpe_variance: float) -> float:
    """Deflated Sharpe Ratio: PSR measured against the luck threshold."""
    return probabilistic_sharpe(sharpe, n, skew, kurt,
                                benchmark=expected_max_sharpe(n_trials, sharpe_variance))


def min_track_record(sharpe: float, skew: float = 0.0, kurt: float = 3.0,
                     benchmark: float = 0.0, confidence: float = 0.95) -> float:
    """Observations needed before `sharpe` beats `benchmark` at `confidence`.

    Returns infinity when the observed Sharpe does not exceed the benchmark.
    """
    if sharpe <= benchmark:
        return math.inf
    z = _NORMAL.inv_cdf(confidence)
    return 1.0 + _sharpe_variance_term(sharpe, skew, kurt) * (z / (sharpe - benchmark)) ** 2


def haircut_sharpe(sharpe: float, n: int, n_trials: float) -> float:
    """Bonferroni-adjusted Sharpe (Harvey & Liu 2015).

    Converts the Sharpe to a p-value, multiplies it by the number of trials,
    and converts back. Uses the normal approximation to the t distribution,
    which is accurate for the sample sizes backtests normally have.
    """
    if n_trials <= 1 or sharpe == 0:
        return sharpe
    t = abs(sharpe) * math.sqrt(n)
    p = math.erfc(t / math.sqrt(2.0))          # two-sided p-value
    p_adj = p * n_trials
    if p_adj >= 1.0:
        return 0.0
    if p_adj < 1e-15:                          # too significant for the haircut to matter
        return sharpe
    t_adj = _NORMAL.inv_cdf(1.0 - p_adj / 2.0)
    return math.copysign(min(t_adj, t) / math.sqrt(n), sharpe)


def mean_squared_correlation(series: Sequence[Sequence[float]], max_pairs: int = 2000,
                             seed: int = 0) -> Optional[float]:
    """Average squared correlation between return series, sampling pairs when there are many.

    Squaring matters: a strategy and its mirror image (long vs. short) have
    correlation -1 and are the same bet, which a plain average would hide.
    Only series sharing the most common length are compared, and flat series
    are skipped. The 1/n that pure noise contributes is subtracted. Returns
    None when fewer than two series are comparable.
    """
    by_len = {}
    for s in series:
        by_len.setdefault(len(s), []).append(s)
    if not by_len:
        return None
    group = max(by_len.values(), key=len)
    standardized: List[List[float]] = []
    for s in group:
        n = len(s)
        if n < 3:
            continue
        mean = math.fsum(s) / n
        var = math.fsum((x - mean) ** 2 for x in s) / n
        if var <= (abs(mean) * 1e-12) ** 2:
            continue
        sd = math.sqrt(var)
        standardized.append([(x - mean) / sd for x in s])
    k = len(standardized)
    if k < 2:
        return None
    n = len(standardized[0])
    total_pairs = k * (k - 1) // 2
    if total_pairs <= max_pairs:
        pairs = ((i, j) for i in range(k) for j in range(i + 1, k))
        count = total_pairs
    else:
        rng = random.Random(seed)
        sampled = set()
        while len(sampled) < max_pairs:
            i, j = rng.sample(range(k), 2)
            sampled.add((min(i, j), max(i, j)))
        pairs = iter(sorted(sampled))
        count = max_pairs
    acc = math.fsum((sum(map(mul, standardized[i], standardized[j])) / n) ** 2 for i, j in pairs)
    return min(1.0, max(0.0, acc / count - 1.0 / n))


def effective_trials(n_trials: int, mean_sq_correlation: Optional[float]) -> float:
    """Number of *independent* trials implied by how correlated the trials are.

    Fifty near-identical variations of one idea are not fifty independent
    bets. This is Nyholt's (2004) effective number of tests,
    1 + (N - 1)(1 - mean squared correlation): perfectly correlated trials
    count once, uncorrelated trials all count. It errs toward over-counting,
    which is the safe direction for a tool whose job is skepticism.
    """
    if n_trials <= 1 or mean_sq_correlation is None:
        return float(n_trials)
    r2 = min(1.0, max(0.0, mean_sq_correlation))
    return 1.0 + (n_trials - 1) * (1.0 - r2)
