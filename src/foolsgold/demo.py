"""A strategy search on pure noise, to show what luck looks like."""
from __future__ import annotations

import random
from typing import Iterator, List, Tuple

FASTS = (2, 3, 5, 8, 10, 13, 16, 20, 25, 30)
SLOWS = (20, 30, 40, 50, 60, 80, 100, 120, 150, 200)


def noise_market(days: int, seed: int, daily_vol: float = 0.01) -> List[float]:
    """Daily returns with zero mean and no memory. Nothing here is predictable."""
    rng = random.Random(seed)
    return [rng.gauss(0.0, daily_vol) for _ in range(days)]


def _prices(returns: List[float]) -> List[float]:
    out, p = [], 100.0
    for r in returns:
        p *= 1.0 + r
        out.append(p)
    return out


def sma_crossover(market: List[float], fast: int, slow: int, direction: int = 1,
                  warmup: int = max(SLOWS)) -> List[float]:
    """Hold +direction when the fast average is above the slow one, else -direction.

    The position decided at the close of day t earns the return of day t+1,
    so the strategy never sees the future. Every variant starts after the same
    warmup so their return series line up.
    """
    prices = _prices(market)
    prefix = [0.0]
    for p in prices:
        prefix.append(prefix[-1] + p)

    def sma(t: int, w: int) -> float:
        return (prefix[t + 1] - prefix[t + 1 - w]) / w

    out = []
    for t in range(warmup - 1, len(market) - 1):
        side = 1 if sma(t, fast) > sma(t, slow) else -1
        out.append(direction * side * market[t + 1])
    return out


def grid() -> Iterator[Tuple[int, int, int]]:
    for fast in FASTS:
        for slow in SLOWS:
            if slow > fast:
                for direction in (1, -1):
                    yield fast, slow, direction


def real_edge(days: int, seed: int, annual_sharpe: float = 2.0, daily_vol: float = 0.01) -> List[float]:
    """A return stream whose true Sharpe is `annual_sharpe` by construction."""
    rng = random.Random(seed)
    mu = annual_sharpe / 252 ** 0.5 * daily_vol
    return [rng.gauss(mu, daily_vol) for _ in range(days)]
