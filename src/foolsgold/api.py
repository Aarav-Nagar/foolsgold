"""The two ways to log a trial: call record(), or decorate your backtest."""
from __future__ import annotations

import functools
import hashlib
import inspect
import math
import os
import sys
from typing import Any, Callable, Dict, List, Optional

from .ledger import Ledger
from .stats import moments

_SCALARS = (str, int, float, bool, type(None))


def _to_floats(x: Any, what: str) -> List[float]:
    """Accept lists, numpy arrays and pandas Series without importing either."""
    if hasattr(x, "to_numpy"):
        x = x.to_numpy()
    if hasattr(x, "tolist"):
        x = x.tolist()
    try:
        vals = [float(v) for v in x]
    except (TypeError, ValueError):
        raise TypeError("%s must be a 1-D sequence of numbers (list, numpy array or pandas Series)" % what)
    return [v for v in vals if math.isfinite(v)]   # pct_change() leaves a leading NaN


def _equity_to_returns(equity: List[float]) -> List[float]:
    if any(v <= 0 for v in equity):
        raise ValueError("equity curve must be strictly positive to convert to returns")
    return [b / a - 1.0 for a, b in zip(equity, equity[1:])]


def _clean_value(v: Any) -> Any:
    if hasattr(v, "item") and not isinstance(v, _SCALARS):   # numpy scalar
        try:
            v = v.item()
        except (TypeError, ValueError):
            pass
    if isinstance(v, float) and not math.isfinite(v):
        return repr(v)
    if isinstance(v, _SCALARS):
        return v
    if isinstance(v, (list, tuple)) and len(v) <= 16 and all(isinstance(i, _SCALARS) for i in v):
        return [_clean_value(i) for i in v]
    return "<%s>" % type(v).__name__      # e.g. a DataFrame passed as an argument


def _clean_params(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return {str(k): _clean_value(v) for k, v in (params or {}).items()}


def _code_fingerprint() -> Optional[str]:
    """Name and short hash of the running script, so a report can say how
    many versions of the code produced these trials. Only the basename is kept."""
    path = sys.argv[0] if sys.argv else ""
    try:
        with open(path, "rb") as fh:
            return "%s@%s" % (os.path.basename(path), hashlib.sha256(fh.read()).hexdigest()[:10])
    except OSError:
        return None


def record(returns: Any = None, *, equity: Any = None, study: str = "default",
           params: Optional[Dict[str, Any]] = None, name: Optional[str] = None,
           periods_per_year: float = 252, ledger: Optional[Ledger] = None) -> Dict[str, Any]:
    """Log one backtest. Pass per-period `returns`, or an `equity` curve.

    `study` groups trials that are attempts at the same thing; the search
    penalty is computed per study. Returns the ledger entry.
    """
    if (returns is None) == (equity is None):
        raise TypeError("pass exactly one of returns= or equity=")
    series = _to_floats(returns, "returns") if returns is not None else _equity_to_returns(_to_floats(equity, "equity"))
    m = moments(series)
    body = {
        "study": str(study),
        "name": name,
        "params": _clean_params(params),
        "n": m.n,
        "mean": m.mean,
        "sharpe": m.sharpe,
        "skew": m.skew,
        "kurt": m.kurt,
        "flat": m.flat,
        "ppy": float(periods_per_year),
        "code": _code_fingerprint(),
    }
    return (ledger or Ledger()).append(body, series)


def trial(func: Optional[Callable] = None, *, study: Optional[str] = None, kind: str = "returns",
          extract: Optional[Callable[[Any], Any]] = None, periods_per_year: float = 252,
          ledger: Optional[Ledger] = None):
    """Decorator: every call to the backtest function is logged as a trial.

    The call's arguments become the trial's params. If the function returns
    something other than the series itself, pass `extract=` to pull it out,
    and `kind="equity"` if that series is an equity curve.
    """
    if kind not in ("returns", "equity"):
        raise ValueError("kind must be 'returns' or 'equity'")

    def decorate(f: Callable) -> Callable:
        sig = inspect.signature(f)

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            out = f(*args, **kwargs)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            params: Dict[str, Any] = {}
            for k, v in bound.arguments.items():
                if sig.parameters[k].kind is inspect.Parameter.VAR_KEYWORD:
                    params.update(v)
                else:
                    params[k] = v
            series = extract(out) if extract else out
            record(**{kind: series}, study=study or f.__name__, params=params,
                   periods_per_year=periods_per_year, ledger=ledger)
            return out

        return wrapper

    return decorate(func) if callable(func) else decorate
