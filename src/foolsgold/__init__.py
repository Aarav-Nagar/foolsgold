"""foolsgold: the odometer for backtests.

Log every backtest you run; find out whether your best one beats luck.

    import foolsgold as fg

    fg.record(returns, study="momentum", params={"lookback": 20})
    print(fg.report("momentum"))
"""
from __future__ import annotations

from typing import Optional

__version__ = "0.1.0"

from .api import record, trial
from .ledger import Ledger
from .reporting import StudyReport, build as _build


def report(study: str = "default", ledger: Optional[Ledger] = None) -> StudyReport:
    """The verdict for one study. Print it, or read its fields."""
    return _build(study, ledger)


__all__ = ["record", "trial", "report", "Ledger", "StudyReport", "__version__"]
