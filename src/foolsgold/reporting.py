"""Turn a ledger into a verdict."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from . import stats
from .ledger import Ledger

PASS, UNPROVEN, FOOLS_GOLD, EMPTY = "PASS", "UNPROVEN", "FOOL'S GOLD", "NO TRIALS"
CONFIDENCE = 0.95


@dataclass
class StudyReport:
    study: str
    verdict: str
    n_trials: int                 # distinct backtests recorded
    n_duplicates: int             # byte-identical re-runs, not counted
    n_effective: float            # independent trials after correlation
    rms_correlation: Optional[float]   # root-mean-square pairwise correlation of trials
    code_versions: int
    best_seq: Optional[int]
    best_name: Optional[str]
    best_params: Dict[str, Any]
    observations: int
    years: float
    sharpe: float                 # annualized, as you would report it
    luck_threshold: float         # annualized expected best of n_effective zero-edge trials
    psr: float                    # P(true Sharpe > 0) ignoring the search
    dsr: float                    # same, after accounting for the search
    haircut_sharpe: float         # annualized, Bonferroni
    years_needed: float           # track record needed at this Sharpe; inf if below threshold
    ledger_head: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if math.isinf(d["years_needed"]):
            d["years_needed"] = None
        return d

    def __str__(self) -> str:
        return render_text(self)


def _variance(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mean = math.fsum(xs) / len(xs)
    return math.fsum((x - mean) ** 2 for x in xs) / (len(xs) - 1)


def build(study: str, ledger: Optional[Ledger] = None) -> StudyReport:
    ledger = ledger or Ledger()
    entries = ledger.entries(study)
    trials = [e for e in entries if not e.get("dup")]
    head = ledger.head()
    if not trials:
        return StudyReport(study, EMPTY, 0, len(entries), 0.0, None, 0, None, None, {}, 0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, math.inf, head)

    best = max(trials, key=lambda e: e["sharpe"] * math.sqrt(e["ppy"]))
    ppy = best["ppy"]
    root = math.sqrt(ppy)
    n_trials = len(trials)

    series = [r for r in (ledger.load_returns(e) for e in trials) if r is not None]
    r2 = stats.mean_squared_correlation(series)
    n_eff = stats.effective_trials(n_trials, r2)

    # Variance of Sharpe across trials, measured annualized so mixed bar sizes
    # compare sensibly, then expressed in the best trial's own periods.
    var = _variance([e["sharpe"] * math.sqrt(e["ppy"]) for e in trials]) / ppy
    sr, n, skew, kurt = best["sharpe"], best["n"], best["skew"], best["kurt"]
    luck = stats.expected_max_sharpe(n_eff, var)
    dsr = stats.probabilistic_sharpe(sr, n, skew, kurt, benchmark=luck)
    psr = stats.probabilistic_sharpe(sr, n, skew, kurt, benchmark=0.0)
    needed = stats.min_track_record(sr, skew, kurt, benchmark=luck, confidence=CONFIDENCE)

    if dsr >= CONFIDENCE:
        verdict = PASS
    elif sr <= luck:
        verdict = FOOLS_GOLD
    else:
        verdict = UNPROVEN

    return StudyReport(
        study=study, verdict=verdict, n_trials=n_trials, n_duplicates=len(entries) - n_trials,
        n_effective=n_eff, rms_correlation=None if r2 is None else math.sqrt(r2),
        code_versions=len({e.get("code") for e in trials if e.get("code")}),
        best_seq=best["seq"], best_name=best.get("name"), best_params=best.get("params") or {},
        observations=n, years=n / ppy, sharpe=sr * root, luck_threshold=luck * root,
        psr=psr, dsr=dsr, haircut_sharpe=stats.haircut_sharpe(sr, n, n_eff) * root,
        years_needed=needed / ppy, ledger_head=head,
    )


def build_all(ledger: Optional[Ledger] = None) -> List[StudyReport]:
    ledger = ledger or Ledger()
    return [build(s, ledger) for s in ledger.studies()]


_EXPLAIN = {
    PASS: "survives the search penalty at 95% confidence.",
    UNPROVEN: "beats the luck threshold, but not at 95% confidence.",
    FOOLS_GOLD: "no better than the best of that many zero-edge tries.",
    EMPTY: "nothing recorded yet.",
}


def _params(p: Dict[str, Any]) -> str:
    text = " ".join("%s=%s" % kv for kv in p.items())
    return text if len(text) <= 48 else text[:45] + "..."


def _pct(x: float) -> str:
    return "%.1f%%" % (100.0 * x)


def render_text(r: StudyReport) -> str:
    rule = "-" * 68
    if r.verdict == EMPTY:
        return 'foolsgold | study "%s"\n%s\n  No trials recorded.\n' % (r.study, rule)
    corr = "" if r.rms_correlation is None else ", typical correlation %.2f" % r.rms_correlation
    label = ("#%d %s" % (r.best_seq, r.best_name or _params(r.best_params))).rstrip()
    needed = ("never: it is under the luck threshold" if math.isinf(r.years_needed)
              else "%.1f yrs at this Sharpe to reach 95%%" % r.years_needed)
    rows = [
        ("Trials recorded", "%d" % r.n_trials,
         "%.1f independent%s" % (r.n_effective, corr)),
        ("Best trial", label, ""),
        ("Track record", "%.1f yrs" % r.years, "%d observations" % r.observations),
        None,
        ("Sharpe you would report", "%.2f" % r.sharpe, "annualized"),
        ("Luck threshold", "%.2f" % r.luck_threshold,
         "expected best of %.0f zero-edge tries" % r.n_effective),
        ("Naive confidence (PSR)", _pct(r.psr), "ignores the search"),
        ("Deflated Sharpe (DSR)", _pct(r.dsr), "counts the search"),
        ("Haircut Sharpe", "%.2f" % r.haircut_sharpe, "Bonferroni-adjusted"),
        ("Track record needed", "", needed),
    ]
    out = ['foolsgold | study "%s"' % r.study, rule]
    for row in rows:
        if row is None:
            out.append("")
        elif row[0] == "Best trial":
            out.append("  %-25s %s" % (row[0], row[1]))
        elif not row[1]:
            out.append("  %-25s %s" % (row[0], row[2]))
        else:
            out.append("  %-25s %-8s %s" % row)
    out += ["", "  Verdict: %s - %s" % (r.verdict, _EXPLAIN[r.verdict])]
    extra = []
    if r.n_duplicates:
        extra.append("%d identical re-runs not counted" % r.n_duplicates)
    if r.code_versions > 1:
        extra.append("%d versions of the code" % r.code_versions)
    if extra:
        out.append("  Also: " + "; ".join(extra) + ".")
    out.append("  Ledger head: %s  (check with: foolsgold verify)" % r.ledger_head[:12])
    return "\n".join(out) + "\n"


def render_markdown(r: StudyReport) -> str:
    """A receipt to paste under any Sharpe ratio you publish."""
    if r.verdict == EMPTY:
        return "**foolsgold receipt** - study `%s`: no trials recorded.\n" % r.study
    return (
        "**foolsgold receipt** - study `{study}`\n\n"
        "| | |\n|---|---|\n"
        "| Trials run | {n} ({eff:.1f} independent) |\n"
        "| Reported Sharpe | {sr:.2f} |\n"
        "| Luck threshold | {luck:.2f} |\n"
        "| Deflated Sharpe Ratio | {dsr} |\n"
        "| Haircut Sharpe | {hc:.2f} |\n"
        "| Verdict | **{verdict}** |\n"
        "| Ledger head | `{head}` |\n"
    ).format(study=r.study, n=r.n_trials, eff=r.n_effective, sr=r.sharpe, luck=r.luck_threshold,
             dsr=_pct(r.dsr), hc=r.haircut_sharpe, verdict=r.verdict, head=r.ledger_head[:16])


_COLORS = {PASS: "#2ea44f", UNPROVEN: "#bf8700", FOOLS_GOLD: "#cf222e", EMPTY: "#6e7781"}


def render_badge(r: StudyReport) -> str:
    """A static SVG badge: trial count and Deflated Sharpe Ratio."""
    left = "foolsgold"
    right = "no trials" if r.verdict == EMPTY else "%d trials | DSR %s" % (r.n_trials, _pct(r.dsr))
    lw, rw = 10 + int(6.6 * len(left)), 10 + int(6.6 * len(right))
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="20" role="img" aria-label="{l}: {r}">'
        '<title>{l}: {r}</title>'
        '<rect width="{lw}" height="20" fill="#24292f"/>'
        '<rect x="{lw}" width="{rw}" height="20" fill="{c}"/>'
        '<g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" font-size="11">'
        '<text x="{lx}" y="14">{l}</text><text x="{rx}" y="14">{r}</text></g></svg>\n'
    ).format(w=lw + rw, lw=lw, rw=rw, l=left, r=right, c=_COLORS[r.verdict],
             lx=lw / 2.0, rx=lw + rw / 2.0)
