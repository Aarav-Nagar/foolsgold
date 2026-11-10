# foolsgold

**The odometer for backtests.** It counts every strategy you try, then tells you
whether your best result beats what luck alone would have produced.

[![CI](https://github.com/Aarav-Nagar/foolsgold/actions/workflows/ci.yml/badge.svg)](https://github.com/Aarav-Nagar/foolsgold/actions/workflows/ci.yml)
![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
![Dependencies: none](https://img.shields.io/badge/dependencies-none-brightgreen)
![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)

```
$ foolsgold demo
Searching 192 moving-average strategies on 10 years of pure noise.
By construction none of them has any edge.

foolsgold | study "noise-search"
--------------------------------------------------------------------
  Trials recorded           192      128.1 independent, typical correlation 0.58
  Best trial                #120 fast=13 side=fade slow=200
  Track record              10.0 yrs 2520 observations

  Sharpe you would report   0.68     annualized
  Luck threshold            0.91     expected best of 128 zero-edge tries
  Naive confidence (PSR)    98.4%    ignores the search
  Deflated Sharpe (DSR)     23.6%    counts the search
  Haircut Sharpe            0.00     Bonferroni-adjusted
  Track record needed       never: it is under the luck threshold

  Verdict: FOOL'S GOLD - no better than the best of that many zero-edge tries.
```

That market was random numbers. The winning strategy still looks 98.4%
certain to be real, until you count the 191 others that were tried first.

## The problem

Try enough strategies and one will look good by chance. The statistics to
correct for that have existed since 2014, and they all need one input: **how
many things did you try?** Nobody knows. The losing runs were in a notebook
cell that got overwritten, a parameter sweep nobody saved, or an AI agent's
overnight session that reported only the winner.

Other tools compute the correction if you type in the trial count. foolsgold
measures it. One line in your backtest logs every run to a tamper-evident
file, and `foolsgold report` does the rest.

## Install

```
pip install git+https://github.com/Aarav-Nagar/foolsgold
```

Pure Python, 3.9+, no dependencies. Accepts lists, NumPy arrays and pandas
Series without importing either.

## Use

Decorate the function that runs one backtest:

```python
import foolsgold as fg

@fg.trial(study="momentum")
def backtest(lookback, hold):
    ...
    return daily_returns        # list, array or Series

for lookback in range(5, 120, 5):
    for hold in (1, 5, 21):
        backtest(lookback, hold)

print(fg.report("momentum"))
```

Or log a run from anywhere:

```python
fg.record(returns, study="momentum", params={"lookback": 20, "hold": 5})
fg.record(equity=equity_curve, study="momentum", periods_per_year=52)
```

A **study** is a set of attempts at the same goal. The search penalty is
computed per study, so every variation you would have been happy to report
belongs in the same one.

Then, from the shell:

| Command | What it does |
|---|---|
| `foolsgold report` | The verdict for each study. `--json` for machines, `--md` for a receipt to paste beside your result. |
| `foolsgold report --strict` | Exits 1 unless every study passes. Use it in CI. |
| `foolsgold log` | Lists recorded trials. |
| `foolsgold verify` | Checks that no trial was edited or removed. |
| `foolsgold badge -o badge.svg` | A badge showing trial count and Deflated Sharpe. |
| `foolsgold demo` | The noise search above, plus one planted real strategy that passes. |

## Reading the report

| Line | Meaning |
|---|---|
| Trials recorded | Distinct backtests in this study, and how many independent bets they amount to once correlation between them is taken into account. |
| Luck threshold | The best Sharpe you should expect from that many strategies if none of them worked. |
| Naive confidence (PSR) | Probability the true Sharpe is above zero if this had been your only attempt. |
| Deflated Sharpe (DSR) | The same probability measured against the luck threshold. 95% or more is a `PASS`. |
| Haircut Sharpe | What the Sharpe is worth after a Bonferroni correction for the number of trials. |
| Track record needed | How much data this Sharpe would need to clear the bar. |

The formulas, their sources and the reasoning are in
[docs/how-it-works.md](docs/how-it-works.md).

## With AI coding agents

An agent asked to "find a profitable strategy" will run dozens of backtests
and report the best one. Put this in your `AGENTS.md` or `CLAUDE.md`:

```markdown
## Backtesting rules
- Log every backtest with `foolsgold.record(returns, study=..., params=...)`,
  including the ones that fail or look bad.
- Never report a Sharpe ratio alone. Run `foolsgold report --study <name>` and
  report its verdict, trial count and Deflated Sharpe Ratio.
- Never edit or delete anything in `.foolsgold/`. Run `foolsgold verify`
  before reporting results.
```

Because each ledger entry includes a hash of the one before it, a deleted or
edited trial is detectable afterwards.

## Does it work?

`examples/calibration.py` repeats the 192-strategy search on 300 different
pure-noise markets (5 years each), where every "discovery" is false:

| | Searches fooled |
|---|---|
| Judging the winner as if it were the only try (PSR >= 95%) | 232 of 300 (77%) |
| foolsgold verdict `PASS` | 0 of 300 |

Zero is stricter than the 5% a perfectly calibrated test would allow: the
independent-trial estimate over-counts on purpose, so a real edge needs a
little more evidence to pass. Run it yourself; it takes a few minutes.

## What it does not do

- **It only counts trials that reach it.** Ideas you tested before installing
  it, or discarded after reading someone else's backtest, are invisible.
  Treat the trial count as a lower bound.
- **It is tamper-evident, not tamper-proof.** Deleting the whole ledger starts
  a clean history. Commit `.foolsgold/ledger.jsonl` to git, or publish the
  ledger head hash with your result, to pin it.
- **It tests for luck, not for bugs.** A backtest that peeks at future data
  can produce a Sharpe so high it passes every test here. Check for lookahead
  bias separately.
- **It assumes returns are roughly independent over time.** Strongly
  autocorrelated returns (illiquid assets, smoothed marks) overstate the
  Sharpe ratio, and this version does not adjust for that.
- **The independent-trial estimate is conservative** and only compares trials
  with the same number of observations.
- Nothing here is investment advice.

## Roadmap

- Automatic logging for backtesting.py, vectorbt and backtrader, with no
  decorator needed
- An MCP server and agent hook so an AI agent's runs are logged whether or not
  it cooperates
- Probability of Backtest Overfitting (CSCV) and autocorrelation-adjusted
  Sharpe
- A notebook cell magic
- PyPI release

Issues and pull requests are welcome, especially adapters for backtest
libraries you use.

## License

MIT
