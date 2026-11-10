# How foolsgold works

Everything below is implemented in [`src/foolsgold/stats.py`](../src/foolsgold/stats.py)
using only the Python standard library. Sharpe ratios are per period unless
stated; the report annualizes them for display.

## 1. Why the best of many backtests is biased

Give a strategy with no edge a few years of data and its measured Sharpe ratio
will not be exactly zero. It will be a random number near zero. Try 200 such
strategies and report the highest: you are now reporting the maximum of 200
random numbers, which is reliably well above zero. Nothing was discovered. This
is selection bias, and the only way to correct for it is to know how many
things were tried.

## 2. The luck threshold

For `N` independent trials whose Sharpe ratios have variance `V`, the expected
maximum under the hypothesis that none has an edge is approximately

```
SR0 = sqrt(V) * ( (1 - g) * Z(1 - 1/N) + g * Z(1 - 1/(N*e)) )
```

where `Z` is the inverse standard normal CDF and `g ~ 0.5772` is the
Euler-Mascheroni constant (Bailey & Lopez de Prado, 2014). foolsgold calls
`SR0` the **luck threshold**. `V` is measured from the Sharpe ratios in your
ledger, which is why it needs every trial and not only the count.

## 3. Deflated Sharpe Ratio

The Probabilistic Sharpe Ratio is the probability that the true Sharpe exceeds
a benchmark `SR*`, given an observed Sharpe `SR` over `T` returns with skewness
`s` and kurtosis `k`:

```
PSR(SR*) = Phi( (SR - SR*) * sqrt(T - 1) / sqrt(1 - s*SR + (k - 1)/4 * SR^2) )
```

The denominator widens the uncertainty for fat-tailed, negatively skewed
returns, which is what many profitable-looking strategies have. The **Deflated
Sharpe Ratio** is `PSR(SR0)`: the same test, with the bar raised from zero to
the luck threshold. The report shows `PSR(0)` beside it as "naive confidence"
so the cost of the search is visible. Tests check the implementation against
the worked example in the 2014 paper.

## 4. Counting independent trials

Two hundred variations of one moving-average rule are not two hundred
independent bets. foolsgold estimates the effective number of trials from the
stored return series with Nyholt's (2004) formula from genetics, where the same
problem appears as correlated statistical tests:

```
N_eff = 1 + (N - 1) * (1 - mean squared pairwise correlation)
```

Squared correlation is used so that a strategy and its mirror image count as
one idea. With more than 2,000 pairs, a fixed random sample of pairs is used.
The estimator is known to over-count somewhat, which makes foolsgold stricter,
not more lenient.

## 5. The other two numbers

**Haircut Sharpe** (Harvey & Liu, 2015): convert the Sharpe to a p-value,
multiply by `N_eff` (Bonferroni), convert back. It answers "what Sharpe would
this be if it had been my only try?"

**Track record needed**: the number of observations at which the observed
Sharpe would clear the luck threshold with 95% confidence (the Minimum Track
Record Length of Bailey & Lopez de Prado, 2012).

## 6. The ledger

`.foolsgold/ledger.jsonl` holds one JSON object per trial: parameters, summary
statistics, a hash of the return series, and `hash = SHA-256(previous hash +
this entry)`. Editing an entry or deleting one from the middle breaks every
hash after it, and `foolsgold verify` reports where. Return series are stored
beside it in `.foolsgold/returns/` so correlations can be computed later.
Byte-identical re-runs within a study are recorded but not counted.

## References

- Bailey, D. H. & Lopez de Prado, M. (2012). The Sharpe Ratio Efficient Frontier. *Journal of Risk*, 15(2).
- Bailey, D. H. & Lopez de Prado, M. (2014). The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality. *Journal of Portfolio Management*, 40(5).
- Harvey, C. R. & Liu, Y. (2015). Backtesting. *Journal of Portfolio Management*, 42(1).
- Nyholt, D. R. (2004). A Simple Correction for Multiple Testing for Single-Nucleotide Polymorphisms in Linkage Disequilibrium with Each Other. *American Journal of Human Genetics*, 74(4).
