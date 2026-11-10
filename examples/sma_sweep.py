"""The usual way a strategy gets "discovered": sweep parameters, keep the best.

Run it, then ask foolsgold what the winner is worth:

    python examples/sma_sweep.py
    foolsgold report

The market below is random noise, so there is nothing real to find.
"""
import foolsgold as fg
from foolsgold import demo

market = demo.noise_market(days=252 * 5 + 200, seed=42)


@fg.trial(study="sma-sweep")           # <- the only line foolsgold needs
def backtest(fast, slow, direction=1):
    return demo.sma_crossover(market, fast, slow, direction)


best = None
for fast, slow, direction in demo.grid():
    returns = backtest(fast, slow, direction)
    total = sum(returns)
    if best is None or total > best[0]:
        best = (total, fast, slow, direction)

print("Best parameters: fast=%d slow=%d direction=%+d" % best[1:])
print(fg.report("sma-sweep"))
