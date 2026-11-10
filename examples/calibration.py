"""How often is foolsgold fooled?

Repeats a 192-strategy search on fresh pure-noise markets and counts how often
the winner (a) looks significant the naive way and (b) gets a PASS from
foolsgold. Nothing here has an edge, so every PASS is a false positive.

    python examples/calibration.py --runs 300
"""
import argparse
import tempfile

import foolsgold as fg
from foolsgold import demo
from foolsgold.reporting import PASS

parser = argparse.ArgumentParser()
parser.add_argument("--runs", type=int, default=300)
parser.add_argument("--years", type=int, default=5)
args = parser.parse_args()

naive = passed = 0
for seed in range(args.runs):
    market = demo.noise_market(252 * args.years + max(demo.SLOWS), seed)
    with tempfile.TemporaryDirectory() as tmp:
        ledger = fg.Ledger(tmp)
        for fast, slow, direction in demo.grid():
            fg.record(demo.sma_crossover(market, fast, slow, direction), study="noise",
                      params={"fast": fast, "slow": slow, "direction": direction}, ledger=ledger)
        r = fg.report("noise", ledger)
    naive += r.psr >= 0.95
    passed += r.verdict == PASS

print("%d searches of 192 strategies on %d years of noise" % (args.runs, args.years))
print("  winner looks significant, ignoring the search: %d (%.0f%%)" % (naive, 100.0 * naive / args.runs))
print("  winner passes foolsgold:                       %d (%.1f%%)" % (passed, 100.0 * passed / args.runs))
