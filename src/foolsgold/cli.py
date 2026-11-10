"""Command line interface: foolsgold report | log | verify | badge | demo."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from typing import List, Optional

from . import __version__, demo as demo_mod, reporting as report_mod
from .api import record
from .ledger import Ledger


def _ledger(args) -> Ledger:
    return Ledger(args.dir) if args.dir else Ledger()


def _reports(args) -> List[report_mod.StudyReport]:
    ledger = _ledger(args)
    if args.study:
        return [report_mod.build(args.study, ledger)]
    return report_mod.build_all(ledger)


def cmd_report(args) -> int:
    reports = _reports(args)
    if args.json:
        print(json.dumps([r.to_dict() for r in reports], indent=2))
    elif not reports:
        print("No trials recorded in %s yet.\nLog one with foolsgold.record(returns), "
              "or run `foolsgold demo`." % _ledger(args).dir)
    else:
        render = report_mod.render_markdown if args.md else report_mod.render_text
        print("\n".join(render(r) for r in reports), end="")
    # --strict lets CI fail a build whose headline result has not earned it.
    if args.strict and any(r.verdict != report_mod.PASS for r in reports):
        return 1
    return 0


def cmd_log(args) -> int:
    entries = _ledger(args).entries(args.study)
    for e in entries[-args.n:]:
        params = " ".join("%s=%s" % kv for kv in (e.get("params") or {}).items())
        print("#%-5d %s  %-16s sharpe %6.2f  %s%s" % (
            e["seq"], e["ts"], e["study"][:16], e["sharpe"] * e["ppy"] ** 0.5,
            e.get("name") or params, "  (duplicate)" if e.get("dup") else ""))
    print("%d trial(s) total" % len(entries))
    return 0


def cmd_verify(args) -> int:
    ledger = _ledger(args)
    ok, problems = ledger.verify()
    n = len(ledger.entries())
    if ok:
        print("OK: %d entries, chain intact. Head: %s" % (n, ledger.head()))
        return 0
    print("TAMPERED: %d problem(s) in %d entries" % (len(problems), n))
    for p in problems:
        print("  - " + p)
    return 1


def cmd_badge(args) -> int:
    reports = _reports(args)
    if not reports:
        print("No trials recorded.", file=sys.stderr)
        return 1
    svg = report_mod.render_badge(reports[0])
    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(svg)
        print("wrote %s" % args.output)
    else:
        print(svg, end="")
    return 0


def cmd_demo(args) -> int:
    days = 252 * args.years + max(demo_mod.SLOWS)
    market = demo_mod.noise_market(days, args.seed)
    combos = list(demo_mod.grid())
    with tempfile.TemporaryDirectory() as tmp:
        ledger = Ledger(tmp)
        print("Searching %d moving-average strategies on %d years of pure noise." % (len(combos), args.years))
        print("By construction none of them has any edge.\n")
        for fast, slow, direction in combos:
            record(demo_mod.sma_crossover(market, fast, slow, direction), study="noise-search",
                   params={"fast": fast, "slow": slow, "side": "trend" if direction > 0 else "fade"},
                   ledger=ledger)
        print(report_mod.render_text(report_mod.build("noise-search", ledger)))

        print("Same search, plus one strategy whose true Sharpe really is %.1f:\n" % args.edge)
        for fast, slow, direction in combos:
            record(demo_mod.sma_crossover(market, fast, slow, direction), study="search-with-edge",
                   params={"fast": fast, "slow": slow, "side": "trend" if direction > 0 else "fade"},
                   ledger=ledger)
        record(demo_mod.real_edge(252 * args.years, args.seed + 1, args.edge), study="search-with-edge",
               name="the real one", ledger=ledger)
        print(report_mod.render_text(report_mod.build("search-with-edge", ledger)), end="")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dir", help="ledger directory (default: nearest .foolsgold/)")
    common.add_argument("--study", help="limit to one study")

    p = argparse.ArgumentParser(prog="foolsgold", description="The odometer for backtests.")
    p.add_argument("--version", action="version", version="foolsgold " + __version__)
    sub = p.add_subparsers(dest="command")

    s = sub.add_parser("report", parents=[common], help="is the best result better than luck?")
    s.add_argument("--json", action="store_true", help="machine-readable output")
    s.add_argument("--md", action="store_true", help="markdown receipt to paste next to a result")
    s.add_argument("--strict", action="store_true", help="exit 1 unless every study passes")
    s.set_defaults(func=cmd_report)

    s = sub.add_parser("log", parents=[common], help="list recorded trials")
    s.add_argument("-n", type=int, default=20, help="how many recent trials to show")
    s.set_defaults(func=cmd_log)

    s = sub.add_parser("verify", parents=[common], help="check that no trial was edited or removed")
    s.set_defaults(func=cmd_verify)

    s = sub.add_parser("badge", parents=[common], help="SVG badge with trial count and DSR")
    s.add_argument("-o", "--output", help="write to this file instead of stdout")
    s.set_defaults(func=cmd_badge)

    s = sub.add_parser("demo", help="watch a search on pure noise find a 'winner'")
    s.add_argument("--years", type=int, default=10)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--edge", type=float, default=2.0, help="true Sharpe of the planted strategy")
    s.set_defaults(func=cmd_demo)

    args = p.parse_args(argv)
    if not getattr(args, "func", None):
        p.print_help()
        return 0
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
