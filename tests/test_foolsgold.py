import json
import math
import random

import pytest

import foolsgold as fg
from foolsgold import demo, stats
from foolsgold.cli import main
from foolsgold.ledger import Ledger
from foolsgold.reporting import FOOLS_GOLD, PASS, build, render_badge, render_markdown


def noise(n, seed, mu=0.0, vol=0.01):
    rng = random.Random(seed)
    return [rng.gauss(mu, vol) for _ in range(n)]


@pytest.fixture
def ledger(tmp_path):
    return Ledger(tmp_path / ".foolsgold")


# ---- statistics -----------------------------------------------------------

def test_moments_of_known_series():
    m = stats.moments([1.0, 2.0, 3.0, 4.0, 5.0])
    assert m.mean == 3.0
    assert m.sd == pytest.approx(math.sqrt(2.5))
    assert m.skew == pytest.approx(0.0)
    assert m.kurt == pytest.approx(1.7)


def test_constant_returns_are_flat_not_a_crash():
    m = stats.moments([0.001] * 50)
    assert m.flat and m.sharpe == 0.0


def test_published_deflated_sharpe_example():
    # Worked example in Bailey & Lopez de Prado (2014): annualized Sharpe 2.5
    # from 1250 daily observations, skew -3, kurtosis 10, best of 100 trials
    # whose annualized Sharpe ratios have variance 1/2.
    sr = 2.5 / math.sqrt(250)
    var = 0.5 / 250
    assert stats.expected_max_sharpe(100, var) == pytest.approx(0.1132, abs=2e-4)
    assert stats.deflated_sharpe(sr, 1250, -3.0, 10.0, 100, var) == pytest.approx(0.9004, abs=2e-3)


def test_one_trial_has_no_search_penalty():
    assert stats.expected_max_sharpe(1, 0.01) == 0.0
    assert stats.deflated_sharpe(0.1, 500, 0, 3, 1, 0.0) == stats.probabilistic_sharpe(0.1, 500)


def test_more_trials_raise_the_bar():
    bars = [stats.expected_max_sharpe(n, 0.01) for n in (2, 10, 100, 1000)]
    assert bars == sorted(bars) and bars[0] > 0


def test_min_track_record():
    assert stats.min_track_record(0.05, benchmark=0.05) == math.inf
    n = stats.min_track_record(0.1, benchmark=0.0)
    assert stats.probabilistic_sharpe(0.1, round(n)) == pytest.approx(0.95, abs=0.01)


def test_haircut_shrinks_with_trials_and_never_flips_sign():
    base = stats.haircut_sharpe(0.1, 1000, 1)
    cut = stats.haircut_sharpe(0.1, 1000, 50)
    assert base == 0.1 and 0 <= cut < base
    assert stats.haircut_sharpe(0.01, 100, 500) == 0.0
    assert stats.haircut_sharpe(-0.1, 1000, 50) == -cut


def test_effective_trials_limits():
    a = noise(500, 1)
    mirrored = [[x for x in a], [-x for x in a], [2 * x for x in a]]
    assert stats.effective_trials(3, stats.mean_squared_correlation(mirrored)) == pytest.approx(1.0, abs=0.02)
    independent = [noise(500, s) for s in range(20)]
    assert stats.effective_trials(20, stats.mean_squared_correlation(independent)) > 19
    assert stats.effective_trials(5, None) == 5.0


# ---- ledger ---------------------------------------------------------------

def test_chain_verifies_and_detects_edits(ledger):
    for s in range(5):
        fg.record(noise(100, s), study="s", params={"seed": s}, ledger=ledger)
    assert ledger.verify() == (True, [])

    lines = ledger.file.read_text(encoding="utf-8").splitlines()
    doctored = json.loads(lines[2])
    doctored["sharpe"] = 9.9
    ledger.file.write_text("\n".join(lines[:2] + [json.dumps(doctored)] + lines[3:]) + "\n", encoding="utf-8")
    ok, problems = ledger.verify()
    assert not ok and any("edited" in p for p in problems)


def test_deleting_a_losing_trial_is_detected(ledger):
    for s in range(5):
        fg.record(noise(100, s), study="s", ledger=ledger)
    lines = ledger.file.read_text(encoding="utf-8").splitlines()
    ledger.file.write_text("\n".join(lines[:1] + lines[2:]) + "\n", encoding="utf-8")
    ok, problems = ledger.verify()
    assert not ok and any("removed" in p for p in problems)


def test_identical_rerun_is_not_a_new_trial(ledger):
    r = noise(100, 1)
    first = fg.record(r, study="s", ledger=ledger)
    again = fg.record(r, study="s", ledger=ledger)
    other_study = fg.record(r, study="t", ledger=ledger)
    assert (first["dup"], again["dup"], other_study["dup"]) == (False, True, False)
    assert build("s", ledger).n_trials == 1 and build("s", ledger).n_duplicates == 1


def test_returns_round_trip(ledger):
    r = noise(64, 3)
    entry = fg.record(r, ledger=ledger)
    assert ledger.load_returns(entry) == r


# ---- recording ------------------------------------------------------------

def test_record_accepts_equity_curve_and_drops_nan(ledger):
    e = fg.record(equity=[100, 101, 99, 102, 103], ledger=ledger)
    assert e["n"] == 4
    e = fg.record([float("nan"), 0.01, -0.02, 0.03, 0.01], ledger=ledger)
    assert e["n"] == 4


def test_record_rejects_bad_input(ledger):
    with pytest.raises(TypeError):
        fg.record(ledger=ledger)
    with pytest.raises(TypeError):
        fg.record([[0.1, 0.2], [0.3, 0.4]], ledger=ledger)
    with pytest.raises(ValueError):
        fg.record([0.1, 0.2], ledger=ledger)


def test_flat_strategy_is_counted_without_crashing(ledger):
    e = fg.record([0.0] * 30, ledger=ledger)
    assert e["flat"] and e["sharpe"] == 0.0
    assert build("default", ledger).n_trials == 1


def test_trial_decorator_logs_arguments(ledger):
    @fg.trial(study="sweep", ledger=ledger)
    def backtest(data, lookback, hold=5, **extra):
        return noise(100, lookback)

    backtest(object(), 20, tag="a")
    (entry,) = ledger.entries("sweep")
    assert entry["params"] == {"data": "<object>", "lookback": 20, "hold": 5, "tag": "a"}


def test_trial_decorator_extracts_equity(ledger):
    @fg.trial(kind="equity", extract=lambda out: out["equity"], ledger=ledger)
    def backtest(n):
        return {"equity": [100.0 + i + (i % 3) for i in range(n)]}

    assert backtest(50)["equity"][0] == 100.0
    assert ledger.entries("backtest")[0]["n"] == 49


# ---- verdicts -------------------------------------------------------------

def _search(ledger, study, days=1260, seed=0):
    market = demo.noise_market(days + max(demo.SLOWS), seed)
    for fast, slow, direction in demo.grid():
        fg.record(demo.sma_crossover(market, fast, slow, direction), study=study,
                  params={"fast": fast, "slow": slow, "dir": direction}, ledger=ledger)


def test_noise_search_does_not_pass_but_real_edge_does(ledger):
    _search(ledger, "noise")
    r = build("noise", ledger)
    assert r.n_trials == 192 and 1 < r.n_effective < 192
    assert r.verdict != PASS and r.psr > r.dsr

    _search(ledger, "edge")
    fg.record(demo.real_edge(1260, 1, annual_sharpe=3.0), study="edge", name="real", ledger=ledger)
    r = build("edge", ledger)
    assert r.verdict == PASS and r.best_name == "real"


def test_single_lucky_looking_trial_keeps_its_naive_confidence(ledger):
    fg.record(noise(1000, 5, mu=0.002), study="one", ledger=ledger)
    r = build("one", ledger)
    assert r.luck_threshold == 0.0 and r.dsr == r.psr


def test_renderers(ledger):
    _search(ledger, "noise", days=500)
    r = build("noise", ledger)
    assert "Luck threshold" in str(r)
    assert "| Trials run | 192" in render_markdown(r)
    assert render_badge(r).startswith("<svg") and "192 trials" in render_badge(r)
    json.dumps(r.to_dict())
    assert "No trials" in str(build("missing", ledger))


# ---- command line ---------------------------------------------------------

def test_cli(ledger, capsys):
    d = str(ledger.dir)
    assert main(["report", "--dir", d]) == 0
    assert "No trials recorded" in capsys.readouterr().out

    _search(ledger, "noise", days=500)
    assert main(["report", "--dir", d, "--json"]) == 0
    (parsed,) = json.loads(capsys.readouterr().out)
    assert parsed["n_trials"] == 192

    assert main(["report", "--dir", d, "--strict"]) == (0 if parsed["verdict"] == PASS else 1)
    assert main(["verify", "--dir", d]) == 0
    assert main(["log", "--dir", d, "-n", "3"]) == 0
    assert "192 trial(s) total" in capsys.readouterr().out


def test_demo_runs(capsys):
    assert main(["demo", "--years", "2"]) == 0
    out = capsys.readouterr().out
    assert "pure noise" in out and "Verdict" in out
