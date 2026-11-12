import math

import numpy as np
import pandas as pd
import pytest

from quantbt import metrics as m


def series(values):
    return pd.Series(values, index=pd.bdate_range("2024-01-01", periods=len(values)), dtype=float)


def test_max_drawdown_and_duration():
    eq = series([100, 120, 90, 95, 130, 117])
    assert m.max_drawdown(eq) == pytest.approx(90 / 120 - 1)
    assert m.max_drawdown_duration(eq) == 2


def test_sharpe_matches_definition():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.01, 1000))
    expected = r.mean() / r.std(ddof=1) * math.sqrt(252)
    assert m.sharpe_ratio(r) == pytest.approx(expected)


def test_sharpe_of_flat_returns_is_zero():
    assert m.sharpe_ratio(pd.Series([0.0] * 10)) == 0.0


def test_cagr_doubling_in_one_year():
    eq = series(np.linspace(100, 200, 253))
    assert m.cagr(eq) == pytest.approx(1.0)


def test_sortino_ignores_upside():
    up_only = pd.Series([0.01, 0.02, 0.0, 0.03])
    assert m.sortino_ratio(up_only) == 0.0  # no downside deviation


def test_trade_stats():
    trades = pd.DataFrame({"pnl": [100.0, -50.0, 30.0, -20.0]})
    out = m.summarize(series([100, 101, 102]), trades=trades)
    assert out["win_rate"] == 0.5
    assert out["profit_factor"] == pytest.approx(130 / 70)
