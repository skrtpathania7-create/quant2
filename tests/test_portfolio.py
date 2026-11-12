import pandas as pd
import pytest

from quantbt import FillEvent, Portfolio

T = pd.Timestamp("2024-01-02")


def fill(qty, price, commission=0.0):
    return FillEvent(T, 1, "X", qty, price, commission, 0.0)


def test_cash_and_average_cost():
    p = Portfolio(10_000)
    p.on_fill(fill(10, 100, commission=1))
    p.on_fill(fill(10, 110))
    assert p.cash == pytest.approx(10_000 - 1_000 - 1 - 1_100)
    assert p.positions["X"] == 20
    assert p.avg_cost["X"] == pytest.approx(105)


def test_realized_pnl_on_partial_close():
    p = Portfolio(10_000)
    p.on_fill(fill(20, 100))
    p.on_fill(fill(-5, 110))
    assert p.trades[-1]["pnl"] == pytest.approx(50)
    assert p.avg_cost["X"] == 100  # unchanged by a reduction


def test_short_pnl_and_flip():
    p = Portfolio(10_000)
    p.on_fill(fill(-10, 100))
    p.on_fill(fill(15, 90))  # cover 10 for +100, then long 5 @ 90
    assert p.trades[-1]["pnl"] == pytest.approx(100)
    assert p.positions["X"] == 5
    assert p.avg_cost["X"] == 90


def test_equity_is_cash_plus_positions():
    p = Portfolio(10_000)
    p.on_fill(fill(10, 100, commission=2))
    p.last_price["X"] = 120
    assert p.equity == pytest.approx(10_000 - 1_000 - 2 + 1_200)
