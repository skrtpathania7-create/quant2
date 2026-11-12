import pytest

from conftest import Scripted, make_bars
from quantbt import Backtest, SimulatedBroker, realistic_broker, synthetic_ohlcv
from quantbt.strategies import MovingAverageCrossover, ZScoreMeanReversion


def test_no_lookahead_signal_fills_at_next_open(prices):
    result = Backtest(prices, Scripted({0: {"ASSET": 1.0}}), initial_cash=10_000).run()
    first = result.fills.iloc[0]
    assert first["timestamp"] == prices.index[1]
    assert first["price"] == prices["open"].iloc[1]
    assert first["quantity"] == int(10_000 / prices["close"].iloc[0])


def test_equity_matches_cash_plus_marked_positions(prices):
    result = Backtest(prices, Scripted({0: {"ASSET": 1.0}, 2: {"ASSET": 0.5}}), initial_cash=10_000).run()
    qty = result.positions["ASSET"]
    expected = result.cash + qty * prices["close"]
    assert result.equity.to_numpy() == pytest.approx(expected.to_numpy())


def test_unchanged_targets_do_not_trade(prices):
    script = {i: {"ASSET": 1.0} for i in range(len(prices))}
    result = Backtest(prices, Scripted(script)).run()
    assert len(result.fills) == 1


def test_costs_reduce_performance():
    data = synthetic_ohlcv(n=600, seed=3)
    free = Backtest(data, MovingAverageCrossover(10, 50), broker=SimulatedBroker(max_participation=None)).run()
    costly = Backtest(data, MovingAverageCrossover(10, 50), broker=realistic_broker()).run()
    assert costly.metrics["num_fills"] > 0
    assert costly.equity.iloc[-1] < free.equity.iloc[-1]
    assert costly.metrics["total_commission"] > 0 and costly.metrics["total_slippage"] > 0


def test_start_trading_skips_warmup(prices):
    result = Backtest(prices, Scripted({0: {"ASSET": 1.0}}), start_trading=prices.index[2]).run()
    assert result.equity.index[0] == prices.index[2]
    # the warm-up target is sent once trading starts, so it fills on the bar after
    assert result.fills.iloc[0]["timestamp"] == prices.index[3]


def test_multi_asset_with_missing_bars():
    a = synthetic_ohlcv(n=300, seed=1)
    b = synthetic_ohlcv(n=300, seed=2).iloc[50:]  # starts later
    result = Backtest({"A": a, "B": b}, ZScoreMeanReversion(lookback=20, entry_z=1.5)).run()
    assert set(result.fills["symbol"]) == {"A", "B"}
    assert result.fills[result.fills["symbol"] == "B"]["timestamp"].min() > b.index[0]


def test_bad_data_is_rejected():
    with pytest.raises(ValueError):
        Backtest(make_bars([(1, 1, 1, 1, 1)]).drop(columns="volume"), Scripted({}))
