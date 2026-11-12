import pandas as pd
import pytest

from conftest import bar, market
from quantbt import (
    FixedBpsSlippage,
    OrderEvent,
    OrderType,
    PerShareCommission,
    SimulatedBroker,
    VolumeImpactSlippage,
)

T0 = pd.Timestamp("2024-01-01")


def order(qty, **kw):
    return OrderEvent(1, T0, "X", qty, **kw)


def test_order_never_fills_on_its_own_bar():
    broker = SimulatedBroker(max_participation=None)
    broker.submit(order(100))
    assert broker.on_market(market(bar(ts=T0))) == []
    fills = broker.on_market(market(bar(open=101, ts="2024-01-02")))
    assert len(fills) == 1 and fills[0].price == 101


@pytest.mark.parametrize("qty, sign", [(100, 1), (-100, -1)])
def test_slippage_always_hurts(qty, sign):
    broker = SimulatedBroker(slippage=FixedBpsSlippage(10), max_participation=None)
    broker.submit(order(qty))
    fill = broker.on_market(market(bar(open=100)))[0]
    assert fill.price == pytest.approx(100 * (1 + sign * 0.001))
    assert fill.slippage == pytest.approx(100 * 0.1)


def test_impact_grows_with_size():
    model = VolumeImpactSlippage(spread_bps=0, impact=1.0)
    b = bar(open=100, high=102, low=98, volume=1_000_000)
    small = model.price(1, 1_000, b, 100) - 100
    big = model.price(1, 100_000, b, 100) - 100
    assert 0 < small < big
    assert big / small == pytest.approx(10)  # square-root law: 100x size -> 10x cost


def test_volume_cap_creates_partial_fills():
    broker = SimulatedBroker(max_participation=0.1)
    broker.submit(order(250))
    first = broker.on_market(market(bar(volume=1_000, ts="2024-01-02")))
    second = broker.on_market(market(bar(volume=1_000, ts="2024-01-03")))
    third = broker.on_market(market(bar(volume=1_000, ts="2024-01-04")))
    assert [f.quantity for f in first + second + third] == [100, 100, 50]
    assert broker.open_orders == {}


def test_buy_limit_fills_only_when_price_trades_through():
    broker = SimulatedBroker(max_participation=None)
    broker.submit(order(10, order_type=OrderType.LIMIT, limit_price=98.0))
    assert broker.on_market(market(bar(low=98.5, ts="2024-01-02"))) == []
    fill = broker.on_market(market(bar(open=99, low=97, ts="2024-01-03")))[0]
    assert fill.price == 98.0


def test_limit_order_gets_better_open_on_gap():
    broker = SimulatedBroker(max_participation=None)
    broker.submit(order(-10, order_type=OrderType.LIMIT, limit_price=100.0))
    fill = broker.on_market(market(bar(open=105, high=106, low=104)))[0]
    assert fill.price == 105  # gapped above the limit, sold at the open


def test_cancel_removes_open_orders():
    broker = SimulatedBroker()
    broker.submit(order(10))
    broker.cancel("X")
    assert broker.on_market(market(bar())) == []


def test_per_share_commission_minimum_and_cap():
    c = PerShareCommission(rate=0.005, minimum=1.0, max_pct=0.01)
    assert c(10, 100) == 1.0  # minimum
    assert c(1_000, 100) == 5.0  # per share
    assert c(1_000, 0.10) == pytest.approx(1.0)  # capped at 1% of $100
