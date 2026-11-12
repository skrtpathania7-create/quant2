"""The event loop that ties data, strategy, portfolio and broker together."""

from __future__ import annotations

import math
from collections import deque

import pandas as pd

from .data import DataHandler
from .events import MarketEvent, OrderEvent, SignalEvent
from .execution import SimulatedBroker
from .portfolio import Portfolio
from .results import BacktestResult
from .strategy import Strategy


class Backtest:
    """Run a strategy over historical data.

    Parameters
    ----------
    data:
        A :class:`DataHandler`, a single OHLCV DataFrame, or a dict of them.
    strategy:
        The strategy to run.
    broker:
        Fill and cost model. Defaults to a frictionless broker.
    start_trading:
        Optional timestamp. Bars before it are still shown to the strategy
        (to warm up indicators) but it cannot trade and they are not part of
        the results. Walk-forward testing uses this.
    """

    def __init__(
        self,
        data: DataHandler | pd.DataFrame | dict[str, pd.DataFrame],
        strategy: Strategy,
        *,
        initial_cash: float = 100_000.0,
        broker: SimulatedBroker | None = None,
        start_trading=None,
        periods_per_year: int = 252,
        risk_free_rate: float = 0.0,
    ):
        self.data = data if isinstance(data, DataHandler) else DataHandler(data)
        self.strategy = strategy
        self.initial_cash = initial_cash
        self.broker = broker or SimulatedBroker(max_participation=None)
        self.start_trading = pd.Timestamp(start_trading) if start_trading is not None else None
        self.periods_per_year = periods_per_year
        self.risk_free_rate = risk_free_rate

    def run(self) -> BacktestResult:
        data, broker, strategy = self.data, self.broker, self.strategy
        data.reset()
        broker.reset()
        portfolio = Portfolio(self.initial_cash)
        strategy.initialize(data)

        desired: dict[str, float] = {}  # latest target per symbol, kept through warm-up
        last_sent: dict[str, float] = {}
        queue: deque = deque()

        while data.has_next():
            queue.append(data.next())
            while queue:
                event = queue.popleft()

                if isinstance(event, MarketEvent):
                    trading = self.start_trading is None or event.timestamp >= self.start_trading
                    for fill in broker.on_market(event):
                        portfolio.on_fill(fill)
                    portfolio.update_prices(event)
                    if trading:
                        portfolio.record(event.timestamp)

                    desired.update(strategy.on_bar(event) or {})
                    if not trading:
                        continue
                    for symbol, weight in desired.items():
                        if symbol not in event.bars:
                            continue  # no price to size against; retry next bar
                        previous = last_sent.get(symbol)
                        if previous is None or not math.isclose(previous, weight, abs_tol=1e-12):
                            last_sent[symbol] = weight
                            queue.append(SignalEvent(event.timestamp, symbol, float(weight)))

                elif isinstance(event, SignalEvent):
                    broker.cancel(event.symbol)  # replace any unfilled remainder
                    order = portfolio.create_order(event)
                    if order is not None:
                        queue.append(order)

                elif isinstance(event, OrderEvent):
                    broker.submit(event)

        return BacktestResult.from_portfolio(
            portfolio, periods_per_year=self.periods_per_year, risk_free_rate=self.risk_free_rate
        )
