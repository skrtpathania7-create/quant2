"""Cash, positions, order sizing and P&L accounting."""

from __future__ import annotations

import math

import pandas as pd

from .events import FillEvent, MarketEvent, OrderEvent, SignalEvent


class Portfolio:
    """Tracks cash and positions and turns target weights into orders.

    Positions are carried at average cost. When a fill reduces or closes a
    position, the realized P&L is recorded as a trade.
    """

    def __init__(self, initial_cash: float = 100_000.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: dict[str, int] = {}
        self.avg_cost: dict[str, float] = {}
        self.last_price: dict[str, float] = {}
        self.fills: list[FillEvent] = []
        self.trades: list[dict] = []
        self._records: list[dict] = []
        self._position_records: list[dict] = []
        self._next_order_id = 1

    # -- valuation --------------------------------------------------------
    @property
    def equity(self) -> float:
        return self.cash + sum(q * self.last_price.get(s, 0.0) for s, q in self.positions.items())

    def update_prices(self, event: MarketEvent) -> None:
        for sym, bar in event.bars.items():
            self.last_price[sym] = bar.close

    def record(self, timestamp: pd.Timestamp) -> None:
        gross = sum(abs(q) * self.last_price.get(s, 0.0) for s, q in self.positions.items())
        self._records.append(
            {"timestamp": timestamp, "cash": self.cash, "equity": self.equity, "gross_exposure": gross}
        )
        self._position_records.append({"timestamp": timestamp, **self.positions})

    # -- orders and fills -------------------------------------------------
    def create_order(self, signal: SignalEvent) -> OrderEvent | None:
        """Size an order that moves the position to the signal's target weight."""
        price = self.last_price.get(signal.symbol)
        if price is None or not math.isfinite(price) or price <= 0:
            return None
        target = int(signal.target_weight * self.equity / price)  # truncates toward zero
        delta = target - self.positions.get(signal.symbol, 0)
        if delta == 0:
            return None
        order = OrderEvent(self._next_order_id, signal.timestamp, signal.symbol, delta)
        self._next_order_id += 1
        return order

    def on_fill(self, fill: FillEvent) -> None:
        self.fills.append(fill)
        sym, qty, px = fill.symbol, fill.quantity, fill.price
        old = self.positions.get(sym, 0)
        avg = self.avg_cost.get(sym, 0.0)
        new = old + qty

        if old == 0 or (old > 0) == (qty > 0):  # opening or adding
            self.avg_cost[sym] = (avg * abs(old) + px * abs(qty)) / abs(new)
        else:  # reducing, closing or flipping
            closed = min(abs(qty), abs(old))
            direction = 1 if old > 0 else -1
            self.trades.append(
                {
                    "timestamp": fill.timestamp,
                    "symbol": sym,
                    "quantity": closed * direction,
                    "entry_price": avg,
                    "exit_price": px,
                    "pnl": closed * (px - avg) * direction,
                }
            )
            if new == 0:
                self.avg_cost.pop(sym, None)
            elif (new > 0) != (old > 0):  # flipped through zero
                self.avg_cost[sym] = px

        if new == 0:
            self.positions.pop(sym, None)
        else:
            self.positions[sym] = new
        self.cash -= qty * px + fill.commission

    # -- outputs ----------------------------------------------------------
    def history(self) -> pd.DataFrame:
        return pd.DataFrame(self._records).set_index("timestamp") if self._records else pd.DataFrame()

    def position_history(self) -> pd.DataFrame:
        if not self._position_records:
            return pd.DataFrame()
        return pd.DataFrame(self._position_records).set_index("timestamp").fillna(0).astype(int)
