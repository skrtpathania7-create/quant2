"""Event types that flow through the backtest event queue.

The lifecycle of one bar is:

    MarketEvent  -> broker fills orders resting from earlier bars (FillEvent)
                 -> portfolio marks to market
                 -> strategy emits target weights (SignalEvent)
    SignalEvent  -> portfolio sizes an order (OrderEvent)
    OrderEvent   -> broker queues it; it can fill no earlier than the NEXT bar

Because orders created on bar t can only fill on bar t+1 or later, a strategy
that decides on the close of bar t can never trade at that same close. That is
the main protection against look-ahead bias.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


@dataclass(frozen=True)
class Bar:
    symbol: str
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class MarketEvent:
    """A new bar is available for every symbol in ``bars``.

    Symbols with no data at this timestamp are simply absent.
    """

    timestamp: pd.Timestamp
    bars: dict[str, Bar]


@dataclass(frozen=True)
class SignalEvent:
    """The strategy wants ``target_weight`` of equity in ``symbol``.

    Weights are signed: 1.0 is fully long, -1.0 is fully short, 0.0 is flat.
    """

    timestamp: pd.Timestamp
    symbol: str
    target_weight: float


@dataclass
class OrderEvent:
    """An instruction to the broker. ``quantity`` is signed (+buy / -sell)."""

    order_id: int
    timestamp: pd.Timestamp
    symbol: str
    quantity: int
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    filled: int = 0

    def __post_init__(self) -> None:
        if self.quantity == 0:
            raise ValueError("order quantity must be non-zero")
        if self.order_type is OrderType.LIMIT and self.limit_price is None:
            raise ValueError("limit orders need a limit_price")

    @property
    def side(self) -> int:
        return 1 if self.quantity > 0 else -1

    @property
    def remaining(self) -> int:
        return self.quantity - self.filled


@dataclass(frozen=True)
class FillEvent:
    """An (possibly partial) execution of an order.

    ``price`` already includes slippage. ``slippage`` is the cost of that
    slippage in currency, kept separately so it can be reported.
    """

    timestamp: pd.Timestamp
    order_id: int
    symbol: str
    quantity: int
    price: float
    commission: float
    slippage: float
