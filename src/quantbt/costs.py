"""Commission and slippage models.

Commission models return a cost in currency for one fill.
Slippage models turn a reference price (e.g. the bar's open) into the price
actually paid, always moving it against the trader.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

from .events import Bar


# -- commissions ----------------------------------------------------------
class CommissionModel(ABC):
    @abstractmethod
    def __call__(self, quantity: int, price: float) -> float: ...


class NoCommission(CommissionModel):
    def __call__(self, quantity: int, price: float) -> float:
        return 0.0


class PerShareCommission(CommissionModel):
    """Per-share pricing with a minimum and a cap, like many retail brokers.

    Defaults: $0.005 per share, at least $1.00, at most 1% of trade value.
    """

    def __init__(self, rate: float = 0.005, minimum: float = 1.0, max_pct: float = 0.01):
        self.rate, self.minimum, self.max_pct = rate, minimum, max_pct

    def __call__(self, quantity: int, price: float) -> float:
        fee = max(abs(quantity) * self.rate, self.minimum)
        return min(fee, abs(quantity) * price * self.max_pct)


class PercentCommission(CommissionModel):
    """A fixed fraction of traded value, in basis points."""

    def __init__(self, bps: float = 1.0):
        self.bps = bps

    def __call__(self, quantity: int, price: float) -> float:
        return abs(quantity) * price * self.bps / 1e4


# -- slippage -------------------------------------------------------------
class SlippageModel(ABC):
    @abstractmethod
    def price(self, side: int, quantity: int, bar: Bar, reference: float) -> float:
        """Execution price for ``quantity`` shares; ``side`` is +1 buy, -1 sell."""


class NoSlippage(SlippageModel):
    def price(self, side: int, quantity: int, bar: Bar, reference: float) -> float:
        return reference


class FixedBpsSlippage(SlippageModel):
    """Every fill pays a fixed number of basis points."""

    def __init__(self, bps: float = 5.0):
        self.bps = bps

    def price(self, side: int, quantity: int, bar: Bar, reference: float) -> float:
        return reference * (1 + side * self.bps / 1e4)


class VolumeImpactSlippage(SlippageModel):
    """Half the bid-ask spread plus square-root market impact.

    cost = spread / 2 + impact * volatility * sqrt(quantity / bar volume)

    The bar's high-low range relative to its open stands in for volatility.
    The square-root law is the standard empirical model of how much a trade
    moves the price: twice the size costs about 1.4x as much per share.
    """

    def __init__(self, spread_bps: float = 2.0, impact: float = 0.5):
        self.spread_bps, self.impact = spread_bps, impact

    def price(self, side: int, quantity: int, bar: Bar, reference: float) -> float:
        participation = abs(quantity) / bar.volume if bar.volume > 0 else 1.0
        volatility = (bar.high - bar.low) / bar.open if bar.open > 0 else 0.0
        cost = self.spread_bps / 2 / 1e4 + self.impact * volatility * math.sqrt(participation)
        return reference * (1 + side * cost)
