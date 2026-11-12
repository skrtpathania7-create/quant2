"""A simulated broker that fills orders against later bars."""

from __future__ import annotations

import math

from .costs import CommissionModel, NoCommission, NoSlippage, SlippageModel
from .events import FillEvent, MarketEvent, OrderEvent, OrderType


class SimulatedBroker:
    """Fills resting orders against each new bar.

    Fill rules:

    * An order never fills on the bar it was created on. Orders sent after the
      close of bar t are first eligible on bar t+1.
    * Market orders fill at the bar's open, plus slippage.
    * Limit orders fill only if the bar traded through the limit, at the
      better of the open and the limit (no slippage: the limit is the price).
    * If ``max_participation`` is set, each bar fills at most that fraction of
      the bar's volume. The rest stays open for later bars (partial fills).
    """

    def __init__(
        self,
        commission: CommissionModel | None = None,
        slippage: SlippageModel | None = None,
        max_participation: float | None = 0.1,
    ):
        self.commission = commission or NoCommission()
        self.slippage = slippage or NoSlippage()
        self.max_participation = max_participation
        self.open_orders: dict[int, OrderEvent] = {}

    def reset(self) -> None:
        self.open_orders.clear()

    def submit(self, order: OrderEvent) -> None:
        self.open_orders[order.order_id] = order

    def cancel(self, symbol: str) -> None:
        """Cancel every open order for ``symbol``."""
        for oid in [o for o, order in self.open_orders.items() if order.symbol == symbol]:
            del self.open_orders[oid]

    def on_market(self, event: MarketEvent) -> list[FillEvent]:
        fills: list[FillEvent] = []
        for order in list(self.open_orders.values()):
            bar = event.bars.get(order.symbol)
            if bar is None or order.timestamp >= event.timestamp:
                continue

            qty = order.remaining
            if self.max_participation is not None:
                cap = math.floor(self.max_participation * bar.volume)
                qty = order.side * min(abs(qty), cap)
            if qty == 0:
                continue

            if order.order_type is OrderType.MARKET:
                reference = bar.open
                price = self.slippage.price(order.side, qty, bar, reference)
            else:
                limit = order.limit_price
                if order.side > 0:
                    if bar.low > limit:
                        continue
                    reference = price = min(bar.open, limit)
                else:
                    if bar.high < limit:
                        continue
                    reference = price = max(bar.open, limit)

            order.filled += qty
            if order.remaining == 0:
                del self.open_orders[order.order_id]
            fills.append(
                FillEvent(
                    timestamp=event.timestamp,
                    order_id=order.order_id,
                    symbol=order.symbol,
                    quantity=qty,
                    price=price,
                    commission=self.commission(qty, price),
                    slippage=abs(qty) * abs(price - reference),
                )
            )
        return fills
