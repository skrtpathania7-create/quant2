"""Trend following: moving-average crossover."""

from __future__ import annotations

from ..data import DataHandler
from ..events import MarketEvent
from ..strategy import Strategy


class MovingAverageCrossover(Strategy):
    """Long while the fast moving average is above the slow one.

    When the fast average is below, the strategy is flat, or short if
    ``allow_short`` is set. Capital is split equally across symbols.
    """

    def __init__(
        self,
        fast: int = 20,
        slow: int = 100,
        allow_short: bool = False,
        weight: float = 1.0,
        symbols: list[str] | None = None,
    ):
        if not 0 < fast < slow:
            raise ValueError("need 0 < fast < slow")
        self.fast, self.slow = fast, slow
        self.allow_short, self.weight = allow_short, weight
        self.symbols = symbols

    def initialize(self, data: DataHandler) -> None:
        super().initialize(data)
        self._symbols = self.symbols or data.symbols

    def on_bar(self, event: MarketEvent) -> dict[str, float]:
        per_symbol = self.weight / len(self._symbols)
        targets = {}
        for sym in self._symbols:
            if sym not in event.bars:
                continue
            closes = self.data.history(sym, self.slow)
            if len(closes) < self.slow:
                continue
            if closes[-self.fast :].mean() > closes.mean():
                targets[sym] = per_symbol
            else:
                targets[sym] = -per_symbol if self.allow_short else 0.0
        return targets
