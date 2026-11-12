"""Base class for strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .data import DataHandler
from .events import MarketEvent


class Strategy(ABC):
    """Subclass this and implement :meth:`on_bar`.

    ``on_bar`` runs after the close of every bar and returns the target
    portfolio weight for any symbols it has an opinion on, e.g.
    ``{"SPY": 1.0}`` for fully long or ``{"SPY": 0.0}`` for flat. Symbols left
    out keep their current target. The engine only sends a signal when a
    target changes, so returning the same weights every bar is fine.

    Read past prices with ``self.data.history(symbol, n)``; it never returns
    bars after the current one.
    """

    data: DataHandler

    def initialize(self, data: DataHandler) -> None:
        self.data = data

    @abstractmethod
    def on_bar(self, event: MarketEvent) -> dict[str, float] | None: ...
