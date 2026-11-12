"""Mean reversion: fade large deviations from a rolling mean."""

from __future__ import annotations

from ..data import DataHandler
from ..events import MarketEvent
from ..strategy import Strategy


class ZScoreMeanReversion(Strategy):
    """Trade the z-score of price against its rolling mean.

    z = (close - rolling mean) / rolling std over ``lookback`` bars.

    * Go long when z <= -entry_z (price unusually low).
    * Go short when z >= entry_z (price unusually high), if shorting is allowed.
    * Exit when z comes back within ``exit_z`` of zero.
    """

    def __init__(
        self,
        lookback: int = 20,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        allow_short: bool = True,
        weight: float = 1.0,
        symbols: list[str] | None = None,
    ):
        if lookback < 2:
            raise ValueError("lookback must be at least 2")
        if not 0 <= exit_z < entry_z:
            raise ValueError("need 0 <= exit_z < entry_z")
        self.lookback, self.entry_z, self.exit_z = lookback, entry_z, exit_z
        self.allow_short, self.weight = allow_short, weight
        self.symbols = symbols

    def initialize(self, data: DataHandler) -> None:
        super().initialize(data)
        self._symbols = self.symbols or data.symbols
        self._state = {s: 0 for s in self._symbols}  # -1 short, 0 flat, 1 long

    def on_bar(self, event: MarketEvent) -> dict[str, float]:
        per_symbol = self.weight / len(self._symbols)
        targets = {}
        for sym in self._symbols:
            if sym not in event.bars:
                continue
            closes = self.data.history(sym, self.lookback)
            if len(closes) < self.lookback:
                continue
            sd = closes.std(ddof=1)
            if sd == 0:
                continue
            z = (closes[-1] - closes.mean()) / sd

            state = self._state[sym]
            if state == 0:
                if z <= -self.entry_z:
                    state = 1
                elif z >= self.entry_z and self.allow_short:
                    state = -1
            elif state == 1 and z >= -self.exit_z:
                state = 0
            elif state == -1 and z <= self.exit_z:
                state = 0
            self._state[sym] = state
            targets[sym] = state * per_symbol
        return targets
