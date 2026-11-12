import pandas as pd
import pytest

from quantbt import Bar, MarketEvent, Strategy


def make_bars(rows, start="2024-01-01"):
    """rows: list of (open, high, low, close, volume)."""
    index = pd.bdate_range(start, periods=len(rows))
    return pd.DataFrame(rows, index=index, columns=["open", "high", "low", "close", "volume"])


def bar(open=100.0, high=101.0, low=99.0, close=100.0, volume=1_000_000, ts="2024-01-02", symbol="X"):
    return Bar(symbol, pd.Timestamp(ts), open, high, low, close, volume)


def market(*bars):
    return MarketEvent(bars[0].timestamp, {b.symbol: b for b in bars})


class Scripted(Strategy):
    """Returns a pre-set target dict on given bar numbers."""

    def __init__(self, script: dict[int, dict[str, float]]):
        self.script = script

    def initialize(self, data):
        super().initialize(data)
        self.i = -1

    def on_bar(self, event):
        self.i += 1
        return self.script.get(self.i)


@pytest.fixture
def prices():
    return make_bars(
        [
            (100, 101, 99, 100, 1e6),
            (102, 104, 101, 103, 1e6),
            (104, 106, 103, 105, 1e6),
            (103, 104, 100, 101, 1e6),
            (101, 102, 98, 99, 1e6),
        ]
    )
