"""quantbt: an event-driven backtesting engine."""

from .costs import (
    FixedBpsSlippage,
    NoCommission,
    NoSlippage,
    PercentCommission,
    PerShareCommission,
    VolumeImpactSlippage,
)
from .data import DataHandler, load_csv, synthetic_ohlcv
from .engine import Backtest
from .events import Bar, FillEvent, MarketEvent, OrderEvent, OrderType, SignalEvent
from .execution import SimulatedBroker
from .portfolio import Portfolio
from .results import BacktestResult
from .sources import CachedSource, DataSource, YFinanceSource, load_prices, validate_bars
from .strategy import Strategy
from .walkforward import WalkForwardResult, walk_forward

__all__ = [
    "Backtest",
    "CachedSource",
    "DataSource",
    "YFinanceSource",
    "load_prices",
    "validate_bars",
    "BacktestResult",
    "Bar",
    "DataHandler",
    "FillEvent",
    "FixedBpsSlippage",
    "MarketEvent",
    "NoCommission",
    "NoSlippage",
    "OrderEvent",
    "OrderType",
    "PerShareCommission",
    "PercentCommission",
    "Portfolio",
    "SignalEvent",
    "SimulatedBroker",
    "Strategy",
    "VolumeImpactSlippage",
    "WalkForwardResult",
    "load_csv",
    "synthetic_ohlcv",
    "walk_forward",
]


def realistic_broker(max_participation: float = 0.1) -> SimulatedBroker:
    """A broker with retail-style commissions and spread + impact slippage."""
    return SimulatedBroker(
        commission=PerShareCommission(),
        slippage=VolumeImpactSlippage(),
        max_participation=max_participation,
    )


__all__.append("realistic_broker")
