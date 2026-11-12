"""Market data: a bar-by-bar feed with no look-ahead, plus loaders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .events import Bar, MarketEvent

FIELDS = ("open", "high", "low", "close", "volume")


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    missing = [c for c in FIELDS if c not in df.columns]
    if missing:
        raise ValueError(f"price data is missing columns: {missing}")
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index))
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df[list(FIELDS)].astype(float)


class DataHandler:
    """Replays OHLCV bars for one or more symbols, one timestamp at a time.

    Strategies read past data through :meth:`history`, which only returns bars
    up to and including the current one, so they cannot see the future.
    """

    def __init__(self, data: pd.DataFrame | dict[str, pd.DataFrame]):
        if isinstance(data, pd.DataFrame):
            data = {"ASSET": data}
        if not data:
            raise ValueError("no data given")
        self._frames = {sym: _normalize(df) for sym, df in data.items()}
        index = pd.DatetimeIndex([])
        for frame in self._frames.values():
            index = index.union(frame.index)
        if len(index) == 0:
            raise ValueError("data has no rows")
        self._index = index
        self._arrays = {s: f.reindex(index).to_numpy() for s, f in self._frames.items()}
        self._cursor = -1

    # -- feed -------------------------------------------------------------
    @property
    def symbols(self) -> list[str]:
        return list(self._frames)

    @property
    def index(self) -> pd.DatetimeIndex:
        return self._index

    def __len__(self) -> int:
        return len(self._index)

    @property
    def current_time(self) -> pd.Timestamp | None:
        return self._index[self._cursor] if self._cursor >= 0 else None

    def reset(self) -> None:
        self._cursor = -1

    def has_next(self) -> bool:
        return self._cursor + 1 < len(self._index)

    def next(self) -> MarketEvent:
        if not self.has_next():
            raise StopIteration
        self._cursor += 1
        ts = self._index[self._cursor]
        bars: dict[str, Bar] = {}
        for sym, arr in self._arrays.items():
            row = arr[self._cursor]
            if np.isnan(row[3]):  # no bar for this symbol at this time
                continue
            o, h, l, c, v = (float(x) for x in row)
            bars[sym] = Bar(sym, ts, o, h, l, c, v)
        return MarketEvent(ts, bars)

    # -- point-in-time access ---------------------------------------------
    def history(self, symbol: str, n: int | None = None, field: str = "close") -> np.ndarray:
        """The last ``n`` values of ``field`` up to and including the current bar."""
        if self._cursor < 0:
            return np.empty(0)
        col = FIELDS.index(field)
        arr = self._arrays[symbol]
        end = self._cursor + 1
        if n is not None:
            window = arr[max(0, end - n) : end, col]
            if not np.isnan(window).any():
                return window
        values = arr[:end, col]
        values = values[~np.isnan(values)]
        return values if n is None else values[-n:]

    def slice(self, start, end) -> "DataHandler":
        """A new handler over [start, end] (inclusive, by timestamp)."""
        return DataHandler({s: f.loc[start:end] for s, f in self._frames.items()})


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load an OHLCV CSV whose first column is the date (e.g. a Yahoo export)."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return _normalize(df)


def synthetic_ohlcv(
    n: int = 2520,
    start: str = "2015-01-02",
    s0: float = 100.0,
    mu: float = 0.08,
    sigma: float = 0.20,
    mean_reversion: float = 0.0,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate realistic-looking daily OHLCV bars.

    Log price = drifting trend + deviation, where the deviation follows
    ``dev_t = (1 - mean_reversion) * dev_{t-1} + noise``. With
    ``mean_reversion=0`` this is geometric Brownian motion; with values like
    0.05 the price oscillates around its trend, which mean-reversion
    strategies can exploit.
    """
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    step_sd = sigma * np.sqrt(dt)
    trend = np.log(s0) + (mu - 0.5 * sigma**2) * dt * np.arange(n)
    dev = np.empty(n)
    d = 0.0
    shocks = rng.normal(0.0, step_sd, n)
    for i in range(n):
        d = (1 - mean_reversion) * d + shocks[i]
        dev[i] = d
    close = np.exp(trend + dev)

    prev_close = np.concatenate([[s0], close[:-1]])
    open_ = prev_close * np.exp(rng.normal(0.0, 0.25 * step_sd, n))
    wick = 0.5 * step_sd
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0.0, wick, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0.0, wick, n)))
    volume = np.round(rng.lognormal(np.log(1_000_000), 0.3, n))

    index = pd.bdate_range(start, periods=n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )
