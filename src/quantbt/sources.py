"""Real market data: pluggable sources, an on-disk cache and sanity checks.

To add a provider (Alpaca, NSE bhavcopy, ...), subclass :class:`DataSource`
and implement ``fetch``. Everything else (caching, validation, the engine)
works unchanged.
"""

from __future__ import annotations

import json
import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from .data import FIELDS, _normalize

DEFAULT_CACHE_DIR = Path("data") / "cache"


class DataSource(ABC):
    """Returns daily OHLCV bars, adjusted for splits and dividends."""

    name: str = "source"

    @abstractmethod
    def fetch(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Bars for ``symbol`` over [start, end], indexed by date (no timezone)."""


class YFinanceSource(DataSource):
    """Yahoo Finance via the ``yfinance`` package (``pip install yfinance``).

    Free and keyless, but unofficial: expect occasional outages, rate limits
    and bad prints, and note it has no delisted tickers (survivorship bias).
    US tickers are plain (``SPY``); NSE tickers take a suffix (``RELIANCE.NS``).
    """

    name = "yfinance"

    def fetch(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError('yfinance is not installed. Run: pip install "quantbt[data]"') from exc

        # yfinance's `end` is exclusive
        raw = yf.Ticker(symbol).history(
            start=start.strftime("%Y-%m-%d"),
            end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=True,
        )
        if raw is None or raw.empty:
            raise ValueError(f"no data returned for {symbol!r} between {start.date()} and {end.date()}")
        raw.index = pd.DatetimeIndex(raw.index).tz_localize(None).normalize()
        return _normalize(raw).dropna(subset=["close"])


class CachedSource(DataSource):
    """Wraps another source and stores each symbol as a CSV on disk.

    A request is served from disk when it falls inside the date range already
    downloaded; otherwise the range is extended and re-downloaded. Adjusted
    prices shift whenever a dividend or split happens, so pass
    ``refresh=True`` now and then to rebuild the cache.
    """

    def __init__(self, source: DataSource, cache_dir: str | Path = DEFAULT_CACHE_DIR, refresh: bool = False):
        self.source = source
        self.name = source.name
        self.dir = Path(cache_dir) / source.name
        self.refresh = refresh

    def _paths(self, symbol: str) -> tuple[Path, Path]:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in symbol)
        return self.dir / f"{safe}.csv", self.dir / f"{safe}.json"

    def fetch(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        csv_path, meta_path = self._paths(symbol)
        if not self.refresh and csv_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            have_start, have_end = pd.Timestamp(meta["start"]), pd.Timestamp(meta["end"])
            if have_start <= start and end <= have_end:
                cached = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                return cached.loc[start:end]
            start_dl, end_dl = min(start, have_start), max(end, have_end)
        else:
            start_dl, end_dl = start, end

        fresh = self.source.fetch(symbol, start_dl, end_dl)
        self.dir.mkdir(parents=True, exist_ok=True)
        fresh.to_csv(csv_path)
        meta_path.write_text(json.dumps({"start": str(start_dl.date()), "end": str(end_dl.date())}))
        return fresh.loc[start:end]


def validate_bars(
    df: pd.DataFrame, max_jump: float = 0.25, max_gap_days: int = 7
) -> pd.DataFrame:
    """List suspicious bars. An empty result means nothing was flagged.

    Checks: missing values, non-positive prices, high/low inconsistent with
    open/close, zero volume, one-day moves larger than ``max_jump`` and
    calendar gaps longer than ``max_gap_days``. Free data has errors, and a
    single bad print can create a fake, very profitable trade.
    """
    df = _normalize(df)
    prices = df[["open", "high", "low", "close"]]
    eps = 1e-9 * prices.abs().max(axis=1)
    checks = {
        "missing value": df[list(FIELDS)].isna().any(axis=1),
        "non-positive price": (prices <= 0).any(axis=1),
        "high below open/close": df["high"] < prices[["open", "close"]].max(axis=1) - eps,
        "low above open/close": df["low"] > prices[["open", "close"]].min(axis=1) + eps,
        "zero volume": df["volume"] == 0,
        f"move over {max_jump:.0%}": df["close"].pct_change().abs() > max_jump,
        f"gap over {max_gap_days} days": pd.Series(
            np.diff(df.index.to_numpy(), prepend=df.index.to_numpy()[:1]) > np.timedelta64(max_gap_days, "D"),
            index=df.index,
        ),
    }
    rows = [
        {"date": ts, "check": name, "close": df.at[ts, "close"]}
        for name, mask in checks.items()
        for ts in df.index[mask.to_numpy()]
    ]
    return pd.DataFrame(rows, columns=["date", "check", "close"]).sort_values("date", ignore_index=True)


def load_prices(
    symbols: str | list[str],
    start: str | pd.Timestamp = "2010-01-01",
    end: str | pd.Timestamp | None = None,
    source: DataSource | None = None,
    validate: bool = True,
) -> dict[str, pd.DataFrame]:
    """Download (or read from cache) adjusted daily bars for each symbol.

    Returns ``{symbol: DataFrame}``, ready to pass to :class:`Backtest`.
    Uses cached Yahoo Finance data unless another ``source`` is given.
    """
    if isinstance(symbols, str):
        symbols = [symbols]
    source = source or CachedSource(YFinanceSource())
    start = pd.Timestamp(start).normalize()
    end = (pd.Timestamp(end) if end is not None else pd.Timestamp.today()).normalize()
    if start > end:
        raise ValueError("start is after end")

    out: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        frame = source.fetch(symbol, start, end)
        if frame.empty:
            raise ValueError(f"no data for {symbol!r} between {start.date()} and {end.date()}")
        if validate:
            issues = validate_bars(frame)
            if len(issues):
                counts = issues["check"].value_counts()
                detail = ", ".join(f"{n} x {c}" for c, n in counts.items())
                warnings.warn(f"{symbol}: {len(issues)} suspicious bars ({detail})", stacklevel=2)
        out[symbol] = frame
    return out
