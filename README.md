# quantbt

An event-driven backtesting engine in Python, built to give **honest** results:
no look-ahead bias, realistic fills, transaction costs, market impact, and
walk-forward (out-of-sample) testing.

```
MarketEvent ──► Broker fills resting orders ──► Portfolio marks to market ──► Strategy
                     ▲                                                          │
                     │                                                    target weights
                OrderEvent ◄──── Portfolio sizes order ◄──── SignalEvent ◄──────┘
```

## Why event-driven?

Vectorized backtests (`signal.shift(1) * returns`) are fast, but they quietly
assume you can trade any size at the close, for free. This engine replays the
market one bar at a time through the same event flow a live system would use,
so the rules that make a backtest realistic are enforced in one place:

| Concern | How it's handled |
|---|---|
| Look-ahead bias | Strategies only see data up to the current bar. Orders created on bar *t* can't fill before bar *t+1*. |
| Fill price | Market orders fill at the next bar's **open**, not the signal bar's close. |
| Slippage | Pluggable models: fixed bps, or **half-spread + square-root market impact** scaled by volatility and order size vs. volume. |
| Commissions | Per-share with minimum and cap, or a percentage of traded value. |
| Liquidity | Fills are capped at a fraction of bar volume (default 10%). Large orders fill **partially** over several bars. |
| Limit orders | Fill only if the bar trades through the limit, at the better of open and limit. |
| Overfitting | **Walk-forward optimization**: tune on a rolling training window, trade the next unseen window, stitch only the out-of-sample results together. |

## Quick start

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -e ".[dev]"

python examples/ma_crossover.py   # trend following, frictionless vs. realistic costs
python examples/walk_forward.py   # mean reversion, walk-forward optimized
pytest                            # 37 tests, no network needed

python examples/ma_crossover.py --symbol SPY --start 2005-01-01   # real data
python examples/walk_forward.py --symbol SPY --start 2005-01-01
```

Without arguments the examples run on synthetic data. `--symbol` downloads
adjusted daily bars from Yahoo Finance (cached in `data/cache/`), and
`--csv path/to/file.csv` reads any daily OHLCV file.

## Market data

```python
from quantbt import load_prices, validate_bars

prices = load_prices(["SPY", "QQQ"], start="2005-01-01")   # {symbol: DataFrame}
validate_bars(prices["SPY"])                               # table of suspicious bars
```

- **Pluggable sources.** `DataSource` is a one-method interface. `YFinanceSource`
  is the first implementation; a broker or exchange feed is one more subclass.
- **Cache.** `CachedSource` stores each symbol as a CSV and only re-downloads
  when a request goes outside the range already on disk.
- **Validation.** Every load is checked for missing values, non-positive
  prices, impossible high/low values, zero volume, one-day moves over 25% and
  long calendar gaps, and warns if anything is flagged. One bad print in free
  data can create a fake, very profitable trade.

Yahoo data is unofficial and has no delisted tickers, so backtests over
"today's index members" suffer from survivorship bias. Prices are adjusted
for splits and dividends, so they shift over time; rebuild the cache with
`CachedSource(YFinanceSource(), refresh=True)`.

## Usage

```python
from quantbt import Backtest, Strategy, realistic_broker, synthetic_ohlcv

class BuyTheDip(Strategy):
    def on_bar(self, event):
        closes = self.data.history("ASSET", 5)       # never includes future bars
        if len(closes) < 5:
            return None
        return {"ASSET": 1.0 if closes[-1] < closes.mean() else 0.0}

result = Backtest(synthetic_ohlcv(seed=1), BuyTheDip(), broker=realistic_broker()).run()
print(result.summary())
result.plot("equity.png")
```

A strategy returns **target portfolio weights** (`1.0` = all-in long, `-0.5` =
half-short, `0.0` = flat). The portfolio converts them into share orders, and
the engine only sends a signal when a target actually changes.

Walk-forward:

```python
from quantbt import walk_forward
from quantbt.strategies import ZScoreMeanReversion

wf = walk_forward(
    prices, ZScoreMeanReversion,
    {"lookback": [10, 20, 40], "entry_z": [1.5, 2.0, 2.5]},
    train_size=504, test_size=126, broker_factory=realistic_broker,
)
print(wf.folds_frame())   # chosen parameters and in/out-of-sample Sharpe per fold
print(wf.summary())       # metrics of the stitched out-of-sample equity curve
```

## Example output

Walk-forward z-score mean reversion on synthetic mean-reverting data, with
realistic costs (16 folds, 2-year train / 6-month test):

```
total_return             132.91%
cagr                      11.15%
sharpe                      0.94
max_drawdown             -16.85%
```

The MA crossover example on a pure random walk **loses money**, as it should:
there is no trend to follow, and costs push the result further below zero. A
backtester that showed a profit here would be one to distrust.

On real data (SPY, Jan 2005 to Nov 2026, realistic costs), neither example
strategy beats simply holding the index, which returned 838% with dividends:

| Strategy | Total return | Sharpe | Max drawdown |
|---|---|---|---|
| MA crossover 20/100 | 310% | 0.60 | -26.9% |
| Z-score mean reversion, walk-forward | 68% | 0.27 | -30.6% |

They are examples of how to use the engine, not trading recommendations.

## Metrics

Total return, CAGR, annualized volatility, Sharpe, Sortino, max drawdown,
max drawdown duration, Calmar, number of fills, commission and slippage paid,
annual turnover, number of round-trip trades, win rate, profit factor.

## Project layout

```
src/quantbt/
  events.py        event types (Market, Signal, Order, Fill) and Bar
  data.py          DataHandler (point-in-time bar feed), CSV loader, synthetic data
  sources.py       real data: DataSource interface, Yahoo source, disk cache, validation
  costs.py         commission and slippage models
  execution.py     SimulatedBroker: fill rules, partial fills, limit orders
  portfolio.py     cash, positions, average-cost P&L, order sizing
  engine.py        the event loop
  metrics.py       performance statistics
  results.py       BacktestResult: equity, fills, trades, summary, plot
  walkforward.py   rolling walk-forward optimization
  strategies/      MovingAverageCrossover, ZScoreMeanReversion
tests/             unit tests for fills, accounting, metrics, look-ahead, walk-forward
examples/          runnable demos (charts go to output/)
```

## Known simplifications

- Bars are daily OHLCV; there is no intrabar order book.
- No margin requirements, borrow fees or short-sale constraints.
- Orders are sized from the signal bar's close, so a 100% target can slightly
  overshoot available cash when the next open gaps up.
- In walk-forward, each test window starts flat with fresh capital, and open
  positions at the end of a window are valued at the last close, not closed out.

## Roadmap

- Stop and stop-limit orders, bracket orders
- Position sizing by volatility targeting
- Benchmark-relative metrics (alpha, beta, information ratio)
- Monte Carlo / bootstrap confidence intervals on Sharpe
- More data sources (Alpaca, NSE bhavcopy) and an Indian-market cost model
- Paper trading through a broker API
