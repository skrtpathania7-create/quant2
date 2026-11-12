# quantbt — event-driven backtesting engine

Portfolio project meant to show recruiters solid engineering: realistic fills,
transaction costs, slippage, walk-forward testing, Sharpe/drawdown metrics,
with two example strategies (MA crossover, z-score mean reversion).

## Commands

- Setup: `python -m venv .venv` then `.venv/Scripts/python -m pip install -e ".[dev]"`
- Tests: `.venv/Scripts/python -m pytest -q`
- Demos: `.venv/Scripts/python examples/ma_crossover.py`, `examples/walk_forward.py` (charts land in `output/`, gitignored)

## Invariants — don't break these

- No look-ahead: `DataHandler.history` returns only bars up to the current
  cursor; the broker never fills an order on the bar it was created on.
- Market orders fill at the next bar's open plus slippage; limit orders get no slippage.
- Strategies return target weights; the engine keeps the latest desired target
  per symbol (including through warm-up) and only emits a SignalEvent when it changes.
- A new signal for a symbol cancels that symbol's unfilled orders, then the
  portfolio re-sizes from the current position.
- Every bug fix or new fill rule gets a test in `tests/`.

## Data

- `load_prices(symbols, start, end)` in `sources.py` downloads adjusted daily
  bars via yfinance, cached as CSV under `data/cache/` (gitignored).
- New providers subclass `DataSource`. Tests must use a fake source, never the network.
- Plan: US market first; Indian market (NSE, own cost model, shorting rules) later.

## Layout

`src/quantbt/`: events, data, costs, execution (broker), portfolio, engine,
metrics, results, walkforward, strategies/. See README for the roadmap.
