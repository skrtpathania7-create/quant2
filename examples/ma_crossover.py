"""Moving-average crossover: frictionless vs. realistic execution.

    python examples/ma_crossover.py                              # synthetic data
    python examples/ma_crossover.py --symbol SPY --start 2005-01-01  # real data (Yahoo Finance, cached)
    python examples/ma_crossover.py --csv SPY.csv                # your own daily OHLCV CSV
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from quantbt import Backtest, SimulatedBroker, load_csv, load_prices, realistic_broker, synthetic_ohlcv
from quantbt.metrics import format_metric
from quantbt.strategies import MovingAverageCrossover


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", help="ticker to download, e.g. SPY")
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--csv", help="daily OHLCV CSV (first column = date)")
    parser.add_argument("--fast", type=int, default=20)
    parser.add_argument("--slow", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.symbol:
        prices, label = load_prices(args.symbol, args.start, args.end)[args.symbol], args.symbol
    elif args.csv:
        prices, label = load_csv(args.csv), Path(args.csv).stem
    else:
        prices, label = synthetic_ohlcv(n=2520, mu=0.08, sigma=0.20, seed=args.seed), "synthetic data"

    runs = {
        "Frictionless": SimulatedBroker(max_participation=None),
        "Realistic costs": realistic_broker(),
    }
    results = {}
    for name, broker in runs.items():
        strategy = MovingAverageCrossover(fast=args.fast, slow=args.slow)
        results[name] = Backtest(prices, strategy, broker=broker).run()

    table = pd.DataFrame(
        {name: {k: format_metric(k, v) for k, v in r.metrics.items()} for name, r in results.items()}
    )
    hold = prices["close"]
    print(
        f"MA crossover ({args.fast}/{args.slow}) on {label}: {len(prices):,} bars, "
        f"{prices.index[0].date()} to {prices.index[-1].date()}"
    )
    print(f"Buy & hold return over the same period: {hold.iloc[-1] / hold.iloc[0] - 1:.2%}\n")
    print(table.to_string())

    out = Path("output")
    out.mkdir(exist_ok=True)
    path = out / "ma_crossover.png"
    results["Realistic costs"].plot(
        path, title=f"MA crossover {args.fast}/{args.slow} on {label} (realistic costs)", benchmark=prices["close"]
    )
    print(f"\nChart saved to {path}")


if __name__ == "__main__":
    main()
