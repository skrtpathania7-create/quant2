"""Walk-forward optimization of a z-score mean-reversion strategy.

Each fold tunes the lookback and entry threshold on 2 years of data, then
trades the next 6 months with the winning parameters. Only those unseen
6-month windows count toward the final result.

    python examples/walk_forward.py
    python examples/walk_forward.py --symbol SPY --start 2005-01-01
    python examples/walk_forward.py --csv SPY.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quantbt import load_csv, load_prices, realistic_broker, synthetic_ohlcv, walk_forward
from quantbt.strategies import ZScoreMeanReversion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", help="ticker to download, e.g. SPY")
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--csv", help="daily OHLCV CSV (first column = date)")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    if args.symbol:
        prices = load_prices(args.symbol, args.start, args.end)[args.symbol]
    elif args.csv:
        prices = load_csv(args.csv)
    else:
        prices = synthetic_ohlcv(n=2520, mu=0.05, sigma=0.20, mean_reversion=0.05, seed=args.seed)

    grid = {"lookback": [10, 20, 40], "entry_z": [1.5, 2.0, 2.5]}
    result = walk_forward(
        prices,
        ZScoreMeanReversion,
        grid,
        train_size=504,  # ~2 years
        test_size=126,  # ~6 months
        broker_factory=realistic_broker,
    )

    print(result.folds_frame().to_string(index=False))
    print()
    print(result.summary())

    out = Path("output")
    out.mkdir(exist_ok=True)
    path = out / "walk_forward.png"
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(result.oos_equity.index, result.oos_equity, lw=1.4)
    for fold in result.folds:
        ax.axvline(fold.test_start, color="grey", lw=0.6, ls="--")
    ax.set_title("Walk-forward out-of-sample equity (dashed lines = re-optimization)")
    ax.set_ylabel("Equity")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"\nChart saved to {path}")


if __name__ == "__main__":
    main()
