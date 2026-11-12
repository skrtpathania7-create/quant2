"""Walk-forward optimization: tune on the past, test on the unseen future.

The data is split into rolling folds:

    |---- train ----|-- test --|
              |---- train ----|-- test --|
                        |---- train ----|-- test --|

For each fold, every parameter combination is backtested on the training
window and the best one (by ``objective``) is then run on the test window,
which it has never seen. Only the test-window results are stitched together,
so the combined equity curve is genuinely out-of-sample.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from . import metrics as m
from .data import DataHandler
from .engine import Backtest
from .execution import SimulatedBroker
from .strategy import Strategy


@dataclass
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_params: dict
    train_score: float
    test_metrics: dict


@dataclass
class WalkForwardResult:
    folds: list[Fold]
    oos_returns: pd.Series
    initial_cash: float
    periods_per_year: int = 252

    @property
    def oos_equity(self) -> pd.Series:
        return (self.initial_cash * (1 + self.oos_returns).cumprod()).rename("equity")

    @property
    def metrics(self) -> dict[str, float]:
        eq = pd.concat(
            [pd.Series([self.initial_cash], index=[self.oos_returns.index[0] - pd.Timedelta(days=1)]), self.oos_equity]
        )
        return m.summarize(eq, periods_per_year=self.periods_per_year)

    def folds_frame(self) -> pd.DataFrame:
        rows = []
        for f in self.folds:
            rows.append(
                {
                    "test_start": f.test_start.date(),
                    "test_end": f.test_end.date(),
                    **{f"param_{k}": v for k, v in f.best_params.items()},
                    "train_sharpe": round(f.train_score, 2),
                    "test_sharpe": round(f.test_metrics["sharpe"], 2),
                    "test_return": f"{f.test_metrics['total_return']:.2%}",
                }
            )
        return pd.DataFrame(rows)

    def summary(self) -> str:
        keys = ["total_return", "cagr", "annual_volatility", "sharpe", "sortino", "max_drawdown", "calmar"]
        lines = ["Walk-forward (out-of-sample only)", "-" * 34]
        lines += [f"{k:<18}  {m.format_metric(k, self.metrics[k]):>12}" for k in keys]
        return "\n".join(lines)


def walk_forward(
    data: DataHandler | pd.DataFrame | dict[str, pd.DataFrame],
    strategy_factory: Callable[..., Strategy],
    param_grid: dict[str, list],
    *,
    train_size: int,
    test_size: int,
    step: int | None = None,
    warmup: int | None = None,
    objective: str = "sharpe",
    broker_factory: Callable[[], SimulatedBroker] | None = None,
    initial_cash: float = 100_000.0,
    periods_per_year: int = 252,
) -> WalkForwardResult:
    """Run a rolling walk-forward optimization.

    ``strategy_factory(**params)`` must build a fresh strategy. ``warmup`` is
    how many bars before each test window are replayed (without trading) so
    indicators are ready; it defaults to ``train_size``.
    """
    handler = data if isinstance(data, DataHandler) else DataHandler(data)
    index = handler.index
    step = step or test_size
    warmup = train_size if warmup is None else warmup
    broker_factory = broker_factory or (lambda: SimulatedBroker(max_participation=None))
    names = list(param_grid)
    combos = [dict(zip(names, values)) for values in itertools.product(*param_grid.values())]
    if not combos:
        raise ValueError("param_grid is empty")

    def run(sub: DataHandler, params: dict, start_trading=None):
        return Backtest(
            sub,
            strategy_factory(**params),
            initial_cash=initial_cash,
            broker=broker_factory(),
            start_trading=start_trading,
            periods_per_year=periods_per_year,
        ).run()

    folds: list[Fold] = []
    pieces: list[pd.Series] = []
    start = 0
    while start + train_size + test_size <= len(index):
        train_lo, train_hi = index[start], index[start + train_size - 1]
        test_lo = index[start + train_size]
        test_hi = index[start + train_size + test_size - 1]

        train_data = handler.slice(train_lo, train_hi)
        best_params, best_score = combos[0], float("-inf")
        for params in combos:
            score = run(train_data, params).metrics[objective]
            if score > best_score:
                best_params, best_score = params, score

        warm_lo = index[max(0, start + train_size - warmup)]
        result = run(handler.slice(warm_lo, test_hi), best_params, start_trading=test_lo)
        folds.append(Fold(train_lo, train_hi, test_lo, test_hi, best_params, best_score, result.metrics))
        pieces.append(result.returns)
        start += step

    if not folds:
        raise ValueError(
            f"not enough data: need at least {train_size + test_size} bars, have {len(index)}"
        )
    oos = pd.concat(pieces)
    oos = oos[~oos.index.duplicated(keep="first")]  # only possible if step < test_size
    return WalkForwardResult(folds, oos.rename("returns"), initial_cash, periods_per_year)
