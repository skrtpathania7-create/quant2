"""The output of a backtest run."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import pandas as pd

from . import metrics as m

if TYPE_CHECKING:
    from .portfolio import Portfolio

FILL_COLUMNS = ["timestamp", "order_id", "symbol", "quantity", "price", "commission", "slippage"]
TRADE_COLUMNS = ["timestamp", "symbol", "quantity", "entry_price", "exit_price", "pnl"]


@dataclass
class BacktestResult:
    equity: pd.Series
    cash: pd.Series
    positions: pd.DataFrame
    fills: pd.DataFrame
    trades: pd.DataFrame
    initial_cash: float
    periods_per_year: int = 252
    risk_free_rate: float = 0.0

    @classmethod
    def from_portfolio(cls, p: "Portfolio", periods_per_year: int = 252, risk_free_rate: float = 0.0):
        hist = p.history()
        fills = pd.DataFrame([f.__dict__ for f in p.fills], columns=FILL_COLUMNS)
        trades = pd.DataFrame(p.trades, columns=TRADE_COLUMNS)
        return cls(
            equity=hist["equity"].rename("equity"),
            cash=hist["cash"].rename("cash"),
            positions=p.position_history(),
            fills=fills,
            trades=trades,
            initial_cash=p.initial_cash,
            periods_per_year=periods_per_year,
            risk_free_rate=risk_free_rate,
        )

    @property
    def returns(self) -> pd.Series:
        """Per-bar returns, including the first bar measured against starting cash."""
        r = self.equity.pct_change()
        if len(r):
            r.iloc[0] = self.equity.iloc[0] / self.initial_cash - 1
        return r

    @cached_property
    def metrics(self) -> dict[str, float]:
        return m.summarize(
            self.equity, self.fills, self.trades, self.periods_per_year, self.risk_free_rate
        )

    def summary(self, title: str = "Backtest") -> str:
        width = max(len(k) for k in self.metrics)
        lines = [title, "-" * (width + 16)]
        lines += [f"{k:<{width}}  {m.format_metric(k, v):>14}" for k, v in self.metrics.items()]
        return "\n".join(lines)

    def plot(self, path=None, title: str = "Backtest", benchmark: pd.Series | None = None):
        """Equity curve and drawdown. Saves to ``path`` if given; returns the figure."""
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
        ax1.plot(self.equity.index, self.equity, label="Strategy", lw=1.4)
        if benchmark is not None:
            bench = benchmark.reindex(self.equity.index).ffill()
            ax1.plot(bench.index, bench / bench.iloc[0] * self.initial_cash, label="Buy & hold", lw=1, alpha=0.7)
        ax1.set_title(title)
        ax1.set_ylabel("Equity")
        ax1.legend(loc="upper left")
        ax1.grid(alpha=0.3)

        dd = m.drawdown(self.equity)
        ax2.fill_between(dd.index, dd * 100, 0, color="tab:red", alpha=0.4)
        ax2.set_ylabel("Drawdown %")
        ax2.grid(alpha=0.3)
        fig.tight_layout()
        if path is not None:
            fig.savefig(path, dpi=120)
        return fig
