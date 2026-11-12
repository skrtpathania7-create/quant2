"""Performance metrics computed from an equity curve and the trade log.

All annualized figures assume ``periods_per_year`` bars per year (252 for
daily data).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def total_return(equity: pd.Series) -> float:
    return float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) > 1 else 0.0


def cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    years = (len(equity) - 1) / periods_per_year
    if years <= 0 or equity.iloc[0] <= 0 or equity.iloc[-1] <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def annual_volatility(rets: pd.Series, periods_per_year: int = 252) -> float:
    if len(rets) < 2:
        return 0.0
    return float(rets.std(ddof=1) * math.sqrt(periods_per_year))


def sharpe_ratio(rets: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """Annualized mean excess return over its standard deviation."""
    if len(rets) < 2:
        return 0.0
    excess = rets - risk_free_rate / periods_per_year
    sd = excess.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return 0.0
    return float(excess.mean() / sd * math.sqrt(periods_per_year))


def sortino_ratio(rets: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """Like Sharpe, but only penalizes downside deviation."""
    if len(rets) < 2:
        return 0.0
    excess = rets - risk_free_rate / periods_per_year
    downside = math.sqrt(float((np.minimum(excess, 0.0) ** 2).mean()))
    if downside == 0:
        return 0.0
    return float(excess.mean() / downside * math.sqrt(periods_per_year))


def drawdown(equity: pd.Series) -> pd.Series:
    """Fractional distance below the running peak (0 at a new high, negative below)."""
    return equity / equity.cummax() - 1


def max_drawdown(equity: pd.Series) -> float:
    return float(drawdown(equity).min()) if len(equity) else 0.0


def max_drawdown_duration(equity: pd.Series) -> int:
    """Longest run of bars spent below a previous peak."""
    longest = current = 0
    for underwater in (drawdown(equity) < 0):
        current = current + 1 if underwater else 0
        longest = max(longest, current)
    return longest


def summarize(
    equity: pd.Series,
    fills: pd.DataFrame | None = None,
    trades: pd.DataFrame | None = None,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    rets = returns(equity)
    mdd = max_drawdown(equity)
    growth = cagr(equity, periods_per_year)
    out: dict[str, float] = {
        "total_return": total_return(equity),
        "cagr": growth,
        "annual_volatility": annual_volatility(rets, periods_per_year),
        "sharpe": sharpe_ratio(rets, risk_free_rate, periods_per_year),
        "sortino": sortino_ratio(rets, risk_free_rate, periods_per_year),
        "max_drawdown": mdd,
        "max_drawdown_duration": max_drawdown_duration(equity),
        "calmar": growth / abs(mdd) if mdd < 0 else 0.0,
    }

    if fills is not None and len(fills):
        traded = (fills["quantity"].abs() * fills["price"]).sum()
        years = max(len(equity) - 1, 1) / periods_per_year
        out["num_fills"] = len(fills)
        out["total_commission"] = float(fills["commission"].sum())
        out["total_slippage"] = float(fills["slippage"].sum())
        out["annual_turnover"] = float(traded / equity.mean() / years)
    else:
        out.update(num_fills=0, total_commission=0.0, total_slippage=0.0, annual_turnover=0.0)

    if trades is not None and len(trades):
        pnl = trades["pnl"]
        losses = -pnl[pnl < 0].sum()
        out["num_trades"] = len(trades)
        out["win_rate"] = float((pnl > 0).mean())
        out["profit_factor"] = float(pnl[pnl > 0].sum() / losses) if losses > 0 else float("inf")
    else:
        out.update(num_trades=0, win_rate=0.0, profit_factor=0.0)
    return out


_PERCENT = {"total_return", "cagr", "annual_volatility", "max_drawdown", "win_rate", "annual_turnover"}
_MONEY = {"total_commission", "total_slippage"}
_INT = {"max_drawdown_duration", "num_fills", "num_trades"}


def format_metric(name: str, value: float) -> str:
    if name in _PERCENT:
        return f"{value:.2%}"
    if name in _MONEY:
        return f"${value:,.2f}"
    if name in _INT:
        return f"{int(value):,}"
    return f"{value:.2f}"
