"""Financial performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_metrics(equity: pd.Series, periods_per_year: int = 252) -> dict[str, float]:
    """Calculate return and risk metrics from an equity curve."""
    equity = equity.dropna().astype(float)
    returns = equity.pct_change().dropna()
    if len(equity) < 2 or returns.empty:
        return {key: 0.0 for key in ["cumulative_return", "cagr", "sharpe", "sortino",
                                     "max_drawdown", "volatility", "win_rate", "profit_factor"]}
    years = max(len(returns) / periods_per_year, 1 / periods_per_year)
    cumulative = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    volatility = returns.std(ddof=1) * np.sqrt(periods_per_year)
    sharpe = returns.mean() / returns.std(ddof=1) * np.sqrt(periods_per_year) if returns.std(ddof=1) else 0.0
    downside = returns[returns < 0].std(ddof=1)
    sortino = returns.mean() / downside * np.sqrt(periods_per_year) if downside and not np.isnan(downside) else 0.0
    drawdown = equity / equity.cummax() - 1
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    return {
        "cumulative_return": float(cumulative),
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(drawdown.min()),
        "volatility": float(volatility),
        "win_rate": float((returns > 0).mean()),
        "profit_factor": float(gains / losses) if losses else float("inf") if gains else 0.0,
    }

