"""Consistent long-only backtesting for all strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd


def backtest(
    prices: pd.Series,
    signals: pd.Series,
    initial_cash: float = 100_000.0,
    transaction_cost: float = 0.001,
    slippage: float = 0.0005,
    position_size: float = 1.0,
) -> pd.DataFrame:
    """Apply signals at each close and mark the resulting portfolio daily."""
    if not 0 < position_size <= 1:
        raise ValueError("position_size must be in (0, 1].")
    frame = pd.DataFrame({"price": prices.astype(float), "signal": signals}).dropna()
    cash, shares = float(initial_cash), 0.0
    records: list[dict[str, float]] = []
    for timestamp, row in frame.iterrows():
        price = float(row["price"])
        signal = int(row["signal"])
        if signal == 0:
            budget = cash * position_size
            notional = budget / (1 + transaction_cost + slippage)
            bought = notional / price
            shares += bought
            cash -= notional * (1 + transaction_cost + slippage)
        elif signal == 1 and shares > 0:
            sold = shares * position_size
            cash += sold * price * (1 - transaction_cost - slippage)
            shares -= sold
        records.append({"date": timestamp, "cash": cash, "shares": shares, "price": price,
                        "equity": cash + shares * price})
    result = pd.DataFrame(records).set_index("date")
    result["returns"] = result["equity"].pct_change().fillna(0.0)
    result["signal"] = frame["signal"].to_numpy()
    return result


def walk_forward_splits(
    data: pd.DataFrame, train_size: int, test_size: int, step: int | None = None
):
    """Yield expanding-window train/test slices in chronological order."""
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be positive.")
    step = step or test_size
    for test_start in range(train_size, len(data) - test_size + 1, step):
        yield data.iloc[:test_start], data.iloc[test_start:test_start + test_size]


def aggregate_equity_curves(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create an equal-weighted portfolio curve from same-strategy ticker curves."""
    if not results:
        raise ValueError("At least one equity curve is required.")
    normalized = pd.concat(
        [result["equity"].div(result["equity"].iloc[0]).rename(name)
         for name, result in results.items()],
        axis=1,
    ).sort_index().ffill().dropna()
    equity = normalized.mean(axis=1) * results[next(iter(results))]["equity"].iloc[0]
    portfolio = pd.DataFrame({"equity": equity})
    portfolio["returns"] = portfolio["equity"].pct_change().fillna(0.0)
    portfolio["signal"] = 2
    return portfolio
