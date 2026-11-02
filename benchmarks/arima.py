"""ARIMA forecasting and signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def arima_signals(
    train: pd.Series,
    prices: pd.Series,
    order: tuple[int, int, int] = (1, 1, 1),
) -> pd.Series:
    """Fit expanding-window ARIMA forecasts and emit 0=buy, 1=sell, 2=hold."""
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError as exc:
        raise ImportError("Install statsmodels to run ARIMA.") from exc
    history = list(train.astype(float))
    signals = pd.Series(2, index=prices.index, dtype=int)
    for index, price in prices.items():
        if len(history) >= max(10, sum(order) + 2):
            forecast = float(ARIMA(history, order=order).fit().forecast(1)[0])
            signals.loc[index] = 0 if forecast > price else 1 if forecast < price else 2
        history.append(float(price))
    return signals

