"""Leakage-aware technical indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Add indicators using only current and historical observations."""
    frame = data.copy()
    close = frame["Close"].astype(float)
    high = frame["High"].astype(float)
    low = frame["Low"].astype(float)
    frame["Return"] = close.pct_change()
    frame["SMA_20"] = close.rolling(20, min_periods=20).mean()
    frame["EMA_20"] = close.ewm(span=20, adjust=False, min_periods=20).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = -delta.clip(upper=0).rolling(14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    frame["RSI_14"] = (100 - (100 / (1 + rs))).fillna(50.0)
    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    frame["MACD"] = ema12 - ema26
    frame["MACD_signal"] = frame["MACD"].ewm(
        span=9, adjust=False, min_periods=9
    ).mean()
    frame["Volatility_20"] = frame["Return"].rolling(20, min_periods=20).std()
    frame["HL_range"] = (high - low) / close.replace(0, np.nan)
    return frame.replace([np.inf, -np.inf], np.nan).dropna()


def chronological_split(
    data: pd.DataFrame, train_ratio: float, validation_ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split chronologically; no observations are shuffled across boundaries."""
    if len(data) < 3:
        raise ValueError("At least three rows are required for chronological splits.")
    train_end = int(len(data) * train_ratio)
    validation_end = train_end + int(len(data) * validation_ratio)
    return data.iloc[:train_end], data.iloc[train_end:validation_end], data.iloc[validation_end:]


class FeatureScaler:
    """Standard scaler fitted only on the training partition."""

    def __init__(self) -> None:
        self.mean_: pd.Series | None = None
        self.std_: pd.Series | None = None

    def fit(self, data: pd.DataFrame, columns: list[str]) -> "FeatureScaler":
        self.mean_ = data[columns].mean()
        self.std_ = data[columns].std().replace(0, 1.0).fillna(1.0)
        return self

    def transform(self, data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("FeatureScaler must be fitted before transform.")
        result = data.copy()
        result[columns] = (result[columns] - self.mean_) / self.std_
        return result

