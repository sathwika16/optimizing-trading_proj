"""Historical market data collection."""

from __future__ import annotations

from typing import Optional

import pandas as pd


def fetch_market_data(
    tickers: str | list[str],
    start: str,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch OHLCV data from Yahoo Finance with a consistent column layout."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("Install yfinance with `pip install yfinance`.") from exc

    symbols = [tickers] if isinstance(tickers, str) else tickers
    data = yf.download(
        symbols,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="column",
    )
    if data.empty:
        raise ValueError("Yahoo Finance returned no data for the requested range.")

    if isinstance(data.columns, pd.MultiIndex):
        ticker_level = next(
            (level for level in range(data.columns.nlevels)
             if set(symbols).intersection(data.columns.get_level_values(level))),
            None,
        )
        if ticker_level is None:
            raise ValueError("Downloaded data did not contain the requested ticker columns.")
        frames = []
        for symbol in symbols:
            if symbol not in data.columns.get_level_values(ticker_level):
                continue
            frame = data.xs(symbol, axis=1, level=ticker_level).copy()
            if isinstance(frame.columns, pd.MultiIndex):
                frame.columns = frame.columns.get_level_values(0)
            frame.columns = pd.MultiIndex.from_product([[symbol], frame.columns])
            frames.append(frame)
        if not frames:
            raise ValueError("No requested tickers were present in the downloaded data.")
        data = pd.concat(frames, axis=1)
    else:
        data.columns = pd.MultiIndex.from_product([[symbols[0]], data.columns])

    data.index = pd.to_datetime(data.index).tz_localize(None)
    return data.sort_index().dropna(how="all")


def single_ticker(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Extract one ticker from a multi-ticker frame."""
    if isinstance(data.columns, pd.MultiIndex):
        if ticker not in data.columns.get_level_values(0):
            raise KeyError(f"Ticker {ticker!r} is not in the data.")
        return data[ticker].copy()
    return data.copy()
