"""LSTM forecasting benchmark with leakage-safe training windows."""

from __future__ import annotations

import numpy as np
import pandas as pd


def lstm_signals(
    train: pd.Series,
    prices: pd.Series,
    lookback: int = 20,
    epochs: int = 10,
    seed: int = 42,
) -> pd.Series:
    """Train on the supplied history and recursively predict test prices."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError("Install tensorflow to run the LSTM benchmark.") from exc
    if len(train) <= lookback:
        raise ValueError("LSTM training history must be longer than lookback.")
    tf.keras.utils.set_random_seed(seed)
    values = train.to_numpy(dtype=np.float32)
    minimum, maximum = float(values.min()), float(values.max())
    scale = max(maximum - minimum, 1e-12)
    scaled = (values - minimum) / scale
    x = np.array([scaled[i - lookback:i] for i in range(lookback, len(scaled))])
    y = scaled[lookback:]
    model = tf.keras.Sequential(
        [tf.keras.layers.Input((lookback, 1)), tf.keras.layers.LSTM(32), tf.keras.layers.Dense(1)]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(x[..., None], y, epochs=epochs, batch_size=32, verbose=0)
    history = list(scaled)
    signals = pd.Series(2, index=prices.index, dtype=int)
    for index, price in prices.items():
        window = np.asarray(history[-lookback:], dtype=np.float32)[None, :, None]
        predicted = float(model.predict(window, verbose=0)[0, 0] * scale + minimum)
        signals.loc[index] = 0 if predicted > float(price) else 1 if predicted < float(price) else 2
        history.append((float(price) - minimum) / scale)
    return signals

