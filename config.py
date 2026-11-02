"""Configuration for the trading research pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    tickers: list[str] = field(default_factory=lambda: ["AAPL"])
    start: str = "2018-01-01"
    end: Optional[str] = None
    initial_cash: float = 100_000.0
    transaction_cost: float = 0.001
    slippage: float = 0.0005
    train_ratio: float = 0.60
    validation_ratio: float = 0.20
    seed: int = 42
    ppo_timesteps: int = 20_000
    ppo_learning_rate: float = 3e-4
    lstm_lookback: int = 20
    lstm_epochs: int = 10
    arima_orders: tuple[tuple[int, int, int], ...] = ((1, 1, 1), (2, 1, 0), (0, 1, 1))
    output_dir: Path = Path("outputs")

    def __post_init__(self) -> None:
        if not self.tickers:
            raise ValueError("At least one ticker is required.")
        if not 0 < self.train_ratio < 1:
            raise ValueError("train_ratio must be between 0 and 1.")
        if not 0 <= self.validation_ratio < 1:
            raise ValueError("validation_ratio must be between 0 and 1.")
        if self.train_ratio + self.validation_ratio >= 1:
            raise ValueError("Train and validation ratios must leave a test period.")
        if self.ppo_timesteps <= 0 or self.lstm_lookback <= 0 or self.lstm_epochs <= 0:
            raise ValueError("Model training parameters must be positive.")
        if not self.arima_orders:
            raise ValueError("At least one ARIMA order is required.")
