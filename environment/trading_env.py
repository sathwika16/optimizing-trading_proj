"""A long-only Gymnasium trading environment."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - import error is reported on construction
    gym = None
    spaces = None


class TradingEnv(gym.Env if gym is not None else object):
    """Buy, sell, or hold one unit-fraction position with costs and slippage."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        data: pd.DataFrame,
        feature_columns: list[str],
        initial_cash: float = 100_000.0,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
    ) -> None:
        if gym is None or spaces is None:
            raise ImportError("Install gymnasium to use TradingEnv.")
        required = {"Close", *feature_columns}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"Trading data is missing columns: {sorted(missing)}")
        if len(data) < 2:
            raise ValueError("TradingEnv requires at least two rows.")
        self.data = data.reset_index(drop=True).copy()
        self.feature_columns = feature_columns
        self.initial_cash = float(initial_cash)
        self.transaction_cost = float(transaction_cost)
        self.slippage = float(slippage)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(feature_columns) + 2,), dtype=np.float32
        )
        self.current_step = 0
        self.cash = self.initial_cash
        self.holdings = 0.0
        self.portfolio_value = self.initial_cash
        self.peak_value = self.initial_cash

    def _price(self) -> float:
        return float(self.data.iloc[self.current_step]["Close"])

    def _observation(self) -> np.ndarray:
        features = self.data.loc[self.current_step, self.feature_columns].to_numpy(dtype=np.float32)
        return np.concatenate(
            [features, np.array([self.cash / self.initial_cash, self.holdings], dtype=np.float32)]
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = self.initial_cash
        self.holdings = 0.0
        self.portfolio_value = self.initial_cash
        self.peak_value = self.initial_cash
        return self._observation(), {}

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        price = self._price()
        previous_value = self.cash + self.holdings * price
        if action == 0:  # Buy: invest all available cash.
            amount = self.cash / (price * (1 + self.transaction_cost + self.slippage))
            self.holdings += amount
            self.cash -= amount * price * (1 + self.transaction_cost + self.slippage)
        elif action == 1:  # Sell: liquidate the position.
            proceeds = self.holdings * price * (1 - self.transaction_cost - self.slippage)
            self.cash += proceeds
            self.holdings = 0.0

        self.current_step += 1
        terminated = self.current_step >= len(self.data) - 1
        next_price = self._price()
        self.portfolio_value = self.cash + self.holdings * next_price
        self.peak_value = max(self.peak_value, self.portfolio_value)
        portfolio_return = self.portfolio_value / max(previous_value, 1e-12) - 1
        drawdown = self.portfolio_value / max(self.peak_value, 1e-12) - 1
        reward = float(np.log1p(max(portfolio_return, -0.999999)) + 0.1 * drawdown)
        return self._observation(), reward, terminated, False, {
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "holdings": self.holdings,
            "action": int(action),
        }
