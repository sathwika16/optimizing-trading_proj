import unittest

import numpy as np
import pandas as pd

from backtesting.engine import aggregate_equity_curves, backtest, walk_forward_splits
from evaluation.metrics import calculate_metrics
from features.indicators import FeatureScaler, add_technical_indicators, chronological_split


class CoreTradingTests(unittest.TestCase):
    def setUp(self):
        index = pd.date_range("2020-01-01", periods=100, freq="D")
        close = pd.Series(100 + np.linspace(0, 10, 100), index=index)
        self.data = pd.DataFrame(
            {
                "Open": close,
                "High": close + 1,
                "Low": close - 1,
                "Close": close,
                "Volume": 1_000,
            },
            index=index,
        )

    def test_indicators_and_scaler_are_finite(self):
        features = add_technical_indicators(self.data)
        train, _, test = chronological_split(features, 0.6, 0.2)
        columns = ["Return", "SMA_20", "EMA_20", "RSI_14", "MACD", "MACD_signal",
                   "Volatility_20", "HL_range"]
        scaler = FeatureScaler().fit(train, columns)
        transformed = scaler.transform(test, columns)
        self.assertTrue(np.isfinite(transformed[columns].to_numpy()).all())

    def test_backtest_charges_costs_and_preserves_equity(self):
        prices = self.data["Close"].iloc[-20:]
        signals = pd.Series(0, index=prices.index)
        result = backtest(prices, signals, transaction_cost=0.01, slippage=0.01)
        self.assertTrue((result["equity"] > 0).all())
        self.assertLess(result["equity"].iloc[0], 100_000)

    def test_walk_forward_splits_are_chronological(self):
        splits = list(walk_forward_splits(self.data, 50, 10))
        self.assertGreater(len(splits), 0)
        for train, test in splits:
            self.assertLess(train.index[-1], test.index[0])

    def test_aggregate_equity_curves(self):
        prices = self.data["Close"].iloc[-20:]
        curves = {
            "AAA": backtest(prices, pd.Series(0, index=prices.index)),
            "BBB": backtest(prices, pd.Series(2, index=prices.index)),
        }
        portfolio = aggregate_equity_curves(curves)
        self.assertEqual(len(portfolio), len(prices))
        self.assertTrue((portfolio["equity"] > 0).all())
        self.assertIn("sharpe", calculate_metrics(portfolio["equity"]))


if __name__ == "__main__":
    unittest.main()
