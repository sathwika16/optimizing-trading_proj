"""Command-line entry point for the trading research pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtesting.engine import aggregate_equity_curves, backtest, walk_forward_splits
from agents.ppo_agent import predict_actions, train_ppo
from benchmarks.arima import arima_signals
from benchmarks.lstm import lstm_signals
from config import Config
from data.collector import fetch_market_data, single_ticker
from environment.trading_env import TradingEnv
from evaluation.metrics import calculate_metrics
from features.indicators import FeatureScaler, add_technical_indicators, chronological_split
from visualization.plots import performance_dashboard, plot_signals


def _run_ticker(
    raw: pd.DataFrame, ticker: str, config: Config
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
    list[dict[str, float]],
]:
    """Run all strategies for one ticker using one shared test period."""
    data = add_technical_indicators(single_ticker(raw, ticker))
    train, validation, test = chronological_split(
        data, config.train_ratio, config.validation_ratio
    )
    features = ["Return", "SMA_20", "EMA_20", "RSI_14", "MACD", "MACD_signal",
                "Volatility_20", "HL_range"]
    scaler = FeatureScaler().fit(train, features)
    test_prices = test["Close"]
    results: dict[str, pd.DataFrame] = {}
    buy_hold = backtest(
        test_prices,
        pd.Series(0, index=test_prices.index, dtype=int),
        initial_cash=config.initial_cash,
        transaction_cost=config.transaction_cost,
        slippage=config.slippage,
    )
    results["buy_and_hold"] = buy_hold

    validation_results: dict[str, dict[str, float]] = {}
    validation_results["buy_and_hold"] = calculate_metrics(
        backtest(
            validation["Close"],
            pd.Series(0, index=validation.index, dtype=int),
            config.initial_cash,
            config.transaction_cost,
            config.slippage,
        )["equity"]
    )
    arima_validation = {
        order: backtest(
            validation["Close"],
            arima_signals(train["Close"], validation["Close"], order),
            config.initial_cash,
            config.transaction_cost,
            config.slippage,
        )
        for order in config.arima_orders
    }
    best_order = max(
        arima_validation,
        key=lambda order: calculate_metrics(arima_validation[order]["equity"])["sharpe"],
    )
    validation_results["arima"] = calculate_metrics(arima_validation[best_order]["equity"])
    arima = arima_signals(train["Close"], test_prices, best_order)
    results["arima"] = backtest(
        test_prices, arima, config.initial_cash, config.transaction_cost, config.slippage
    )
    lstm = lstm_signals(
        train["Close"], test_prices, config.lstm_lookback, config.lstm_epochs, config.seed
    )
    results["lstm"] = backtest(
        test_prices, lstm, config.initial_cash, config.transaction_cost, config.slippage
    )

    train_scaled = scaler.transform(train, features)
    test_scaled = scaler.transform(test, features)
    environment_kwargs = {
        "feature_columns": features,
        "initial_cash": config.initial_cash,
        "transaction_cost": config.transaction_cost,
        "slippage": config.slippage,
    }
    training_env = TradingEnv(train_scaled, **environment_kwargs)
    model_path = config.output_dir / f"{ticker}_ppo_trading_model"
    model = train_ppo(
        training_env, config.ppo_timesteps, config.seed, model_path, config.ppo_learning_rate
    )
    evaluation_env = TradingEnv(test_scaled, **environment_kwargs)
    actions = predict_actions(model, evaluation_env)
    ppo_signals = pd.Series(actions, index=test_prices.index[:len(actions)], dtype=int)
    results["ppo"] = backtest(
        test_prices.loc[ppo_signals.index],
        ppo_signals,
        config.initial_cash,
        config.transaction_cost,
        config.slippage,
    )
    metrics = {name: calculate_metrics(result["equity"]) for name, result in results.items()}
    walk_forward_metrics: list[dict[str, float]] = []
    fold_size = max(10, len(data) // 10)
    for fold_train, fold_test in walk_forward_splits(
        data, max(20, len(data) // 2), fold_size
    ):
        fold_signals = arima_signals(fold_train["Close"], fold_test["Close"], best_order)
        fold_curve = backtest(
            fold_test["Close"],
            fold_signals,
            config.initial_cash,
            config.transaction_cost,
            config.slippage,
        )
        walk_forward_metrics.append(calculate_metrics(fold_curve["equity"]))
    return results, metrics, validation_results, walk_forward_metrics


def run(config: Config) -> dict[str, dict[str, float]]:
    """Run the full research pipeline independently for every requested ticker."""
    np.random.seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    raw = fetch_market_data(config.tickers, config.start, config.end)
    all_results: dict[str, pd.DataFrame] = {}
    metrics: dict[str, dict[str, float]] = {}
    validation_metrics: dict[str, dict[str, float]] = {}
    walk_forward_metrics: dict[str, list[dict[str, float]]] = {}
    for ticker in config.tickers:
        ticker_results, ticker_metrics, ticker_validation, ticker_walk_forward = _run_ticker(
            raw, ticker, config
        )
        for strategy, result in ticker_results.items():
            key = f"{ticker}_{strategy}"
            all_results[key] = result
            metrics[key] = ticker_metrics[strategy]
            if strategy != "buy_and_hold":
                plot_signals(
                    result["price"] if "price" in result else
                    single_ticker(raw, ticker).loc[result.index, "Close"],
                    result["signal"],
                    config.output_dir / f"{ticker}_{strategy}_signals.png",
                )
        for strategy, strategy_metrics in ticker_validation.items():
            validation_metrics[f"{ticker}_{strategy}"] = strategy_metrics
        walk_forward_metrics[ticker] = ticker_walk_forward
    strategies = set()
    for ticker in config.tickers:
        prefix = f"{ticker}_"
        strategies.update(
            key[len(prefix):] for key in metrics if key.startswith(prefix)
        )
    for strategy in strategies:
        ticker_curves = {
            key[: -len(f"_{strategy}")]: result
            for key, result in all_results.items()
            if key.endswith(f"_{strategy}")
        }
        if ticker_curves:
            portfolio = aggregate_equity_curves(ticker_curves)
            all_results[f"portfolio_{strategy}"] = portfolio
            metrics[f"portfolio_{strategy}"] = calculate_metrics(portfolio["equity"])
    performance_dashboard(all_results, config.output_dir)
    (config.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (config.output_dir / "validation_metrics.json").write_text(
        json.dumps(validation_metrics, indent=2), encoding="utf-8"
    )
    (config.output_dir / "walk_forward_metrics.json").write_text(
        json.dumps(walk_forward_metrics, indent=2), encoding="utf-8"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=["AAPL"])
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--ppo-timesteps", type=int, default=20_000)
    parser.add_argument("--lstm-lookback", type=int, default=20)
    parser.add_argument("--lstm-epochs", type=int, default=10)
    args = parser.parse_args()
    metrics = run(Config(tickers=args.tickers, start=args.start, end=args.end,
                         output_dir=Path(args.output_dir),
                         ppo_timesteps=args.ppo_timesteps,
                         lstm_lookback=args.lstm_lookback,
                         lstm_epochs=args.lstm_epochs))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
