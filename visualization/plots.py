"""Matplotlib visualizations for research outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from evaluation.metrics import calculate_metrics


def performance_dashboard(results: dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    """Save equity, drawdown, and return comparison charts."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    for name, result in results.items():
        equity = result["equity"]
        axes[0].plot(equity.index, equity / equity.iloc[0], label=name)
        axes[1].plot(equity.index, equity / equity.cummax() - 1, label=name)
        axes[2].plot(result.index, result["returns"].cumsum(), label=name)
    axes[0].set_title("Normalized portfolio equity")
    axes[1].set_title("Drawdown")
    axes[2].set_title("Cumulative daily returns")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "performance_dashboard.png", dpi=150)
    plt.close(fig)
    sharpes = {name: calculate_metrics(result["equity"])["sharpe"]
               for name, result in results.items()}
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.bar(sharpes.keys(), sharpes.values())
    axis.set_title("Sharpe ratio comparison")
    axis.set_ylabel("Sharpe ratio")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "sharpe_comparison.png", dpi=150)
    plt.close(fig)


def plot_signals(prices: pd.Series, signals: pd.Series, output_path: str | Path) -> None:
    """Save price chart annotated with buy and sell signals."""
    fig, axis = plt.subplots(figsize=(12, 5))
    axis.plot(prices.index, prices, label="Close")
    buys = signals.reindex(prices.index) == 0
    sells = signals.reindex(prices.index) == 1
    axis.scatter(prices.index[buys], prices[buys], marker="^", color="green", label="Buy")
    axis.scatter(prices.index[sells], prices[sells], marker="v", color="red", label="Sell")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
