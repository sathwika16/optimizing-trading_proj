# Optimizing Trading Strategy using Deep Reinforcement Learning

Modular quantitative research project for comparing PPO, ARIMA, LSTM, and buy-and-hold strategies. The project uses chronological splits and fits feature scaling on training data only to prevent look-ahead leakage.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

TensorFlow and Stable-Baselines3 are only needed for their respective models, but are included for a complete environment.

## Run

```powershell
python main.py --tickers AAPL MSFT --start 2018-01-01 --end 2025-01-01 --output-dir outputs
```

Training settings can be overridden with `--ppo-timesteps`, `--lstm-lookback`, and
`--lstm-epochs`. Use a smaller PPO timestep count for a quick smoke run, not for
final research.

The pipeline downloads OHLCV data for every requested ticker, computes indicators, creates leakage-safe splits, trains and evaluates every strategy independently, writes per-ticker metrics to `outputs/metrics.json`, and saves dashboard charts. Network data is never replaced with fabricated results. Multiple tickers intentionally produce separate portfolios; they are not silently combined into one portfolio.

Additional outputs include `validation_metrics.json`, ARIMA walk-forward metrics, per-strategy signal charts, and an equal-weighted `portfolio_*` result for each strategy.

## Tests

Run the dependency-light core tests with:

```powershell
python -m unittest discover -s tests -v
```

## Modules

- `data/`: yfinance collection and ticker extraction
- `features/`: SMA, EMA, RSI, MACD, volatility, scaling, and chronological splits
- `environment/`: Gymnasium trading environment with cash, holdings, costs, and slippage
- `agents/`: Stable-Baselines3 PPO training and prediction
- `benchmarks/`: ARIMA and TensorFlow LSTM signal generation
- `backtesting/`: shared execution engine and walk-forward splits
- `evaluation/`: CAGR, Sharpe, Sortino, drawdown, volatility, win rate, and profit factor
- `visualization/`: equity, drawdown, return, and signal charts

## Research safeguards

All model outputs should be generated from the training partition and evaluated on the untouched test partition. Use `walk_forward_splits` for repeated expanding-window validation. Transaction costs and slippage are applied consistently by `backtest`; results are not a guarantee of future performance.
