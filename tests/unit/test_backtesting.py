import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.backtesting import Strategy
from src.model import MarketRegimeHMM
from src.features import FeatureEngineer

@pytest.fixture
def sample_data():
    dates = pd.date_range(start='2024-01-01', periods=100, freq='15min')
    base_price = 100

    # simulate price sequences
    prices = []
    prices.extend(base_price * np.exp(np.random.normal(0.001, 0.005, 30)).cumprod())
    prices.extend(prices[-1] * np.exp(np.random.normal(-0.001, 0.008, 40)).cumprod())
    prices.extend(prices[-1] * np.exp(np.random.normal(0, 0.002, 30)).cumprod())

    df = pd.DataFrame(index=dates)
    df['close'] = prices
    df['high'] = df['close'] * (1 + np.random.uniform(0, 0.02, len(prices)))
    df['low'] = df['close'] * (1 - np.random.uniform(0, 0.02, len(prices)))
    df['open'] = df['close'].shift(1).fillna(df['close'])
    df['volume'] = np.random.randint(1000, 10000, len(prices))

    return df

def test_backtest_strategy(sample_data):
    engineer = FeatureEngineer()
    features = engineer.calculate_features(sample_data)

    model = MarketRegimeHMM()
    model.fit(features)

    strategy = Strategy(initial_capital=10000)
    metrics = strategy.backtest(model, sample_data.loc[features.index], features)

    # check for metrics that match the source implementation
    assert isinstance(metrics, dict)
    expected_metrics = [
        'model_return', 'hold_return',
        'model_sharpe', 'hold_sharpe',
        'model_drawdown', 'hold_drawdown',
        'n_trades', 'stop_losses'
    ]
    assert all(metric in metrics for metric in expected_metrics)

def test_portfolio_tracking(sample_data):
    engineer = FeatureEngineer()
    features = engineer.calculate_features(sample_data)

    model = MarketRegimeHMM()
    model.fit(features)

    strategy = Strategy(initial_capital=10000)
    _ = strategy.backtest(model, sample_data.loc[features.index], features)

    assert len(strategy.portfolio_value) == len(features)
    assert strategy.portfolio_value[0] == 10000  # initial capital
    assert all(v >= 0 for v in strategy.portfolio_value)
