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

   # create price data with clear regimes
   base_price = 100
   prices = []

   # simulate different market regimes
   prices.extend(base_price * np.exp(np.random.normal(0.001, 0.005, 30)).cumprod())
   prices.extend(prices[-1] * np.exp(np.random.normal(-0.001, 0.008, 40)).cumprod())
   prices.extend(prices[-1] * np.exp(np.random.normal(0, 0.002, 30)).cumprod())

   df = pd.DataFrame(index=dates)
   df['close'] = prices
   df['high'] = df['close'] * (1 + np.random.uniform(0, 0.02, len(prices)))
   df['low'] = df['close'] * (1 - np.random.uniform(0, 0.02, len(prices)))
   df['open'] = df['close'].shift(1).fillna(df['close'])
   df['volume'] = np.random.randint(1000, 10000, 100)
   return df

def test_backtest_strategy(sample_data):
   engineer = FeatureEngineer()
   features = engineer.calculate_features(sample_data)

   model = MarketRegimeHMM(n_states=3)
   model.fit(features)

   strategy = Strategy(initial_capital=10000)
   results = strategy.backtest(model, sample_data, features)

   assert isinstance(results, dict)
   assert 'total_return' in results
   assert 'sharpe_ratio' in results
   assert 'max_drawdown' in results
   assert 'n_trades' in results
   assert results['n_trades'] > 0

def test_portfolio_tracking(sample_data):
    engineer = FeatureEngineer()
    features = engineer.calculate_features(sample_data)

    model = MarketRegimeHMM(n_states=3)
    model.fit(features)

    strategy = Strategy(initial_capital=10000)
    strategy.backtest(model, sample_data, features)

    # check against features length instead of data length
    assert len(strategy.portfolio_value) == len(features)
    assert strategy.portfolio_value[0] == 10000
    assert all(v >= 0 for v in strategy.portfolio_value)
