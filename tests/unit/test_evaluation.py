import pytest
import pandas as pd
import numpy as np
from src.evaluation import ModelEvaluator

@pytest.fixture
def sample_data():
   dates = pd.date_range(start='2024-01-01', periods=500, freq='15min')

   # simulate multiple market regimes
   base_price = 100
   prices = []

   # bull, bear, neutral sequences
   regimes = [
       (0.001, 0.005, 100),  # bull
       (-0.001, 0.008, 200), # bear
       (0, 0.002, 100),      # neutral
       (0.002, 0.004, 100)   # bull
   ]

   for mu, sigma, length in regimes:
       if not prices:
           start_price = base_price
       else:
           start_price = prices[-1]
       regime_prices = start_price * np.exp(np.random.normal(mu, sigma, length)).cumprod()
       prices.extend(regime_prices)

   df = pd.DataFrame(index=dates[:len(prices)])
   df['close'] = prices
   df['high'] = df['close'] * (1 + np.random.uniform(0, 0.02, len(prices)))
   df['low'] = df['close'] * (1 - np.random.uniform(0, 0.02, len(prices)))
   df['open'] = df['close'].shift(1).fillna(df['close'])
   df['volume'] = np.random.randint(1000, 10000, len(prices))
   return df

def test_cross_validation(sample_data):
   evaluator = ModelEvaluator(initial_capital=10000)
   results = evaluator.cross_validate(sample_data, n_splits=3)

   assert len(results) == 3
   for result in results:
       assert 'total_return' in result
       assert 'sharpe_ratio' in result
       assert 'max_drawdown' in result
       assert 'n_trades' in result

def test_evaluate_split(sample_data):
   evaluator = ModelEvaluator(initial_capital=10000)

   # split data in half for test
   mid_point = len(sample_data) // 2
   train_data = sample_data.iloc[:mid_point]
   test_data = sample_data.iloc[mid_point:]

   result = evaluator.evaluate_split(train_data, test_data)

   assert isinstance(result, dict)
   assert 'total_return' in result
   assert 'sharpe_ratio' in result
   assert result['total_return'] != 0  # should have some non-zero return
