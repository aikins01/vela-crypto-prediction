import pytest
import pandas as pd
import numpy as np
from src.features import FeatureEngineer

@pytest.fixture
def sample_data():
   # have to make sure high is always higher than low
   dates = pd.date_range(start='2024-01-01', periods=100, freq='15min')
   base = np.random.randn(100).cumsum() + 100
   data = {
       'open': base,
       'close': base + np.random.randn(100),
       'high': base + np.abs(np.random.randn(100)) + 2,  # ensure high is highest
       'low': base - np.abs(np.random.randn(100)) - 2,   # ensure low is lowest
       'volume': np.random.randint(1000, 10000, 100)
   }
   return pd.DataFrame(data, index=dates)

def test_feature_calculation(sample_data):
   engineer = FeatureEngineer()
   features = engineer.calculate_features(sample_data)

   expected_features = ['returns', 'hl_ratio', 'volume_ma_ratio', 'rsi', 'volatility']
   assert all(col in features.columns for col in expected_features)

   assert not features.isna().values.any()
   assert features['rsi'].between(0, 100).all()
   assert (features['hl_ratio'] >= 1).all()  # this should pass now
   assert not np.isinf(features['volume_ma_ratio']).any()

def test_rsi_calculation():
   engineer = FeatureEngineer()
   prices = pd.Series([10, 11, 12, 11, 10, 9, 10, 11, 12, 13, 14, 13, 12, 11, 12])
   rsi = engineer._calculate_rsi(prices)

   assert isinstance(rsi, pd.Series)
   assert all(0 <= x <= 100 for x in rsi.dropna())

def test_volatility_calculation():
   engineer = FeatureEngineer()
   constant_prices = pd.Series([100] * 20)
   volatility = engineer._calculate_volatility(constant_prices)

   assert isinstance(volatility, pd.Series)
   assert all(vol == 0 for vol in volatility.dropna())
