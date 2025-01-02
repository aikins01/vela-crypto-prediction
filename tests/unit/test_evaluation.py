import pytest
import pandas as pd
import numpy as np
from src.evaluation import ModelEvaluator

@pytest.fixture
def sample_data():
    # create test data with multiple regimes
    dates = pd.date_range(start='2024-01-01', periods=500, freq='15min')
    base_price = 100

    # simulate price sequences
    prices = []
    regimes = [
        (0.001, 0.005, 100),  # bull
        (-0.001, 0.008, 200), # bear
        (0, 0.002, 100),      # neutral
        (0.002, 0.004, 100)   # bull
    ]

    for mu, sigma, length in regimes:
        if prices:
            start = prices[-1]
        else:
            start = base_price
        regime_prices = start * np.exp(np.random.normal(mu, sigma, length)).cumprod()
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

    # check results structure
    assert len(results) == 2
    for result in results:
        assert all(k in result for k in ['model_return', 'model_sharpe', 'model_drawdown', 'n_trades'])

def test_evaluate_split(sample_data):
    evaluator = ModelEvaluator(initial_capital=10000)

    # ensure we have enough data for meaningful testing
    if len(sample_data) < 100:
        pytest.skip("Not enough data for meaningful testing")

    # make train/test split
    split_point = len(sample_data) * 2 // 3  # use 2/3 for training
    train = sample_data.iloc[:split_point]
    test = sample_data.iloc[split_point:]

    result = evaluator.evaluate_split(train, test)

    assert isinstance(result, dict)
    assert all(key in result for key in [
        'model_return', 'hold_return',
        'model_sharpe', 'hold_sharpe',
        'model_drawdown', 'hold_drawdown',
        'n_trades', 'stop_losses'
    ])

    # check that metrics are calculated
    assert isinstance(result['model_return'], (float, np.float64))
    assert isinstance(result['n_trades'], (int, np.integer))

    # relaxed return check - we care that it calculated something
    if result['n_trades'] > 0:
        assert result['model_return'] != 0

def test_insufficient_data():
    # test handling of small datasets
    evaluator = ModelEvaluator(initial_capital=10000)
    small_data = pd.DataFrame({
        'open': [100] * 10,
        'high': [101] * 10,
        'low': [99] * 10,
        'close': [100] * 10,
        'volume': [1000] * 10
    }, index=pd.date_range(start='2024-01-01', periods=10, freq='15min'))

    with pytest.raises(ValueError):
        evaluator.cross_validate(small_data, n_splits=3)
