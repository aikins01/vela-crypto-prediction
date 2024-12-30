import pytest
import pandas as pd
from src.data_collection import BinanceDataCollector
from src.features import FeatureEngineer
from src.model import MarketRegimeHMM
from src.backtesting import Strategy
from src.evaluation import ModelEvaluator

def test_small_cap_pipeline():
    # 1. Find small-cap tokens
    collector = BinanceDataCollector()
    tokens = collector.get_small_cap_symbols(max_market_cap=100_000_000)

    # we need at least one token to test
    assert len(tokens) > 0
    token = tokens[0]

    # 2. Get historical data
    data = collector.fetch_historical_data(
        symbol=token['symbol'],
        interval='15m',
        limit=1000
    )

    # 3. Feature engineering
    engineer = FeatureEngineer()
    features = engineer.calculate_features(data)

    # 4. Train and evaluate
    evaluator = ModelEvaluator()
    results = evaluator.cross_validate(data, n_splits=3)

    # Check results structure
    assert len(results) == 2
    for result in results:
        assert 'total_return' in result
        assert 'sharpe_ratio' in result
        assert 'max_drawdown' in result
        assert 'n_trades' in result
