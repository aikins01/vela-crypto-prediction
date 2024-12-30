import pytest
import pandas as pd
from datetime import datetime, timedelta
from src.data_collection import BinanceDataCollector
from src.features import FeatureEngineer
from src.model import MarketRegimeHMM
from src.backtesting import Strategy
from src.evaluation import ModelEvaluator

def test_small_cap_pipeline():
    # 1. Find recently listed small-cap tokens
    collector = BinanceDataCollector()
    tokens = collector.get_small_cap_symbols(
        max_market_cap=100_000_000,
        max_days=90
    )

    print(f"\nFound {len(tokens)} matching tokens")
    if tokens:
        for t in tokens[:3]:
            print(f"Symbol: {t['symbol']}")
            print(f"Market Cap: ${t['market_cap']:,.0f}")
            print(f"Days Listed: {t['days_listed']}")
            print(f"24h Volume: ${t['volume_24h']:,.0f}\n")

    assert len(tokens) > 0
    for token in tokens:
        assert token['market_cap'] <= 100_000_000
        assert token['days_listed'] <= 90
        assert token['volume_24h'] >= 1_000_000

    # 2. Test data collection
    if tokens:
        token = tokens[0]
        data = collector.fetch_historical_data(
            symbol=token['symbol'],
            interval='5m',  # use available interval
            limit=1000
        )

        assert not data.empty
        assert all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume'])

        # 3. Feature engineering
        engineer = FeatureEngineer()
        features = engineer.calculate_features(data)

        assert not features.empty
        assert all(col in features.columns for col in ['returns', 'hl_ratio', 'volume_ma_ratio', 'rsi', 'volatility'])

        # 4. Model evaluation
        evaluator = ModelEvaluator()
        results = evaluator.cross_validate(data, n_splits=3)

        assert len(results) == 2  # n_splits - 1 results
        for result in results:
            assert 'total_return' in result
            assert 'sharpe_ratio' in result
            assert 'max_drawdown' in result
            assert 'n_trades' in result
