# tests/integration/test_full_pipeline.py

import pytest
from src.data_collection import BinanceDataCollector
from src.features import FeatureEngineer
from src.model import MarketRegimeHMM
from src.backtesting import Strategy
from src.visualization import DashboardGenerator

def test_full_pipeline():
    collector = BinanceDataCollector()
    tokens = collector.get_small_cap_symbols()
    assert len(tokens) > 0

    data = collector.fetch_historical_data(
        tokens[0]['symbol'],
        interval='5m',
        limit=1000
    )
    assert not data.empty

    engineer = FeatureEngineer()
    features = engineer.calculate_features(data)
    assert not features.empty

    # Align data with features
    data = data.loc[features.index]  # key fix

    model = MarketRegimeHMM()
    model.fit(features)
    states = model.predict_states(features)
    assert len(states) == len(features)

    strategy = Strategy(10000)
    results = strategy.backtest(model, data, features)
    assert 'total_return' in results

    dashboard = DashboardGenerator()
    fig = dashboard.generate_dashboard(data, states, strategy.portfolio_value, results)
    assert 'Trading Performance Dashboard' in str(fig.layout)
