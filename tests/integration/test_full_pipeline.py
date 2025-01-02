import pytest
from src.data_collection import BinanceDataCollector
from src.features import FeatureEngineer
from src.model import MarketRegimeHMM
from src.backtesting import Strategy
from src.visualization import DashboardGenerator

def test_full_pipeline():
    # collect a small amount of data to test the full pipeline
    collector = BinanceDataCollector()
    tokens = collector.get_small_cap_symbols(max_market_cap=100_000_000)
    assert len(tokens) > 0

    # fetch data for first token
    symbol = tokens[0]['symbol']
    data = collector.fetch_historical_data(
        symbol=symbol,
        interval='15m'
    )
    assert not data.empty

    # use only first week of data for faster testing
    data = data.iloc[:4 * 24 * 7]  # 7 days of 15-min bars

    # generate features
    engineer = FeatureEngineer()
    features = engineer.calculate_features(data)
    assert not features.empty

    # align data with features
    data = data.loc[features.index]

    # train model
    model = MarketRegimeHMM()
    model.fit(features)

    # run backtest using only the features timeframe
    initial_capital = 10000
    strategy = Strategy(initial_capital)
    metrics = strategy.backtest(model, data, features)

    # verify results structure
    assert isinstance(metrics, dict)
    assert all(k in metrics for k in [
        'model_return', 'hold_return',
        'model_sharpe', 'hold_sharpe',
        'model_drawdown', 'hold_drawdown',
        'n_trades', 'stop_losses'
    ])

    # verify portfolio values were tracked
    assert len(strategy.portfolio_value) == len(features)

    # allow for small deviation due to trading costs/fees
    assert abs(strategy.portfolio_value[0] - initial_capital) < initial_capital * 0.01  # 1% tolerance
    assert all(isinstance(v, (int, float)) for v in strategy.portfolio_value)
