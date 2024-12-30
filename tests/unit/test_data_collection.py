import pytest
from src.data_collection import BinanceDataCollector
import pandas as pd

def test_binance_data_collector():
    collector = BinanceDataCollector()

    # test with valid parameters
    data = collector.fetch_historical_data(
        symbol='BTCUSDT',
        interval='15m',
        limit=100
    )

    # check structure
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert len(data) <= 100  # check limit

    # check columns
    expected_columns = [
        'high', 'low', 'open', 'close', 'volume',
        'close_time', 'quote_volume', 'trades',
        'taker_buy_volume', 'taker_buy_quote_volume', 'ignored'
    ]
    assert all(col in data.columns for col in expected_columns)

    # check data types
    assert isinstance(data.index, pd.DatetimeIndex)
    assert data['volume'].dtype == float
    assert data['trades'].dtype == int

def test_get_small_cap_symbols():
    collector = BinanceDataCollector()
    symbols = collector.get_small_cap_symbols()

    assert isinstance(symbols, list)
    if symbols:  # if we found any matching tokens
        first_token = symbols[0]
        assert isinstance(first_token, dict)
        assert 'symbol' in first_token
        assert 'market_cap' in first_token
        assert 'volume_24h' in first_token
        assert 'days_listed' in first_token
        assert first_token['market_cap'] <= 100_000_000
        assert first_token['volume_24h'] >= 1_000_000
        assert first_token['days_listed'] <= 90

def test_invalid_interval():
    collector = BinanceDataCollector()
    with pytest.raises(ValueError, match="Invalid interval"):
        collector.fetch_historical_data('BTCUSDT', interval='invalid')

def test_limit_validation():
    collector = BinanceDataCollector()
    with pytest.raises(ValueError, match="Limit cannot exceed 1500"):
        collector.fetch_historical_data('BTCUSDT', limit=2000)
