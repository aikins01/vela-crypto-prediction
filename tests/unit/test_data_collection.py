import pytest
from src.data_collection import BinanceDataCollector
import pandas as pd

def test_fetch_historical_data():
    collector = BinanceDataCollector()
    data = collector.fetch_historical_data(
        symbol='BTCUSDT',
        interval='15m'
    )

    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    required_cols = [
        'high', 'low', 'open', 'close', 'volume',
        'close_time', 'quote_volume', 'trades',
        'taker_buy_volume', 'taker_buy_quote_volume'
    ]
    assert all(col in data.columns for col in required_cols)

def test_get_small_cap_symbols():
    collector = BinanceDataCollector()
    symbols = collector.get_small_cap_symbols()

    assert isinstance(symbols, list)
    if symbols:
        first_token = symbols[0]
        assert isinstance(first_token, dict)
        required_keys = ['symbol', 'market_cap', 'volume_24h', 'days_listed']
        assert all(key in first_token for key in required_keys)
        assert first_token['market_cap'] <= 100_000_000
        assert first_token['days_listed'] <= 90

def test_invalid_interval():
    collector = BinanceDataCollector()
    with pytest.raises(ValueError):
        collector.fetch_historical_data('BTCUSDT', interval='invalid')

def test_process_klines():
    collector = BinanceDataCollector()
    klines = [
        [1704067200000, "100", "101", "99", "100.5", "1000",
         1704067500000, "100000", 50, "500", "50000", "0"]
    ]

    df = collector._process_klines(klines)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert df.index.name == 'open_time'
    assert isinstance(df.index, pd.DatetimeIndex)
