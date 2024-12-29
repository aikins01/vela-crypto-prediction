import pytest
from src.data_collection import BinanceDataCollector
import pandas as pd

def test_binance_data_collector():
    # create collector instance
    collector = BinanceDataCollector()

    # test data fetch for a known pair
    data = collector.fetch_historical_data('BTCUSDT', lookback='1d')

    # basic assertions
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume'])
    assert isinstance(data.index, pd.DatetimeIndex)

if __name__ == "__main__":
    pytest.main([__file__])
