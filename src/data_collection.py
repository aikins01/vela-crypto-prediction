from binance.client import Client
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
import numpy as np
from typing import Optional

load_dotenv()

class BinanceDataCollector:
    def __init__(self):
        self.client = Client(None, None)
        # got them from the binance docs
        self.valid_intervals = [
            '1m', '3m', '5m', '15m', '30m',
            '1h', '2h', '4h', '6h', '8h', '12h',
            '1d', '3d', '1w', '1M'
        ]

    def get_small_cap_symbols(self, max_market_cap: float = 100_000_000) -> list[str]:
        tickers = self.client.get_ticker()
        trading_info = self.client.get_ticker()
        small_caps = []

        for ticker in tickers:
            symbol = ticker['symbol']
            if not symbol.endswith('USDT'):
                continue

            try:
                volume_24h = float(ticker['quoteVolume'])
                # need to filter out low liquidity, too easy to manipulate
                if volume_24h < 1_000_000:
                    continue

                market_info = self.client.get_ticker(symbol=symbol)
                if float(market_info.get('marketCap', float('inf'))) <= max_market_cap:
                    small_caps.append(symbol)
            except (KeyError, ValueError):
                # better skip this one, data looks weird
                continue

        return small_caps

    def fetch_historical_data(
        self,
        symbol: str,
        interval: str = '15m',
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 500
    ) -> pd.DataFrame:
        if interval not in self.valid_intervals:
            raise ValueError(f"Invalid interval. Must be one of {self.valid_intervals}")

        if limit > 1500:
            raise ValueError("Limit cannot exceed 1500 candlesticks")

        klines = self.client.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str=start_time,
            end_str=end_time,
            limit=limit
        )

        if not klines:
            return pd.DataFrame()

        df = pd.DataFrame(data=klines)
        # need to make these columns readable...
        df = df.rename(columns={
            0: 'open_time', 1: 'open', 2: 'high', 3: 'low', 4: 'close',
            5: 'volume', 6: 'close_time', 7: 'quote_volume', 8: 'trades',
            9: 'taker_buy_volume', 10: 'taker_buy_quote_volume', 11: 'ignored'
        })

        df['open_time'] = pd.to_datetime(df['open_time'].astype(float), unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'].astype(float), unit='ms')

        numeric_cols = ['open', 'high', 'low', 'close', 'volume',
                       'quote_volume', 'taker_buy_volume', 'taker_buy_quote_volume']
        df[numeric_cols] = df[numeric_cols].astype(float)
        df['trades'] = df['trades'].astype(int)

        return df.set_index('open_time')
