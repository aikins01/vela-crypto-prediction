from binance.client import Client
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import numpy as np
from typing import Optional, List, Dict

load_dotenv()

class BinanceDataCollector:
    def __init__(self):
        self.client = Client(None, None)
        self.valid_intervals = [
            '1m', '3m', '5m', '15m', '30m',
            '1h', '2h', '4h', '6h', '8h', '12h',
            '1d', '3d', '1w', '1M'
        ]  # got them from the binance docs

    def get_small_cap_symbols(self, max_market_cap: float = 100_000_000, max_days: int = 90) -> List[Dict]:
        # get all data in one batch where possible
        exchange_info = self.client.get_exchange_info()
        all_tickers = {t['symbol']: t for t in self.client.get_ticker()}
        print(f"Total symbols found: {len(exchange_info['symbols'])}")

        # filter USDT pairs first
        usdt_pairs = []
        for symbol_info in exchange_info['symbols']:
            symbol = symbol_info['symbol']
            if not symbol.endswith('USDT') or symbol_info['status'] != 'TRADING':
                continue

            ticker = all_tickers.get(symbol, {})
            try:
                volume_24h = float(ticker.get('quoteVolume', 0))
                if volume_24h >= 1_000_000:  # check volume before klines
                    usdt_pairs.append(symbol)
            except:
                continue

        print(f"High volume USDT pairs found: {len(usdt_pairs)}")

        # check klines in smaller batches to respect rate limits
        candidates = []
        batch_size = 10  # process 10 at a time

        for i in range(0, len(usdt_pairs), batch_size):
            batch = usdt_pairs[i:i + batch_size]

            for symbol in batch:
                try:
                    klines = self.client.get_klines(
                        symbol=symbol,
                        interval='1d',
                        limit=90
                    )

                    if len(klines) < 90:
                        ticker = all_tickers[symbol]
                        market_cap = float(ticker.get('marketCap', float(ticker['quoteVolume']) * 7))

                        if market_cap <= max_market_cap:
                            candidates.append({
                                'symbol': symbol,
                                'market_cap': market_cap,
                                'volume_24h': float(ticker['quoteVolume']),
                                'days_listed': len(klines)
                            })
                            print(f"Found candidate: {symbol}")

                except Exception as e:
                    print(f"Error processing {symbol}: {str(e)}")
                    continue

        return sorted(candidates, key=lambda x: x['days_listed'])

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
