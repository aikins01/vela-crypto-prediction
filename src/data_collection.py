from datetime import datetime
import pandas as pd
from binance.client import Client
from typing import Optional, List, Dict
from tqdm import tqdm

class BinanceDataCollector:
    def __init__(self):
        self.client = Client(None, None)
        self.valid_intervals = [
            '1m', '3m', '5m', '15m', '30m',
            '1h', '2h', '4h', '6h', '8h', '12h',
            '1d', '3d', '1w', '1M'
        ]

    def get_small_cap_symbols(
        self,
        max_market_cap: float = 100_000_000,
        max_days: int = 90
    ) -> List[Dict]:
        # get all data in one batch where possible
        exchange_info = self.client.get_exchange_info()
        all_tickers = {t['symbol']: t for t in self.client.get_ticker()}
        print(f"Total symbols found: {len(exchange_info['symbols'])}")

        # filter USDT pairs first
        usdt_pairs = []
        for symbol_info in tqdm(exchange_info['symbols'], desc="Checking pairs"):
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

        # check klines in smaller batches
        candidates = []
        batch_size = 10

        # process batches with progress
        batches = list(range(0, len(usdt_pairs), batch_size))
        for i in tqdm(batches, desc="Processing batches"):
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
        interval: str = '15m'
    ) -> pd.DataFrame:
        """fetch exactly 3 months + 1 week of data for train/test using 15min intervals"""

        if interval not in self.valid_intervals:
            raise ValueError(f"Invalid interval. Must be one of {self.valid_intervals}")

        # calculate bars needed for 15min intervals
        bars_per_hour = 4  # 15min gives 4 bars per hour
        bars_per_day = bars_per_hour * 24  # 96 bars per day

        train_days = 90  # 3 months
        test_days = 7    # 1 week
        total_days = train_days + test_days

        total_bars_needed = total_days * bars_per_day

        print(f"Fetching {total_bars_needed} bars for {symbol}...")
        print(f"This will cover {total_days} days of 15min data")

        chunks = []
        bars_collected = 0

        while bars_collected < total_bars_needed:
            chunk_size = min(1500, total_bars_needed - bars_collected)
            try:
                klines = self.client.get_historical_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=chunk_size
                )

                if not klines:
                    break

                df = self._process_klines(klines)
                if len(df) == 0:
                    break

                chunks.append(df)
                bars_collected += len(df)
                print(f"Collected {bars_collected}/{total_bars_needed} bars")

                if bars_collected >= total_bars_needed:
                    break

            except Exception as e:
                print(f"Error fetching data: {str(e)}")
                break

        if not chunks:
            return pd.DataFrame()

        result = pd.concat(chunks[::-1])  # reverse to get chronological order
        print(f"Total bars collected: {len(result)}")

        return result

    def _process_klines(self, klines: List) -> pd.DataFrame:
        """process raw kline data into dataframe"""
        with tqdm(total=4, desc="Processing data") as pbar:
            df = pd.DataFrame(data=klines)
            df = df.rename(columns={
                0: 'open_time', 1: 'open', 2: 'high', 3: 'low', 4: 'close',
                5: 'volume', 6: 'close_time', 7: 'quote_volume', 8: 'trades',
                9: 'taker_buy_volume', 10: 'taker_buy_quote_volume', 11: 'ignored'
            })
            pbar.update(1)

            df['open_time'] = pd.to_datetime(df['open_time'].astype(float), unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'].astype(float), unit='ms')
            pbar.update(1)

            numeric_cols = ['open', 'high', 'low', 'close', 'volume',
                           'quote_volume', 'taker_buy_volume', 'taker_buy_quote_volume']
            df[numeric_cols] = df[numeric_cols].astype(float)
            pbar.update(1)

            df['trades'] = df['trades'].astype(int)
            df = df.set_index('open_time')
            pbar.update(1)

            return df
