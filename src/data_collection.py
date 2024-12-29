from binance.client import Client
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
import numpy as np

load_dotenv()

class BinanceDataCollector:
   def __init__(self):
       self.client = Client(None, None)

   def fetch_historical_data(self, symbol: str, interval: str = '10m', lookback: str = '3m') -> pd.DataFrame:
       """fetch ohlcv data from binance"""
       # define column names first
       columns = ['timestamp', 'open', 'high', 'low', 'close',
                 'volume', 'close_time', 'quote_volume', 'trades',
                 'buy_base_volume', 'buy_quote_volume', 'ignore']

       # get data
       klines = self.client.get_historical_klines(
           symbol,
           interval,
           lookback
       )

       # create dataframe with index=[range]
       df = pd.DataFrame(data=klines, index=range(len(klines)))

       # rename columns after creation
       df.columns = columns

       # clean data
       df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
       numeric_cols = ['open', 'high', 'low', 'close', 'volume']
       df[numeric_cols] = df[numeric_cols].astype(float)

       return df.set_index('timestamp')
