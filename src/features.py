import pandas as pd
import numpy as np
from typing import List
from pandas import Series, DataFrame
from tqdm import tqdm

class FeatureEngineer:
    def __init__(self):
        # hmm what periods do i typically use for these indicators
        self.rsi_period = 14
        self.volatility_period = 10
        self.ma_period = 20

    def calculate_features(self, df: DataFrame) -> DataFrame:
        features = DataFrame(index=df.index)
        close_series = Series(df['close'].astype(float))
        high_series = Series(df['high'].astype(float))
        low_series = Series(df['low'].astype(float))
        volume_series = Series(df['volume'].astype(float))

        # lets calculate each feature with progress tracking
        with tqdm(total=5, desc="Calculating features") as pbar:
            # lets start with the basic price stuff
            features['returns'] = np.log(close_series / close_series.shift(1))
            pbar.update(1)

            features['hl_ratio'] = high_series / low_series
            pbar.update(1)

            # volume is important for regime detection
            features['volume_ma_ratio'] = volume_series / volume_series.rolling(self.ma_period).mean()
            pbar.update(1)

            # add some technical indicators
            features['rsi'] = self._calculate_rsi(close_series)
            pbar.update(1)

            features['volatility'] = self._calculate_volatility(close_series)
            pbar.update(1)

        return features.dropna()

    def _calculate_rsi(self, prices: Series) -> Series:
        changes = prices.diff()
        gains = changes.where(changes > 0, 0)
        losses = -changes.where(changes < 0, 0)

        avg_gains = gains.rolling(self.rsi_period).mean()
        avg_losses = losses.rolling(self.rsi_period).mean()

        rs = avg_gains / avg_losses
        return Series(100 - (100 / (1 + rs)))

    def _calculate_volatility(self, prices: Series) -> Series:
        # standard deviation of log returns, annualized
        returns = np.log(prices / prices.shift(1))
        return Series(returns.rolling(self.volatility_period).std() * np.sqrt(self.volatility_period))
