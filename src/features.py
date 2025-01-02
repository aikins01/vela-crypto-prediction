import pandas as pd
import numpy as np
from typing import List
from pandas import Series, DataFrame
from tqdm import tqdm

class FeatureEngineer:
    def __init__(self):
        # indicator periods
        self.rsi_period = 21
        self.volatility_period = 21
        self.ma_period = 50
        self.adx_period = 14
        self.bb_period = 20
        self.bb_std = 2

    def calculate_features(self, df: DataFrame) -> DataFrame:
        features = DataFrame(index=df.index)

        # ensure clean input data
        close_series = Series(df['close'].astype(float))
        high_series = Series(df['high'].astype(float))
        low_series = Series(df['low'].astype(float))
        volume_series = Series(df['volume'].astype(float))

        with tqdm(total=5, desc="Calculating features") as pbar:
            # price features with nan handling
            features['returns'] = Series(np.log(close_series / close_series.shift(1))).fillna(0)
            features['hl_ratio'] = (high_series / low_series).fillna(1)
            pbar.update(1)

            # volume features
            volume_ma = volume_series.rolling(window=self.ma_period, min_periods=1).mean()
            features['volume_ma_ratio'] = Series(volume_series / volume_ma).fillna(1)
            features['volume_std'] = Series(
                volume_series.rolling(window=self.ma_period, min_periods=1).std() / volume_ma
            ).fillna(0)
            pbar.update(1)

            # technical indicators
            features['rsi'] = self._calculate_rsi(close_series)
            features['volatility'] = self._calculate_volatility(close_series)
            features['adx'] = self._calculate_adx(high_series, low_series, close_series)
            pbar.update(1)

            # trend features
            features['trend_strength'] = self._calculate_trend_strength(close_series)
            features['bb_position'] = self._calculate_bb_position(close_series)
            pbar.update(1)

            # final validation and cleaning
            features = features.ffill().bfill()  # using new pandas methods
            self._validate_features(features)
            pbar.update(1)

        return features.dropna()

    def _calculate_rsi(self, prices: Series) -> Series:
        changes = prices.diff()
        gains = changes.where(changes > 0, 0)
        losses = -changes.where(changes < 0, 0)

        avg_gains = gains.rolling(window=self.rsi_period, min_periods=1).mean()
        avg_losses = losses.rolling(window=self.rsi_period, min_periods=1).mean()

        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        return Series(rsi).fillna(50)

    def _calculate_volatility(self, prices: Series) -> Series:
        returns = Series(np.log(prices / prices.shift(1))).fillna(0)
        vol = returns.rolling(
            window=self.volatility_period,
            min_periods=1
        ).std() * np.sqrt(self.volatility_period)
        return Series(vol).ffill()  # using new pandas method

    def _calculate_adx(self, high: Series, low: Series, close: Series) -> Series:
        tr1 = Series(high - low)
        tr2 = Series(abs(high - close.shift(1)))
        tr3 = Series(abs(low - close.shift(1)))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        plus_dm = Series(high.diff())
        minus_dm = Series(low.diff())

        # smooth with minimum periods
        tr_smooth = Series(tr.rolling(window=self.adx_period, min_periods=1).mean())
        plus_dm_smooth = Series(plus_dm.rolling(window=self.adx_period, min_periods=1).mean())
        minus_dm_smooth = Series(minus_dm.rolling(window=self.adx_period, min_periods=1).mean())

        plus_di = Series(100 * (plus_dm_smooth / tr_smooth))
        minus_di = Series(100 * (minus_dm_smooth / tr_smooth))

        dx = Series(100 * abs(plus_di - minus_di) / (plus_di + minus_di))
        adx = Series(dx.rolling(window=self.adx_period, min_periods=1).mean())

        return adx.fillna(25)

    def _calculate_trend_strength(self, prices: Series) -> Series:
        prices = Series(prices)

        ma20 = Series(prices.rolling(20, min_periods=1).mean())
        ma50 = Series(prices.rolling(50, min_periods=1).mean())
        ma100 = Series(prices.rolling(100, min_periods=1).mean())

        trend = ((ma20 > ma50) & (ma50 > ma100)) | ((ma20 < ma50) & (ma50 < ma100))
        strength = Series(trend.astype(float).rolling(20, min_periods=1).mean())

        return strength.fillna(0.5)

    def _calculate_bb_position(self, prices: Series) -> Series:
        prices = Series(prices)

        ma = Series(prices.rolling(self.bb_period, min_periods=1).mean())
        std = Series(prices.rolling(self.bb_period, min_periods=1).std())

        upper = ma + self.bb_std * std
        lower = ma - self.bb_std * std

        bb_range = upper - lower
        position = (prices - lower) / bb_range

        # clamp values to [0, 1] range
        position = np.minimum(np.maximum(position, 0), 1)

        return Series(position).fillna(0.5)

    def _validate_features(self, features: DataFrame) -> None:
        # check for remaining NaN values
        assert not features.isnull().values.any(), "Features contain NaN values"
        assert not np.isinf(features.values).any(), "Features contain infinite values"

        # validate value ranges
        assert features['rsi'].between(0, 100).all(), "RSI values out of range"
        assert Series(features['volume_ma_ratio']).ge(0).all(), "Invalid volume ratio"
        assert features['bb_position'].between(0, 1).all(), "BB position out of range"
