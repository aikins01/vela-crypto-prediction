import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from .model import MarketRegimeHMM
from .features import FeatureEngineer
from .backtesting import Strategy

class ModelEvaluator:
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.engineer = FeatureEngineer()

    def cross_validate(
        self,
        data: pd.DataFrame,
        n_splits: int = 5
    ) -> List[Dict]:
        # split data into n_splits chunks
        chunk_size = len(data) // n_splits
        results = []

        for i in range(n_splits):
            # get train/test splits
            train_end = (i + 1) * chunk_size
            test_end = min(train_end + chunk_size, len(data))

            train_data = data.iloc[:train_end]
            test_data = data.iloc[train_end:test_end]

            # evaluate on this split
            split_results = self.evaluate_split(train_data, test_data)
            results.append(split_results)

        return results

    def evaluate_split(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame
    ) -> Dict:
        # prepare features
        train_features = self.engineer.calculate_features(train_data)
        test_features = self.engineer.calculate_features(test_data)

        # train model
        model = MarketRegimeHMM(n_states=3)
        model.fit(train_features)

        # backtest
        strategy = Strategy(self.initial_capital)
        metrics = strategy.backtest(model, test_data, test_features)

        return metrics
