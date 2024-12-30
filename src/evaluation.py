import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from src.model import MarketRegimeHMM
from src.features import FeatureEngineer
from src.backtesting import Strategy
from tqdm import tqdm

class ModelEvaluator:
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.engineer = FeatureEngineer()

    def cross_validate(
        self,
        data: pd.DataFrame,
        n_splits: int = 5
    ) -> List[Dict]:
        print("Starting cross-validation...")
        with tqdm(total=2, desc="Initial setup") as pbar:
            features = self.engineer.calculate_features(data)
            pbar.update(1)

            # make sure we have enough data
            min_samples = len(features) // n_splits
            if min_samples < 30:  # need reasonable amount of data per split
                raise ValueError(f"Not enough samples per split. Got {min_samples}, need at least 30")
            pbar.update(1)

        results = []
        # use expanding window approach
        for i in tqdm(range(2, n_splits + 1), desc="Processing folds"):
            split_point = (len(features) * i) // n_splits

            train_features = features.iloc[:split_point - min_samples]
            test_features = features.iloc[split_point - min_samples:split_point]

            train_data = data.loc[train_features.index]
            test_data = data.loc[test_features.index]

            if len(test_features) > 0:  # only evaluate if we have test data
                split_results = self.evaluate_split(train_data, test_data)
                results.append(split_results)

        return results

    def evaluate_split(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame
    ) -> Dict:
        with tqdm(total=4, desc="Evaluating split") as pbar:
            train_features = self.engineer.calculate_features(train_data)
            pbar.update(1)

            test_features = self.engineer.calculate_features(test_data)
            pbar.update(1)

            model = MarketRegimeHMM(n_states=3)
            model.fit(train_features)
            pbar.update(1)

            strategy = Strategy(self.initial_capital)
            metrics = strategy.backtest(model, test_data, test_features)
            pbar.update(1)

        return metrics
