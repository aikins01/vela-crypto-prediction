import numpy as np
from hmmlearn import hmm
import pandas as pd
from tqdm import tqdm
from numpy.typing import NDArray

class MarketRegimeHMM:
    def __init__(self, n_states: int = 3):
        self.n_states = n_states

        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=100,
            random_state=42
        )

        self.trained = False

    def fit(self, features: pd.DataFrame) -> float:
        """train model on features and return likelihood"""
        print("Starting HMM training...")

        features_array = np.array(features[['returns', 'volatility', 'rsi', 'volume_ma_ratio']])
        X = self._normalize_features(features_array)

        with tqdm(total=2, desc="Training model") as pbar:
            log_likelihood = self.model.fit(X).score(X)
            pbar.update(2)

        self.trained = True
        print(f"Training complete with log-likelihood: {log_likelihood:.2f}")
        return log_likelihood

    def predict_states(self, features: pd.DataFrame) -> np.ndarray:
        """get predicted state for each timestep"""
        with tqdm(total=2, desc="Predicting states") as pbar:
            features_array = np.array(features[['returns', 'volatility', 'rsi', 'volume_ma_ratio']])
            X = self._normalize_features(features_array)
            pbar.update(1)
            states = self.model.predict(X)
            pbar.update(1)
        return states

    def get_state_probabilities(self, features: pd.DataFrame) -> np.ndarray:
        """get probability of each state for each timestep"""
        with tqdm(total=2, desc="Calculating probabilities") as pbar:
            features_array = np.array(features[['returns', 'volatility', 'rsi', 'volume_ma_ratio']])
            X = self._normalize_features(features_array)
            pbar.update(1)
            probs = self.model.predict_proba(X)
            pbar.update(1)
        return probs

    def _normalize_features(self, X: NDArray) -> NDArray:
        """standardize features to mean 0, std 1"""
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / (std + 1e-8)
