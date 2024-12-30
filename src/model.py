import numpy as np
from hmmlearn import hmm
from typing import Tuple, Optional
import pandas as pd

class MarketRegimeHMM:
    def __init__(self, n_states: int = 3, n_iterations: int = 100):
        self.n_states = n_states
        self.n_iterations = n_iterations
        # adding regularization to prevent covariance issues
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="diag",  # use diagonal covariance matrix
            n_iter=n_iterations,
            random_state=42
        )
        self.trained = False

    def fit(self, features: pd.DataFrame) -> float:
        # normalize features to help with numerical stability
        X = self._normalize_features(features.values)
        log_likelihood = self.model.fit(X).score(X)
        self.trained = True
        return log_likelihood

    def _normalize_features(self, X: np.ndarray) -> np.ndarray:
        # scale features to zero mean and unit variance
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / (std + 1e-8)

    def predict_states(self, features: pd.DataFrame) -> np.ndarray:
        if not self.trained:
            raise ValueError("Model must be trained first using fit()")
        X = self._normalize_features(features.values)
        return self.model.predict(X)

    def get_state_probabilities(self, features: pd.DataFrame) -> np.ndarray:
        if not self.trained:
            raise ValueError("Model must be trained first using fit()")
        X = self._normalize_features(features.values)
        return self.model.predict_proba(X)
