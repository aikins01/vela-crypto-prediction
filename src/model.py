import numpy as np
from hmmlearn import hmm
import pandas as pd
from tqdm import tqdm
from numpy.typing import NDArray

class MarketRegimeHMM:
    def __init__(self, n_states: int = 3):
        self.n_states = n_states

        # init hmm with stable defaults
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=100,
            random_state=42,
            init_params=""  # don't init anything randomly
        )

        # set sensible starting probabilities
        self.model.startprob_ = np.array([0.4, 0.2, 0.4])

        # states tend to persist
        self.model.transmat_ = np.array([
            [0.8, 0.1, 0.1],  # bull -> bull more likely
            [0.1, 0.8, 0.1],  # bear -> bear more likely
            [0.1, 0.1, 0.8],  # neutral -> neutral more likely
        ])

        # set initial state means and covariances
        self.model.means_ = np.array([
            [0.002, 0.01, 65, 1.2],   # bull: up move, higher vol, high rsi, high vol
            [-0.002, 0.02, 35, 1.5],  # bear: down move, highest vol, low rsi, highest vol
            [0.0, 0.005, 50, 0.8]     # neutral: flat, low vol, mid rsi, low vol
        ])

        # set non-zero covariances to avoid numerical issues
        self.model.covars_ = np.array([
            [0.001, 0.001, 10, 0.1],  # small variances for stability
            [0.001, 0.001, 10, 0.1],
            [0.001, 0.001, 10, 0.1]
        ])

        self.trained = False

    def fit(self, features: pd.DataFrame) -> float:
        """train model on features and return likelihood"""
        print("Starting HMM training...")

        # convert to numpy and normalize
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
        return (X - mean) / (std + 1e-8)  # avoid div by 0
