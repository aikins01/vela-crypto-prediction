import pytest
import numpy as np
import pandas as pd
from src.model import MarketRegimeHMM

@pytest.fixture
def sample_features():
    # create feature data that mimics real market regimes
    np.random.seed(42)
    n_samples = 300

    features = pd.DataFrame()

    # simulate bull/bear/neutral patterns
    features['returns'] = np.concatenate([
        np.random.normal(0.001, 0.005, n_samples),  # bull
        np.random.normal(-0.001, 0.008, n_samples), # bear
        np.random.normal(0, 0.002, n_samples)       # neutral
    ])

    features['volatility'] = np.concatenate([
        np.random.normal(0.01, 0.005, n_samples),
        np.random.normal(0.02, 0.008, n_samples),
        np.random.normal(0.005, 0.002, n_samples)
    ])

    features['rsi'] = np.concatenate([
        np.random.uniform(60, 80, n_samples),
        np.random.uniform(20, 40, n_samples),
        np.random.uniform(40, 60, n_samples)
    ])

    features['volume_ma_ratio'] = np.concatenate([
        np.random.normal(1.2, 0.2, n_samples),
        np.random.normal(1.5, 0.3, n_samples),
        np.random.normal(1.0, 0.1, n_samples)
    ])

    return features

def test_model_initialization():
    model = MarketRegimeHMM(n_states=3)
    assert model.n_states == 3
    assert not model.trained
    assert model.model is not None

def test_model_training(sample_features):
    model = MarketRegimeHMM(n_states=3)
    log_likelihood = model.fit(sample_features)

    # basic training checks
    assert model.trained
    assert isinstance(log_likelihood, float)
    assert log_likelihood < 0

def test_state_prediction(sample_features):
    model = MarketRegimeHMM(n_states=3)
    model.fit(sample_features)
    states = model.predict_states(sample_features)

    # verify state predictions are valid
    assert len(states) == len(sample_features)
    assert all(isinstance(s, (int, np.integer)) for s in states)
    assert all(0 <= s < 3 for s in states)

def test_state_probabilities(sample_features):
    model = MarketRegimeHMM(n_states=3)
    model.fit(sample_features)
    probs = model.get_state_probabilities(sample_features)

    # check probability properties
    assert probs.shape == (len(sample_features), 3)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.all(probs >= 0) and np.all(probs <= 1)

def test_predict_without_training():
    model = MarketRegimeHMM()
    with pytest.raises(Exception):
        model.predict_states(pd.DataFrame(np.random.randn(10, 4)))

def test_feature_normalization(sample_features):
    model = MarketRegimeHMM()
    normalized = model._normalize_features(np.array(sample_features))

    # check normalized feature properties
    assert normalized.shape == sample_features.shape
    assert np.allclose(normalized.mean(axis=0), 0, atol=0.1)
    assert np.allclose(normalized.std(axis=0), 1, atol=0.1)
