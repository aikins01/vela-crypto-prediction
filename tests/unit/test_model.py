import pytest
import numpy as np
import pandas as pd
from src.model import MarketRegimeHMM

@pytest.fixture
def sample_features():
   n_samples = 300
   np.random.seed(42)

   bull = np.column_stack((
       np.random.normal(0.001, 0.005, n_samples),
       np.random.normal(0.01, 0.005, n_samples)
   ))

   bear = np.column_stack((
       np.random.normal(-0.001, 0.008, n_samples),
       np.random.normal(0.02, 0.008, n_samples)
   ))

   neutral = np.column_stack((
       np.random.normal(0, 0.002, n_samples),
       np.random.normal(0.005, 0.002, n_samples)
   ))

   features = np.vstack([bull, bear, neutral])
   df = pd.DataFrame(data=features)
   df = df.rename(columns={0: 'returns', 1: 'volatility'})
   return df

def test_model_training(sample_features):
   model = MarketRegimeHMM(n_states=3)
   log_likelihood = model.fit(sample_features)

   assert model.trained
   assert isinstance(log_likelihood, float)

def test_state_prediction(sample_features):
   model = MarketRegimeHMM(n_states=3)
   model.fit(sample_features)

   states = model.predict_states(sample_features)
   assert len(states) == len(sample_features)
   assert all(0 <= s < 3 for s in states)

def test_predict_without_training():
   model = MarketRegimeHMM()
   with pytest.raises(ValueError):
       model.predict_states(pd.DataFrame(np.random.randn(10, 2)))
