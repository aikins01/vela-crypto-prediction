import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.visualization import DashboardGenerator
import plotly.graph_objects as go
from typing import Dict, List

@pytest.fixture
def sample_data() -> pd.DataFrame:
   # need some realistic looking price data for testing
   dates = pd.date_range(start='2024-01-01', periods=100, freq='15min')
   data = pd.DataFrame(index=dates)
   data['close'] = 100 * (1 + np.random.normal(0, 0.01, 100)).cumprod()
   data['high'] = data['close'] * (1 + np.random.uniform(0, 0.01, 100))
   data['low'] = data['close'] * (1 - np.random.uniform(0, 0.01, 100))
   data.name = 'TESTUSDT'
   return data

@pytest.fixture
def sample_states() -> np.ndarray:
   # lets have an even mix of states for testing
   return np.array([0, 1, 2] * 34)[:100]

@pytest.fixture
def sample_portfolio() -> List[float]:
   # simulate some portfolio returns
   return [float(x) for x in 10000 * (1 + np.random.normal(0, 0.02, 100)).cumprod()]

@pytest.fixture
def sample_metrics() -> Dict[str, float]:
   return {
       'total_return': 0.15,
       'sharpe_ratio': 1.2,
       'max_drawdown': -0.1,
       'n_trades': 25.0
   }

def test_dashboard_generation(sample_data, sample_states, sample_portfolio, sample_metrics):
   generator = DashboardGenerator()
   fig = generator.generate_dashboard(sample_data, sample_states, sample_portfolio, sample_metrics)

   # make sure we got a proper figure object
   assert isinstance(fig, go.Figure)
   traces = [t for t in fig.data]
   assert len(traces) >= 5
   assert fig.layout.height == 800
   assert 'Trading Performance Dashboard' in fig.layout.title.text

def test_dashboard_save(tmp_path, sample_data, sample_states, sample_portfolio, sample_metrics):
   generator = DashboardGenerator()
   fig = generator.generate_dashboard(sample_data, sample_states, sample_portfolio, sample_metrics)

   # check if we can save and file exists
   output_file = tmp_path / "dashboard.html"
   generator.save_dashboard(fig, str(output_file))
   assert output_file.exists()
   assert output_file.stat().st_size > 0

def test_state_colors(sample_data, sample_metrics):
    generator = DashboardGenerator()
    states = np.array([0, 1, 2])
    portfolio = [10000.0, 10000.0, 10000.0]

    fig = generator.generate_dashboard(sample_data[:3], states, portfolio, sample_metrics)
    state_traces = []
    for trace in fig.data:
        if isinstance(trace, go.Scatter) and trace.name in ['Bullish', 'Bearish', 'Neutral']:
            state_traces.append(trace)

    assert state_traces[0].marker.color == 'green'
    assert state_traces[1].marker.color == 'red'
    assert state_traces[2].marker.color == 'gray'
