import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.visualization import DashboardGenerator
import plotly.graph_objects as go
from typing import Dict, List

@pytest.fixture
def sample_data() -> pd.DataFrame:
    dates = pd.date_range(start='2024-01-01', periods=100, freq='15min')
    data = pd.DataFrame(index=dates)
    data['close'] = 100 * (1 + np.random.normal(0, 0.01, 100)).cumprod()
    data['high'] = data['close'] * (1 + np.random.uniform(0, 0.01, 100))
    data['low'] = data['close'] * (1 - np.random.uniform(0, 0.01, 100))
    return data

@pytest.fixture
def sample_states() -> np.ndarray:
    return np.array([0, 1, 2] * 34)[:100]

@pytest.fixture
def sample_portfolio() -> List[float]:
    return [float(x) for x in 10000 * (1 + np.random.normal(0, 0.02, 100)).cumprod()]

@pytest.fixture
def sample_metrics() -> Dict[str, float]:
    return {
        'model_return': 0.15,
        'hold_return': 0.10,
        'model_sharpe': 1.2,
        'hold_sharpe': 0.8,
        'model_drawdown': 0.1,
        'hold_drawdown': 0.15,
        'n_trades': 25,
        'stop_losses': 3
    }

def test_dashboard_generation(sample_data, sample_states, sample_portfolio, sample_metrics):
    generator = DashboardGenerator()
    fig = generator.generate_dashboard(
        sample_data,
        sample_states,
        sample_portfolio,
        sample_metrics,
        symbol='TESTUSDT'
    )

    assert isinstance(fig, go.Figure)
    assert len([t for t in fig.data]) >= 5
    assert fig.layout.height == 1000
    assert 'TESTUSDT' in fig.layout.title.text

def test_dashboard_save(tmp_path, sample_data, sample_states, sample_portfolio, sample_metrics):
    generator = DashboardGenerator()
    fig = generator.generate_dashboard(
        sample_data,
        sample_states,
        sample_portfolio,
        sample_metrics,
        symbol='TESTUSDT'
    )

    output_file = tmp_path / "dashboard.html"
    generator.save_dashboard(fig, str(output_file))
    assert output_file.exists()
    assert output_file.stat().st_size > 0

def test_state_colors(sample_data, sample_metrics):
    generator = DashboardGenerator()
    states = np.array([0, 1, 2])
    portfolio = [10000.0, 10000.0, 10000.0]

    fig = generator.generate_dashboard(
        sample_data[:3],
        states,
        portfolio,
        sample_metrics,
        symbol='TESTUSDT'
    )

    state_colors = {
        'Bullish': 'green',
        'Bearish': 'red',
        'Neutral': 'gray'
    }

    for trace in fig.data:
        if isinstance(trace, go.Scatter) and trace.name in state_colors:
            expected_color = state_colors[trace.name]
            marker_dict = getattr(trace, 'marker', {})
            if isinstance(marker_dict, dict):
                assert marker_dict.get('color') == expected_color, f"Wrong color for {trace.name}"
