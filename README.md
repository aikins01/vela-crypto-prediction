# vela crypto prediction

Short-term crypto prediction for small-cap assets using Hidden Markov Models.

## requirements

- python 3.8+
- binance api access
- 8GB+ RAM recommended

## setup
```bash
git clone <repo-url>
cd vela-crypto-prediction
python -m venv venv
source venv/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

## structure
- `src/`:
  - `data_collection.py`: binance api integration, small-cap filtering
  - `features.py`: technical indicator calculation
  - `model.py`: hmm with baum-welch implementation
  - `backtesting.py`: strategy testing
  - `evaluation.py`: cross-validation
  - `visualization.py`: performance dashboards

## key features
- targets recently listed (<90 days) tokens under $100M market cap
- uses 5-minute candles for short-term predictions
- identifies market regimes using hmm
- provides backtesting with standard metrics
- includes interactive visualizations

## usage
```python
from src.data_collection import BinanceDataCollector
from src.features import FeatureEngineer
from src.model import MarketRegimeHMM
from src.backtesting import Strategy
from src.visualization import DashboardGenerator

# get data
collector = BinanceDataCollector()
tokens = collector.get_small_cap_symbols()
data = collector.fetch_historical_data(tokens[0]['symbol'], interval='5m')

# train & backtest
engineer = FeatureEngineer()
features = engineer.calculate_features(data)
model = MarketRegimeHMM()
model.fit(features)

strategy = Strategy(10000)
results = strategy.backtest(model, data, features)

# visualize
dashboard = DashboardGenerator()
fig = dashboard.generate_dashboard(
    data,
    model.predict_states(features),
    strategy.portfolio_value,
    results
)
dashboard.save_dashboard(fig, 'results.html')
```

## model choices

- three states (bull/bear/neutral):
  * matches typical market phases
  * balances complexity vs accuracy
  * reduces overfitting risk

- 5m intervals:
  * closest to recommended 10m
  * good signal/noise ratio
  * sufficient sample size

- features:
  * returns: momentum
  * volatility: risk measure
  * volume ratios: activity level
  * rsi: overbought/oversold
  * high-low ratio: price range

- training period: 3 months
  * matches listing age filter
  * provides adequate samples
  * relevant to current conditions

## performance metrics
- total return
- sharpe ratio
- maximum drawdown
- trade count
- win rate

## limitations
- binance-only
- needs $1M+ daily volume
- past performance ≠ future results
- sensitive to market regime changes

## testing
```bash
python -m pytest tests/unit/* tests/integration/*
```
