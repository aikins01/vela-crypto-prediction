# vela crypto prediction

short-term crypto prediction for small-cap assets using hidden markov models with baum-welch algorithm.

## overview
built to predict short-term moves in recently listed, low-cap cryptos. focuses on:
- market regime detection (bull/bear/neutral)
- small caps under $100M
- uses 15min data from binance
- backtests against buy & hold

## why these choices?

### timeframe choice
started with suggested 10min interval but found:
- binance api only offers 1m, 3m, 5m, 15m, 30m
- went with 15m as best balance of:
  * enough trades to be useful
  * less noise than 5m
  * better sample size for training
  * manageable api calls

### market regimes
picked 3 states because:
- matches how markets actually behave
- 2 states too simple for crypto
- 4+ states led to overfitting
- clear signals for trading

### features used
built on required ohlc/volume with:
- returns: momentum
- volatility: risk gauge
- rsi: overbought/oversold
- volume ratios: unusual activity

picked these after testing different combos - gave clearest regime signals

### training approach
used 3 months training because:
- matches our <90 day token filter
- enough data to train hmm properly
- recent enough to matter
- worked better than 1-2 months

## setup

### requirements
- python 3.8+
- binance api access
- 8gb ram recommended

### install
```bash
git clone <repo-url>
cd vela-crypto
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on windows
pip install -r requirements.txt
```

## how it works

### getting data
```python
from src.data_collection import BinanceDataCollector
collector = BinanceDataCollector()
tokens = collector.get_small_cap_symbols()
data = collector.fetch_historical_data(tokens[0]['symbol'])
```

### training model
```python
from src.features import FeatureEngineer
from src.model import MarketRegimeHMM

engineer = FeatureEngineer()
features = engineer.calculate_features(data)

model = MarketRegimeHMM()
model.fit(features)
```

### backtesting
```python
from src.strategy import Strategy

strategy = Strategy(initial_capital=10000)
metrics = strategy.backtest(model, data, features)
```

## project layout
```
src/
├── data_collection.py  # binance api stuff
├── features.py         # technical indicators
├── model.py           # hmm implementation
├── backtesting.py     # strategy testing
├── evaluation.py      # cross validation
└── visualization.py   # performance dashboards
```

## risk controls
tuned for small cap volatility:
- 1.5% per trade: limits downside
- 3% stops: room for noise
- 6% take profit: realistic targets
- min 12 bars hold: avoid churn
- vol filters: skip crazy periods

## performance tracking
- returns vs buy & hold
- sharpe ratio
- max drawdown
- win rate
- trade count
- stop losses hit

## known limits
- binance only data
- needs decent volume ($1M/day)
- past results aren't future promises
- execution risk in thin markets

## running tests
```bash
python -m pytest tests/
```

## future ideas
- smarter position sizing
- multi-timeframe analysis
- more technical features
- portfolio risk mgmt

## results viewing
makes interactive dashboards showing:
- price with market regimes
- strategy vs buy/hold
- drawdowns over time
- trade entries/exits
- key metrics

saved as html in results/
