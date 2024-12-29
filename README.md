# vela crypto prediction

short-term crypto prediction using hidden markov models for small-cap assets

## setup

```bash
# create virtual environment
python -m venv venv

# activate environment
source venv/bin/activate  # unix/mac
# venv\Scripts\activate   # windows

# install requirements
pip install -r requirements.txt
```

## structure

- `src/`: source code
  - `data_collection.py`: binance api data fetching
  - `features.py`: feature engineering
  - `model.py`: hmm implementation
  - `backtesting.py`: strategy testing
- `tests/`: unit tests
- `data/`: stored market data
- `notebooks/`: analysis notebooks

## overview

this project uses a hidden markov model with baum-welch algorithm to:
- predict short-term movements in small-cap cryptocurrencies
- identify market regimes (bullish, bearish, neutral)
- backtest trading strategies based on predictions

## usage

[to be added as implementation progresses]
