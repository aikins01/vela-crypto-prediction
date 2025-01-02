import pandas as pd
import numpy as np
from typing import Dict, List
from src.model import MarketRegimeHMM
from tqdm import tqdm

class Strategy:
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.portfolio_value = []

        # risk params
        self.position_size = 0.02  # 2% per trade
        self.stop_loss = 0.01      # 1% stop
        self.min_vol = 0.001       # min vol to trade
        self.max_vol = 0.02        # max vol to trade
        self.min_state_prob = 0.6  # need confident signal
        self.min_hold_bars = 12    # hold at least 1hr (12 x 5min)

    def should_trade(self, vol: float, prob: float) -> bool:
        # check both vol and signal strength
        return (self.min_vol <= vol <= self.max_vol and
                prob >= self.min_state_prob)

    def backtest(
        self,
        model: MarketRegimeHMM,
        data: pd.DataFrame,
        features: pd.DataFrame
    ) -> Dict:
        data = data.loc[features.index]

        # tracking vars
        cash = self.initial_capital
        shares = 0
        values = []
        trades = []
        entry_price = None
        last_trade = -self.min_hold_bars  # allow first trade immediately

        # buy & hold comparison
        hold_shares = self.initial_capital / data['close'].iloc[0]
        hold_values = []

        # get states and probabilities
        states = model.predict_states(features)
        probs = model.get_state_probabilities(features)

        for i in tqdm(range(len(data)), desc="Running backtest"):
            price = data['close'].iloc[i]
            vol = features['volatility'].iloc[i]
            state_prob = probs[i].max()

            hold_values.append(hold_shares * price)

            # check stops
            if shares > 0:
                loss = (price - entry_price) / entry_price
                if loss < -self.stop_loss:
                    cash += shares * price
                    shares = 0
                    entry_price = None
                    trades.append({'type': 'stop'})
                    last_trade = i

            # only trade if conditions good and enough time passed
            if (self.should_trade(vol, state_prob) and
                i - last_trade >= self.min_hold_bars):
                state = states[i]
                if shares == 0 and state == 0:  # bullish entry
                    position = min(cash * self.position_size, cash)
                    shares = position / price
                    cash -= position
                    entry_price = price
                    trades.append({'type': 'buy'})
                    last_trade = i
                elif shares > 0 and state == 1:  # bearish exit
                    cash += shares * price
                    shares = 0
                    entry_price = None
                    trades.append({'type': 'sell'})
                    last_trade = i

            values.append(cash + shares * price)

        self.portfolio_value = values
        return self._calculate_metrics(trades, hold_values)

    def _calculate_metrics(self, trades: List[Dict], hold_values: List[float]) -> Dict:
        if not self.portfolio_value:
            return {}

        final_return = (self.portfolio_value[-1] - self.initial_capital) / self.initial_capital * 100
        hold_return = (hold_values[-1] - self.initial_capital) / self.initial_capital * 100

        returns = pd.Series(self.portfolio_value).pct_change().fillna(0)
        hold_rets = pd.Series(hold_values).pct_change().fillna(0)

        return {
            'model_return': final_return,
            'hold_return': hold_return,
            'model_sharpe': returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 1 else 0,
            'hold_sharpe': hold_rets.mean() / hold_rets.std() * np.sqrt(252) if len(hold_rets) > 1 else 0,
            'model_drawdown': self._calculate_drawdown(self.portfolio_value),
            'hold_drawdown': self._calculate_drawdown(hold_values),
            'n_trades': len([t for t in trades if t['type'] in ['buy', 'sell']]),
            'stop_losses': len([t for t in trades if t['type'] == 'stop'])
        }

    def _calculate_drawdown(self, values: List[float]) -> float:
        peaks = pd.Series(values).cummax()
        drawdown = ((pd.Series(values) - peaks) / peaks).min() * 100
        return abs(float(drawdown))
