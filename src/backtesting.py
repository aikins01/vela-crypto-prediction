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
        self.position_size = 0.015  # 1.5% position
        self.stop_loss = 0.03       # 3% stop
        self.min_vol = 0.003        # min vol
        self.max_vol = 0.03         # max vol
        self.min_state_prob = 0.7   # 70% confidence
        self.min_hold_bars = 12     # min hold time
        self.take_profit = 0.06     # 6% target

    def should_trade(self, vol: float, prob: float) -> bool:
        return (self.min_vol <= vol <= self.max_vol and
                prob >= self.min_state_prob)

    def backtest(self, model: MarketRegimeHMM, data: pd.DataFrame, features: pd.DataFrame) -> Dict:
        data = data.loc[features.index]

        # track portfolio
        cash = self.initial_capital
        shares = 0
        values = []
        trades = []
        entry_price = None
        last_trade = -self.min_hold_bars

        # buy & hold comparison
        hold_shares = self.initial_capital / data['close'].iloc[0]
        hold_values = []

        states = model.predict_states(features)
        probs = model.get_state_probabilities(features)

        for i in tqdm(range(len(data)), desc="running backtest"):
            price = data['close'].iloc[i]
            vol = features['volatility'].iloc[i]
            state_prob = probs[i].max()

            hold_values.append(hold_shares * price)

            # check stops
            if shares > 0:
                pnl = (price - entry_price) / entry_price

                if pnl < -self.stop_loss:
                    cash += shares * price * 0.999
                    shares = 0
                    entry_price = None
                    trades.append({'type': 'stop'})
                    last_trade = i

                elif pnl > self.take_profit:
                    cash += shares * price * 0.999
                    shares = 0
                    entry_price = None
                    trades.append({'type': 'profit'})
                    last_trade = i

            if (self.should_trade(vol, state_prob) and
                i - last_trade >= self.min_hold_bars):

                state = states[i]
                # check next bar state
                if i < len(data) - 1 and states[i+1] == state:

                    if shares == 0 and state == 0:  # bull entry
                        position = cash * self.position_size
                        shares = (position * 0.999) / price
                        cash -= position
                        entry_price = price
                        trades.append({'type': 'buy'})
                        last_trade = i

                    elif shares > 0 and state == 1:  # bear exit
                        cash += shares * price * 0.999
                        shares = 0
                        entry_price = None
                        trades.append({'type': 'sell'})
                        last_trade = i

            values.append(cash + shares * price)

        self.portfolio_value = values
        return self._calculate_metrics(trades, values, hold_values)

    def _calculate_metrics(self, trades: List[Dict], values: List[float],
                         hold_values: List[float]) -> Dict:
        if not values:
            return {}

        final_return = ((values[-1] - self.initial_capital) /
                       self.initial_capital) * 100
        hold_return = ((hold_values[-1] - self.initial_capital) /
                      self.initial_capital) * 100

        returns = pd.Series(values).pct_change().fillna(0)
        hold_rets = pd.Series(hold_values).pct_change().fillna(0)

        def safe_sharpe(rets):
            std = rets.std()
            return rets.mean() / std * np.sqrt(252) if std > 0 else 0

        return {
            'model_return': final_return,
            'hold_return': hold_return,
            'model_sharpe': safe_sharpe(returns),
            'hold_sharpe': safe_sharpe(hold_rets),
            'model_drawdown': self._calculate_drawdown(values),
            'hold_drawdown': self._calculate_drawdown(hold_values),
            'n_trades': len([t for t in trades if t['type'] in ['buy', 'sell']]),
            'stop_losses': len([t for t in trades if t['type'] == 'stop']),
            'take_profits': len([t for t in trades if t['type'] == 'profit'])
        }

    def _calculate_drawdown(self, values: List[float]) -> float:
        peaks = pd.Series(values).expanding().max()
        drawdowns = (pd.Series(values) - peaks) / peaks
        return abs(float(drawdowns.min() * 100))
