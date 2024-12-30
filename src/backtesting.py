import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from .model import MarketRegimeHMM

class Strategy:
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.positions = []
        self.portfolio_value = []

    def backtest(
        self,
        model: MarketRegimeHMM,
        data: pd.DataFrame,
        features: pd.DataFrame
    ) -> Dict:
        capital = self.initial_capital
        position = 0
        trades = []

        states = model.predict_states(features)

        for i in range(len(data)):
            state = states[i]

            # no position and bullish state
            if position == 0 and state == 0:
                position = capital / data['close'].iloc[i]
                capital = 0
                trades.append({
                    'type': 'buy',
                    'price': data['close'].iloc[i],
                    'size': position
                })

            # have position and bearish state
            elif position > 0 and state == 1:
                capital = position * data['close'].iloc[i]
                position = 0
                trades.append({
                    'type': 'sell',
                    'price': data['close'].iloc[i],
                    'size': position
                })

            # track portfolio value
            self.portfolio_value.append(capital + position * data['close'].iloc[i])

        return self._calculate_metrics(trades)

    def _calculate_metrics(self, trades: List[Dict]) -> Dict:
        # need to calculate performance stats
        returns = pd.Series(self.portfolio_value).pct_change()

        return {
            'total_return': (self.portfolio_value[-1] - self.initial_capital) / self.initial_capital,
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252),
            'max_drawdown': self._calculate_max_drawdown(),
            'n_trades': len(trades)
        }

    def _calculate_max_drawdown(self) -> float:
        portfolio = pd.Series(self.portfolio_value)
        rolling_max = portfolio.expanding().max()
        drawdown = portfolio - rolling_max
        return drawdown.min()
