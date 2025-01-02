import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Sequence
from numpy.typing import NDArray
from tqdm import tqdm

class DashboardGenerator:
   def generate_dashboard(
       self,
       price_data: pd.DataFrame,
       states: NDArray[np.int_],
       portfolio_values: Sequence[float],
       metrics: Dict[str, float],
       symbol: str  # add symbol parameter
   ) -> go.Figure:
       print("Generating dashboard...")
       fig = make_subplots(
           rows=4, cols=1,  # added row for comparison
           shared_xaxes=True,
           subplot_titles=(
               'Price & Market Regimes',
               'Strategy vs Buy & Hold',
               'Drawdowns',
               'Strategy Statistics'
           ),
           vertical_spacing=0.1,
           row_heights=[0.4, 0.2, 0.2, 0.2]
       )

       dates = price_data.index
       state_names = ['Bullish', 'Bearish', 'Neutral']
       colors = ['green', 'red', 'gray']

       # hmm need to show price colored by state
       with tqdm(total=len(np.unique(states)), desc="Adding market states") as pbar:
           for i, state in enumerate(np.unique(states)):
               mask = states == state
               fig.add_trace(
                   go.Scatter(
                       x=dates[mask],
                       y=price_data.loc[dates[mask], 'close'],
                       mode='markers',
                       name=state_names[int(i)],
                       marker=dict(color=colors[int(i)]),
                   ),
                   row=1, col=1
               )
               pbar.update(1)

       # lets compare strategy vs buy & hold
       with tqdm(total=2, desc="Adding performance comparison") as pbar:
           # Strategy performance
           fig.add_trace(
               go.Scatter(
                   x=dates,
                   y=portfolio_values,
                   name='Strategy',
                   line=dict(color='blue'),
               ),
               row=2, col=1
           )
           pbar.update(1)

           # Buy & Hold performance
           initial_value = 10000  # match strategy initial capital
           hold_values = initial_value * (price_data['close'] / price_data['close'].iloc[0])
           fig.add_trace(
               go.Scatter(
                   x=dates,
                   y=hold_values,
                   name='Buy & Hold',
                   line=dict(color='gray', dash='dash'),
               ),
               row=2, col=1
           )
           pbar.update(1)

       # lets see those drawdowns
       with tqdm(total=2, desc="Calculating drawdowns") as pbar:
           # Strategy drawdown
           strategy_dd = self._calculate_drawdown(portfolio_values)
           fig.add_trace(
               go.Scatter(
                   x=dates,
                   y=strategy_dd,
                   name='Strategy DD',
                   fill='tonexty',
                   line=dict(color='red'),
               ),
               row=3, col=1
           )
           pbar.update(1)

           # Buy & Hold drawdown
           hold_dd = self._calculate_drawdown(hold_values)
           fig.add_trace(
               go.Scatter(
                   x=dates,
                   y=hold_dd,
                   name='Buy & Hold DD',
                   line=dict(color='gray', dash='dash'),
               ),
               row=3, col=1
           )
           pbar.update(1)

       # better show all metrics clearly
       metrics_text = (
           f"Strategy Metrics:<br>"
           f"Return: {metrics['model_return']:.2%}<br>"
           f"Sharpe: {metrics['model_sharpe']:.2f}<br>"
           f"Max DD: {metrics['model_drawdown']:.2%}<br>"
           f"Trades: {metrics['n_trades']}<br>"
           f"Stop Losses: {metrics['stop_losses']}<br><br>"
           f"Buy & Hold Metrics:<br>"
           f"Return: {metrics['hold_return']:.2%}<br>"
           f"Sharpe: {metrics['hold_sharpe']:.2f}<br>"
           f"Max DD: {metrics['hold_drawdown']:.2%}"
       )

       fig.add_annotation(
           xref='x domain', yref='y domain',
           x=0.5, y=0.5,
           text=metrics_text,
           showarrow=False,
           font=dict(size=10),
           bgcolor='white',
           bordercolor='black',
           borderwidth=1,
           row=4, col=1
       )

       fig.update_layout(
           height=1000,
           showlegend=True,
           title=f"Trading Performance Dashboard - {symbol}"
       )

       return fig

   def save_dashboard(self, fig: go.Figure, filename: str) -> None:
       print(f"Saving dashboard to {filename}...")
       fig.write_html(filename)
       print("Dashboard saved successfully.")

   def _calculate_drawdown(self, values: Sequence[float]) -> np.ndarray:
       series = pd.Series(values)
       peak = series.expanding().max()
       drawdown = ((series - peak) / peak) * 100
       return drawdown.values
