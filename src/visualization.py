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
        symbol: str
    ) -> go.Figure:
        print("generating dashboard...")
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            subplot_titles=(
                'price & market states',
                'strategy vs buy & hold',
                'drawdowns',
                'key metrics'
            ),
            vertical_spacing=0.1,
            row_heights=[0.4, 0.2, 0.2, 0.2],
            specs=[[{"type": "scatter"}],
                  [{"type": "scatter"}],
                  [{"type": "scatter"}],
                  [{"type": "table"}]]
        )

        dates = price_data.index
        state_names = ['bullish', 'bearish', 'neutral']
        colors = ['green', 'red', 'gray']

        # i'm showing price colored by what state i think the market is in
        with tqdm(total=len(np.unique(states)), desc="adding market states") as pbar:
            for i, state in enumerate(np.unique(states)):
                mask = states == state
                fig.add_trace(
                    go.Scatter(
                        x=dates[mask],
                        y=price_data.loc[dates[mask], 'close'],
                        mode='markers',
                        name=state_names[int(i)],
                        marker=dict(color=colors[int(i)], size=8),
                    ),
                    row=1, col=1
                )
                pbar.update(1)

        # comparing how my strategy did vs just holding
        with tqdm(total=2, desc="adding performance comparison") as pbar:
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=portfolio_values,
                    name='my strategy',
                    line=dict(color='blue', width=2),
                ),
                row=2, col=1
            )
            pbar.update(1)

            initial_value = portfolio_values[0]
            hold_values = initial_value * (price_data['close'] / price_data['close'].iloc[0])
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=hold_values,
                    name='buy & hold',
                    line=dict(color='gray', dash='dash'),
                ),
                row=2, col=1
            )
            pbar.update(1)

        # tracking the pain - how far underwater did we go
        with tqdm(total=2, desc="calculating drawdowns") as pbar:
            strategy_dd = self._calculate_drawdown(portfolio_values)
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=strategy_dd,
                    name='my drawdown',
                    fill='tonexty',
                    line=dict(color='rgba(255,0,0,0.5)'),
                ),
                row=3, col=1
            )
            pbar.update(1)

            hold_dd = self._calculate_drawdown(hold_values)
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=hold_dd,
                    name='buy & hold drawdown',
                    line=dict(color='gray', dash='dash'),
                ),
                row=3, col=1
            )
            pbar.update(1)

        # putting all metrics in an easy to read table
        metrics_table = go.Table(
            header=dict(
                values=['what i measure', 'my strategy', 'buy & hold'],
                font=dict(size=12, color='white'),
                fill_color='darkblue',
                align='left'
            ),
            cells=dict(
                values=[
                    ['total return', 'sharpe ratio', 'max pain', 'trades made', 'stops hit'],
                    [
                        f"{metrics['model_return']:.2%}",
                        f"{metrics['model_sharpe']:.2f}",
                        f"{metrics['model_drawdown']:.2%}",
                        f"{metrics['n_trades']}",
                        f"{metrics['stop_losses']}"
                    ],
                    [
                        f"{metrics['hold_return']:.2%}",
                        f"{metrics['hold_sharpe']:.2f}",
                        f"{metrics['hold_drawdown']:.2%}",
                        "n/a",
                        "n/a"
                    ]
                ],
                font=dict(size=11),
                fill_color=[['white', 'lightblue', 'white']],
                align='left'
            )
        )
        fig.add_trace(metrics_table, row=4, col=1)

        fig.update_layout(
            height=1000,
            showlegend=True,
            title=f"how i traded {symbol}"
        )

        return fig

    def save_dashboard(self, fig: go.Figure, filename: str) -> None:
        print(f"saving dashboard to {filename}...")
        fig.write_html(filename)
        print("dashboard saved successfully.")

    def _calculate_drawdown(self, values: Sequence[float]) -> np.ndarray:
        """track how far underwater we go"""
        series = pd.Series(values)
        peak = series.expanding().max()
        drawdown = ((series - peak) / peak) * 100
        return drawdown.values
