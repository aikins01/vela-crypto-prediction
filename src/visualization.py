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
        metrics: Dict[str, float]
    ) -> go.Figure:
        print("Generating dashboard...")
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=(
                'Price & Market Regimes',
                'Portfolio Performance',
                'Drawdown'
            ),
            vertical_spacing=0.1,
            row_heights=[0.5, 0.3, 0.2]
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

        # lets show portfolio value and drawdown separately
        with tqdm(total=2, desc="Adding performance plots") as pbar:
            portfolio_array = np.array(portfolio_values, dtype=float)
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=portfolio_array,
                    name='Portfolio Value',
                    line=dict(color='blue'),
                ),
                row=2, col=1
            )
            pbar.update(1)

            portfolio_series = pd.Series(portfolio_array, index=dates)
            rolling_max = portfolio_series.expanding().max()
            drawdown = (portfolio_series - rolling_max) / rolling_max * 100

            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=drawdown,
                    name='Drawdown %',
                    fill='tonexty',
                    line=dict(color='red'),
                ),
                row=3, col=1
            )
            pbar.update(1)

        # all key metrics in one place
        metrics_text = (
            f"Total Return: {metrics['total_return']:.2%}<br>"
            f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}<br>"
            f"Max Drawdown: {metrics['max_drawdown']:.2%}<br>"
            f"Number of Trades: {int(metrics['n_trades'])}"
        )

        fig.add_annotation(
            xref='paper', yref='paper',
            x=1.0, y=1.0,
            text=metrics_text,
            showarrow=False,
            font=dict(size=10),
            bgcolor='white',
            bordercolor='black',
            borderwidth=1
        )

        symbol_name = price_data.name if hasattr(price_data, 'name') else 'Unknown'
        fig.update_layout(
            height=800,
            showlegend=True,
            title=f"Trading Performance Dashboard<br><sup>Symbol: {symbol_name}</sup>"
        )

        return fig

    def save_dashboard(self, fig: go.Figure, filename: str) -> None:
        print(f"Saving dashboard to {filename}...")
        fig.write_html(filename)
        print("Dashboard saved successfully.")
