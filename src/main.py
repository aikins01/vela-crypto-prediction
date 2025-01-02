from src.data_collection import BinanceDataCollector
from src.features import FeatureEngineer
from src.model import MarketRegimeHMM
from src.backtesting import Strategy
from src.visualization import DashboardGenerator
from tqdm import tqdm
import pandas as pd
import os

def train_test_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """split 3 months training, 1 week testing"""
    # for 15min intervals
    bars_per_hour = 4
    bars_per_day = bars_per_hour * 24

    train_days = 90
    train_bars = train_days * bars_per_day

    train = data.iloc[:train_bars]
    test = data.iloc[train_bars:]

    return train, test

def main():
    os.makedirs('results', exist_ok=True)

    collector = BinanceDataCollector()
    tokens = collector.get_small_cap_symbols()
    print(f"\nFound {len(tokens)} eligible tokens")

    results = []
    for token in tqdm(tokens[:5], desc="Analyzing tokens"):
        symbol = token['symbol']
        print(f"\nAnalyzing {symbol}")

        with tqdm(total=4, desc=f"Processing {symbol}") as pbar:
            # get data with 15min intervals
            data = collector.fetch_historical_data(
                symbol=symbol,
                interval='15m'
            )
            pbar.update(1)

            try:
                train_data, test_data = train_test_split(data)
                print(f"Training size: {len(train_data)} bars, Test size: {len(test_data)} bars")
            except ValueError as e:
                print(f"Skipping {symbol}: {str(e)}")
                continue

            # prepare features
            engineer = FeatureEngineer()
            train_features = engineer.calculate_features(train_data)
            test_features = engineer.calculate_features(test_data)
            pbar.update(1)

            # train model on training data only
            model = MarketRegimeHMM()
            model.fit(train_features)
            test_states = model.predict_states(test_features)
            pbar.update(1)

            # backtest on test period
            strategy = Strategy(10000)  # start with 10k USDT
            metrics = strategy.backtest(
                model,
                test_data.loc[test_features.index],
                test_features
            )
            results.append((symbol, metrics))

            # visualize test period results
            dashboard = DashboardGenerator()
            fig = dashboard.generate_dashboard(
                test_data.loc[test_features.index],
                test_states,
                strategy.portfolio_value,
                metrics,
                symbol=symbol
            )
            dashboard.save_dashboard(fig, os.path.join('results', f"results_{symbol}.html"))
            pbar.update(1)

    # show final results
    print("\nResults Summary:")
    for symbol, metrics in results:
        print(f"\n{symbol}:")
        print(f"Strategy Return: {metrics['model_return']:.2%}")
        print(f"Buy & Hold Return: {metrics['hold_return']:.2%}")
        print(f"Strategy Sharpe: {metrics['model_sharpe']:.2f}")
        print(f"Strategy Max DD: {metrics['model_drawdown']:.2%}")
        print(f"Trades: {metrics['n_trades']}")
        print(f"Stop Losses Hit: {metrics['stop_losses']}")

if __name__ == "__main__":
    main()
