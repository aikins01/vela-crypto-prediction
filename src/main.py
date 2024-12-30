from src.data_collection import BinanceDataCollector
from src.features import FeatureEngineer
from src.model import MarketRegimeHMM
from src.backtesting import Strategy
from src.visualization import DashboardGenerator
from tqdm import tqdm
import os

def main():
    os.makedirs('results', exist_ok=True)

    # first get our list of tokens to analyze
    collector = BinanceDataCollector()
    tokens = collector.get_small_cap_symbols()
    print(f"Found {len(tokens)} eligible tokens")

    results = []
    # let's process each token one by one with progress tracking
    for token in tqdm(tokens[:5], desc="Analyzing tokens"):
        symbol = token['symbol']
        print(f"\nAnalyzing {symbol}")

        with tqdm(total=4, desc=f"Processing {symbol}") as pbar:
            # need price data first
            data = collector.fetch_historical_data(symbol, interval='5m')
            pbar.update(1)

            # calculate all our technical features
            engineer = FeatureEngineer()
            features = engineer.calculate_features(data)
            pbar.update(1)

            # train hmm and get states
            model = MarketRegimeHMM()
            model.fit(features)
            states = model.predict_states(features)
            pbar.update(1)

            # see how we would have performed
            strategy = Strategy(10000)
            metrics = strategy.backtest(model, data.loc[features.index], features)
            results.append((symbol, metrics))

            # save nice visualization
            dashboard = DashboardGenerator()
            fig = dashboard.generate_dashboard(
                data.loc[features.index],
                states,
                strategy.portfolio_value,
                metrics
            )
            dashboard.save_dashboard(
                fig=fig,
                filename=os.path.join('results', f"results_{symbol}.html")
            )
            pbar.update(1)

    # show our results
    print("\nResults Summary:")
    for symbol, metrics in results:
        print(f"\n{symbol}:")
        print(f"Return: {metrics['total_return']:.2%}")
        print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
        print(f"Max DD: {metrics['max_drawdown']:.2%}")

if __name__ == "__main__":
    main()
