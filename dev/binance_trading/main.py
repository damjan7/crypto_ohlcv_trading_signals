import sys
import os
import datetime
import pandas as pd

# Add the current directory to sys.path to ensure imports work if run from here
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.pipeline import DataPipeline
from strategies.examples import SimpleSortingStrategy, AverageFeatureStrategy
from backtest.engine import BacktestEngine, BacktestConfig
from visualization.plots import plot_backtest_results

def main():
    print("Starting Binance Trading Backtest...")
    
    # 1. Configuration
    # Use real dates?
    START_DATE = datetime.datetime(2023, 1, 1)
    END_DATE = datetime.datetime(2023, 12, 31)
    INTERVAL = "1d"
    DOWNLOAD = True # Set to True to verify donwload if package installed
    
    # 2. Pipeline
    print("Initializing Data Pipeline...")
    # resolved path: .../dev/binance_trading/data_storage
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_storage")
    pipeline = DataPipeline(data_dir=data_dir)
    
    print("Loading/Processing All Available Data...")
    processed_data = pipeline.load_and_process(
        pairs=None, # binance_historical_data Loads all USDT pairs in that case
        start_date=START_DATE,
        end_date=END_DATE,
        interval=INTERVAL,
        update_data=DOWNLOAD # TRUE = DOWNLOAD/UPDATE (ALL) DATA
    )
        
        
    # Create Cross-Sectional Data Dictionary
    print("Creating Cross-Sectional Data...")
    cs_data = pipeline.create_data_dict_for_backtest(processed_data)
    
    # 3. Strategy
    print("Initializing Strategy...")    
    # Example: Simple Sorting on RSI (Long Top Quintile)
    # Ensure RSI exists
    if 'RSI' in cs_data:
        strategy = SimpleSortingStrategy(feature_name="RSI", num_quantiles=5)
    else:
        print("RSI feature missing, falling back to Momentum if available or creating random weights")
        strategy = SimpleSortingStrategy(feature_name="momentum_12", num_quantiles=2) if 'momentum_12' in cs_data else None

    # 4. Backtest
    if strategy and cs_data:
        print("Running Backtest...")
        config = BacktestConfig(initial_balance=10000, trading_fee=0.001)
        engine = BacktestEngine(data_dict=cs_data, strategy=strategy, config=config)
        results = engine.run()
        
        # 5. Results
        print("Performance Metrics:")
        metrics = engine.get_performance_metrics()
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
            
        print("Plotting results (close window to finish)...")
        # In a headless env this might just print/save.
        try:
            plot_backtest_results(results)
        except Exception as e:
            print(f"Plotting failed (expected in headless): {e}")
    else:
        print("Could not run backtest due to missing data/strategy.")

if __name__ == "__main__":
    import numpy as np # Imported here for dummy generation
    main()
