import datetime
import numpy as np
import pandas as pd
import queue
from typing import Dict, List, Optional
from dataclasses import dataclass

from feature_generator import FeatureGenerator
from base_trainer import ModelTrainer, get_model, create_pipeline
from data_processor import create_feature_dataset, create_target_dataset

import matplotlib.pyplot as plt


"""
Simpler Backtester containing only logic for Long Short Strategies
- i.e. starting capital of xy
- commission rate of 0.1%
- slippage of 0.05%
- position size can be varied depending on strength of signal
- leverage?

- signals are -1, 0, 1 (short, flat, long)

First implementation:
pre-defined pairs, long short, 100% of capital invested at all times, no fees or slippage
"""

@dataclass
class BacktestConfig:
    initial_capital: float = 100000
    commission_rate: float = 0.0
    slippage: float = 0.0
    position_size: float = 1  # Fraction of capital to risk per trade

"""
TODO:
- implement weight check in run backtest, i.e. signal * position_size, 
 if </> 100% of capital, scale balance accordingly
"""


class Backtester:
    def __init__(
        self, 
        data_dict: Dict[str, pd.DataFrame], 
        weights: pd.DataFrame,
        initial_balance: float = 10000, 
        trading_fee: float = 0.0, # 0.001
        leverage: float = 1,
        start_date: datetime.datetime = None,
        end_date: datetime.datetime = None
        ):
        """
        close_prices: Dict[str, pd.DataFrame] with 'Close' prices.
        weights: DataFrame with weights for each asset.
        initial_balance: Starting capital.
        trading_fee: Trading cost per transaction.
        leverage: Maximum leverage allowed.
        """
        self.data_dict = data_dict  # mat of the close prices of all pairs 
        self.weights = weights
        self.initial_balance = initial_balance
        self.trading_fee = trading_fee
        self.leverage = leverage
        self.results = None

        # match indices of close_prices and weights if not already aligned
        if self.data_dict['close'].index.equals(self.weights.index):
            print("Indices of close_prices and weights already aligned.")
        else:
            print("Indices of close_prices and weights not aligned. Aligning...")
            self.data_dict['close'] = self.data_dict['close'].reindex(self.weights.index)
    
    def run_backtest(self):
        df = self.data_dict
        df['returns'] = df['close'].pct_change()
        
        # Calculate strategy returns based on weights
        df['strategy returns'] = (df['returns'] * self.weights.shift(1)).sum(axis=1) * self.leverage
        
        # Calculate actual portfolio weights after market movements (before rebalancing)
        drifted_weights = pd.DataFrame(index=self.weights.index, columns=self.weights.columns)
        
        for t in range(1, len(self.weights)):
            prev_weights = self.weights.iloc[t-1]
            asset_returns = df['returns'].iloc[t]
            
            # Calculate how weights drift due to asset performance
            portfolio_return = (prev_weights * asset_returns).sum()
            drifted_weights.iloc[t] = prev_weights * (1 + asset_returns) / (1 + portfolio_return)
        
        # True turnover: difference between target weights and drifted weights
        df['turnover'] = (self.weights - drifted_weights).abs().sum(axis=1)
        
        # Account for trading costs (based on actual turnover)
        df['trades'] = df['turnover']
        df['strategy returns'] -= df['trades'] * self.trading_fee
        
        # Compute equity curve
        df['equity'] = (1 + df['strategy returns']).cumprod() * self.initial_balance
        
        self.results = df
        return df
    
    def performance_metrics(self):
        if self.results is None:
            print("Run backtest first.")
            return None
        
        df = self.results.copy()
        
        # Infer data frequency and calculate annualization factor
        if len(df['returns'].index) > 1:
            # Calculate median time delta between consecutive timestamps
            time_deltas = pd.Series(df['returns'].index[1:]) - pd.Series(df['returns'].index[:-1])
            median_delta = time_deltas.median()
            
            # Determine annualization factor based on frequency
            if median_delta <= pd.Timedelta(minutes=60):  # Hourly or less
                minutes_per_bar = median_delta.total_seconds() / 60
                annualization_factor = (60 * 24 * 365) / minutes_per_bar
            elif median_delta <= pd.Timedelta(days=1):  # Daily
                annualization_factor = 365
            elif median_delta <= pd.Timedelta(days=7):  # Weekly
                annualization_factor = 52
            elif median_delta <= pd.Timedelta(days=31):  # Monthly
                annualization_factor = 12
            else:  # Quarterly or less frequent
                annualization_factor = 4
        else:
            # Default to daily if we can't infer
            annualization_factor = 365
            
        total_return = df['equity'].iloc[-1] / self.initial_balance - 1
        annualized_return = (1 + total_return) ** (annualization_factor / len(df)) - 1
        volatility = df['strategy returns'].std() * np.sqrt(annualization_factor)
        sharpe_ratio = annualized_return / volatility if volatility != 0 else np.nan
        max_drawdown = ((df['equity'].cummax() - df['equity']) / df['equity'].cummax()).max()
        avg_turnover = df['turnover'].mean()
        annualized_turnover = avg_turnover * annualization_factor
        
        return {
            "Total Return": total_return,
            "Annualized Return": annualized_return,
            "Volatility": volatility,
            "Sharpe Ratio": sharpe_ratio,
            "Max Drawdown": max_drawdown,
            "Average Daily Turnover": avg_turnover,
            "Annualized Turnover": annualized_turnover
        }
    
    def plot_results(self):
        if self.results is None:
            print("Run backtest first.")
            return
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.results['equity'], label='equity Curve', color='blue')
        plt.title('Backtest Performance')
        plt.xlabel('Time')
        plt.ylabel('equity ($)')
        plt.legend()
        plt.show()

# Example Usage:
if __name__ == "__main__":
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
    prices = np.cumprod(1 + np.random.randn(365) * 0.01) * 1000  # Simulated price data
    weights = pd.DataFrame({'Weight': np.random.rand(365)})  # Random weights
    
    data = pd.DataFrame({'Close': prices}, index=dates)
    
    backtester = Backtester(data, weights)
    backtester.run_backtest()
    metrics = backtester.performance_metrics()
    print(metrics)
    backtester.plot_results()
