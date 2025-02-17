import datetime
import numpy as np
import pandas as pd
import queue
from typing import Dict, List, Optional
from dataclasses import dataclass

from signal_generator import SignalGenerator, SignalConfig
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


class CryptoBacktester:
    def __init__(
        self, 
        data: pd.DataFrame, 
        signals: pd.Series, 
        initial_balance: float = 10000, 
        trading_fee: float = 0.001, 
        leverage: float = 1,
        start_date: datetime.datetime = None,
        end_date: datetime.datetime = None
        ):
        """
        data: DataFrame with 'Close' prices.
        signals: Series with trading signals (-1, 0, 1), is calculated by the signal generator (needs to be implemented).
        initial_balance: Starting capital.
        trading_fee: Trading cost per transaction.
        leverage: Maximum leverage allowed.
        """
        self.data = data
        self.signals = signals
        self.initial_balance = initial_balance
        self.trading_fee = trading_fee
        self.leverage = leverage
        self.results = None
    
    def run_backtest(self):
        df = self.data.copy()
        df['Signal'] = self.signals
        df['Returns'] = df['Close'].pct_change()
        df['Strategy Returns'] = df['Returns'] * df['Signal'].shift(1) * self.leverage
        
        # Account for trading costs
        df['Trades'] = df['Signal'].diff().abs()  # A trade occurs when the signal changes
        df['Strategy Returns'] -= df['Trades'] * self.trading_fee
        
        # Compute equity curve
        df['Equity'] = (1 + df['Strategy Returns']).cumprod() * self.initial_balance
        
        self.results = df
        return df
    
    def performance_metrics(self):
        if self.results is None:
            print("Run backtest first.")
            return None
        
        df = self.results.copy()
        total_return = df['Equity'].iloc[-1] / self.initial_balance - 1
        annualized_return = (1 + total_return) ** (252 / len(df)) - 1  # Assuming daily data
        volatility = df['Strategy Returns'].std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility != 0 else np.nan
        max_drawdown = ((df['Equity'].cummax() - df['Equity']) / df['Equity'].cummax()).max()
        
        return {
            "Total Return": total_return,
            "Annualized Return": annualized_return,
            "Volatility": volatility,
            "Sharpe Ratio": sharpe_ratio,
            "Max Drawdown": max_drawdown
        }
    
    def plot_results(self):
        if self.results is None:
            print("Run backtest first.")
            return
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.results['Equity'], label='Equity Curve', color='blue')
        plt.title('Backtest Performance')
        plt.xlabel('Time')
        plt.ylabel('Equity ($)')
        plt.legend()
        plt.show()

# Example Usage:
if __name__ == "__main__":
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
    prices = np.cumprod(1 + np.random.randn(365) * 0.01) * 1000  # Simulated price data
    signals = np.random.choice([-1, 0, 1], size=365, p=[0.3, 0.4, 0.3])  # Random long/short signals
    
    data = pd.DataFrame({'Close': prices}, index=dates)
    signal_df = pd.Series(signals, index=dates)
    
    backtester = CryptoBacktester(data, signal_df)
    backtester.run_backtest()
    metrics = backtester.performance_metrics()
    print(metrics)
    backtester.plot_results()
