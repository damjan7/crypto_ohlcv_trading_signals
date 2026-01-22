import pandas as pd
import numpy as np
from typing import Dict, Optional
from strategies.base import BaseStrategy
from .metrics import calculate_performance_metrics

from dataclasses import dataclass, field

@dataclass
class BacktestConfig:
    initial_balance: float = 10000.0
    trading_fee: float = 0.0
    leverage: float = 1.0
    slippage: float = 0.0      
    position_size: float = 1.0 

class BacktestEngine:
    def __init__(
        self, 
        data_dict: Dict[str, pd.DataFrame], 
        strategy: BaseStrategy = None,
        config: BacktestConfig = BacktestConfig()
    ):
        """
        data_dict: Dictionary containing 'close' prices df and other feature dfs.
                   Must contain at least 'close'.
        """
        self.data_dict = data_dict
        self.strategy = strategy
        self.config = config
        self.results = None
        self.weights = None

    def run(self, weights: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Run the backtest.
        Can provide weights directly or let the strategy generate them.
        """
        if weights is not None:
            self.weights = weights
        elif self.strategy is not None:
            print("Generating signals...")
            self.weights = self.strategy.generate_signals(self.data_dict)
            print("Signals generated.")
        else:
            raise ValueError("No strategy or weights provided.")

        df_close = self.data_dict.get('close')
        if df_close is None:
             raise ValueError("Data dictionary must contain 'close' prices.")

        # Align indices
        common_index = df_close.index.intersection(self.weights.index)
        df_close = df_close.loc[common_index]
        self.weights = self.weights.loc[common_index]

        # Calculate returns
        returns = df_close.pct_change()
        
        # Strategy returns
        # Shift weights by 1 because weights determined at t are for returns at t+1
        strategy_returns = (returns * self.weights.shift(1)).sum(axis=1) * self.config.leverage
        
        # Turnover & Cost
        drifted_weights = pd.DataFrame(index=self.weights.index, columns=self.weights.columns)
        
        print("Calculating portfolio drift and turnover...")
        
        drifted_weights_list = []
        
        self.weights = self.weights.fillna(0)
        
        # Loop
        for i in range(1, len(self.weights)):
            # Weights desired at t-1 (for period t)
            target_weights_prev = self.weights.iloc[i-1] 
            
            # Asset returns at t
            asset_rets = returns.iloc[i]
            
            # Portfolio return at t
            port_ret = (target_weights_prev * asset_rets).sum()
            
            # Drifted weights at t (before rebalancing to target_weights_t)
            # IF port_ret is -1 (bust), handle
            if port_ret == -1:
                 w_drifted = 0 # Bust
            else:
                 w_drifted = target_weights_prev * (1 + asset_rets) / (1 + port_ret)
            
            drifted_weights_list.append(w_drifted)
            
        # Add first row
        drifted_weights = pd.concat([pd.DataFrame([np.zeros(len(self.weights.columns))], columns=self.weights.columns, index=[self.weights.index[0]]), 
                                     pd.DataFrame(drifted_weights_list, index=self.weights.index[1:], columns=self.weights.columns)])

        # Turnover
        turnover = (self.weights - drifted_weights).abs().sum(axis=1)
        
        # Transaction Costs
        transaction_costs = turnover * self.config.trading_fee
        
        # Subtract costs
        strategy_returns_after_costs = strategy_returns - transaction_costs
        
        # Equity curve
        equity_curve = (1 + strategy_returns_after_costs).cumprod() * self.config.initial_balance
        
        # Store comprehensive results
        # We start with a DataFrame indexed by time
        self.results = pd.DataFrame(index=self.weights.index)
        self.results['equity'] = equity_curve
        self.results['strategy_returns'] = strategy_returns  # Gross returns
        self.results['strategy_returns_after_costs'] = strategy_returns_after_costs # Net returns
        self.results['turnover'] = turnover
        self.results['transaction_costs'] = transaction_costs
        
        # We can also store the weights if needed, but they are multi-dimensional (time x asset)
        # Usually kept separate or in a separate object property like self.weights
        # self.weights and self.data_dict are already accessible on the object.
        
        return self.results

    def get_performance_metrics(self) -> Dict[str, float]:
        if self.results is None:
            return {}
        return calculate_performance_metrics(self.results['equity'], self.results['strategy_returns_after_costs'], self.results['turnover'], self.config.initial_balance)
