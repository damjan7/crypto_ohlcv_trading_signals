import pandas as pd
import numpy as np
from typing import Dict

def calculate_performance_metrics(equity_curve: pd.Series, strategy_returns: pd.Series, turnover: pd.Series = None, initial_balance: float = 10000) -> Dict[str, float]:
    """
    Calculate performance metrics for a backtest.
    """
    df_returns = strategy_returns
    
    # Infer data frequency and calculate annualization factor
    if len(df_returns.index) > 1:
        time_deltas = pd.Series(df_returns.index[1:]) - pd.Series(df_returns.index[:-1])
        median_delta = time_deltas.median()
        
        if median_delta <= pd.Timedelta(minutes=60):
            minutes_per_bar = median_delta.total_seconds() / 60
            annualization_factor = (60 * 24 * 365) / minutes_per_bar
        elif median_delta <= pd.Timedelta(days=1):
            annualization_factor = 365
        elif median_delta <= pd.Timedelta(days=7):
            annualization_factor = 52
        elif median_delta <= pd.Timedelta(days=31):
            annualization_factor = 12
        else:
            annualization_factor = 4
    else:
        annualization_factor = 365
        
    total_return = equity_curve.iloc[-1] / initial_balance - 1
    annualized_return = (1 + total_return) ** (annualization_factor / len(df_returns)) - 1
    volatility = df_returns.std() * np.sqrt(annualization_factor)
    sharpe_ratio = annualized_return / volatility if volatility != 0 else np.nan
    max_drawdown = ((equity_curve.cummax() - equity_curve) / equity_curve.cummax()).max()
    
    metrics = {
        "Total Return": total_return,
        "Annualized Return": annualized_return,
        "Volatility": volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
    }

    if turnover is not None:
         avg_turnover = turnover.mean()
         annualized_turnover = avg_turnover * annualization_factor
         metrics["Average Daily Turnover"] = avg_turnover #it's per-bar turnover actually
         metrics["Annualized Turnover"] = annualized_turnover

    return metrics
