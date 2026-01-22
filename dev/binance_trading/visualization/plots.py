import matplotlib.pyplot as plt
import pandas as pd

def plot_backtest_results(results: pd.DataFrame, title: str = 'Backtest Performance'):
    """
    Plot equity curve and potentially other metrics.
    """
    if results is None or 'equity' not in results.columns:
        print("No results to plot.")
        return
    
    plt.figure(figsize=(12, 6))
    plt.plot(results.index, results['equity'], label='Equity Curve', color='blue')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Equity ($)')
    plt.legend()
    plt.grid(True)
    plt.show()
