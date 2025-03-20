import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_return_series(returns, title="Return Series", figsize=(12, 6)):
    plt.figure(figsize=figsize)
    plt.plot(returns, label='Returns')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Returns')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_multiple_returns(returns_dict, title="Multiple Return Series", figsize=(12, 6)):
    plt.figure(figsize=figsize)
    for name, returns in returns_dict.items():
        plt.plot(returns, label=name)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Returns')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_cumulative_returns(returns, title="Cumulative Returns", figsize=(12, 6)):
    cumulative = (1 + returns).cumprod()
    plt.figure(figsize=figsize)
    plt.plot(cumulative, label='Cumulative Returns')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_predicted_vs_actual(actual, predicted, title="Predicted vs Actual Returns", figsize=(12, 6)):
    plt.figure(figsize=figsize)
    
    # Time series plot
    plt.subplot(2, 1, 1)
    plt.plot(actual, label='Actual', alpha=0.7)
    plt.plot(predicted, label='Predicted', alpha=0.7)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Returns')
    plt.legend()
    plt.grid(True)
    
    # Scatter plot
    plt.subplot(2, 1, 2)
    plt.scatter(actual, predicted, alpha=0.5)
    plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', label='Perfect Prediction')
    plt.xlabel('Actual Returns')
    plt.ylabel('Predicted Returns')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_return_distribution(returns, title="Return Distribution", figsize=(12, 6)):
    plt.figure(figsize=figsize)
    sns.histplot(returns, kde=True)
    plt.title(title)
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
