import datetime
import numpy as np
import pandas as pd
import queue
from typing import Dict, List, Optional
from dataclasses import dataclass

from signal_generator import SignalGenerator, SignalConfig
from base_trainer import ModelTrainer, get_model, create_pipeline
from data_processor import create_feature_dataset, create_target_dataset

@dataclass
class BacktestConfig:
    """Configuration for the backtesting engine"""
    initial_capital: float = 100000
    commission_rate: float = 0.001
    slippage: float = 0.0005
    position_size: float = 0.1  # Fraction of capital to risk per trade
    stop_loss: float = 0.02     # 2% stop loss
    take_profit: float = 0.04   # 4% take profit

class BacktestEngine:
    """
    Enhanced backtesting engine that integrates signal generation, 
    ML model predictions, and portfolio management.
    """
    def __init__(self, config: BacktestConfig = BacktestConfig()):
        self.config = config
        self.portfolio = Portfolio(config.initial_capital)
        self.trades = []
        self.signals = []
        self.performance_metrics = {}
        
        # Initialize event queue
        self.events = queue.Queue()
        
    def prepare_data(self, df: pd.DataFrame, train_start_date: datetime.datetime, train_end_date: datetime.datetime) -> Dict:
        """
        Prepares data for backtesting by creating features and splitting into train/test
        """
        feature_df, feature_names = create_feature_dataset(df)
        target_df, target_names = create_target_dataset(
            df, 
            horizon_lst=[1],  # For simplicity, predict next period only
            target_type='regression'
        )
        
        # Split data
        train_start_mask = feature_df.index >= train_start_date
        train_end_mask = feature_df.index < train_end_date
        test_mask = feature_df.index >= train_end_date
        
        return {
            'X_train': feature_df[train_start_mask & train_end_mask],
            'y_train': target_df[train_start_mask & train_end_mask]['target_regr_horizon1'],
            'X_test': feature_df[test_mask],
            'y_test': target_df[test_mask]['target_regr_horizon1'],
            'feature_names': feature_names,
            'target_names': target_names
        }

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> ModelTrainer:
        """Trains the ML model for signal generation"""
        model = get_model('random_forest_regression', 
                         n_estimators=100, 
                         random_state=42)
        pipeline = create_pipeline(model)
        
        trainer = ModelTrainer(pipeline, X_train, y_train, X_train, y_train)
        trainer.train()
        return trainer

    def run(self, df: pd.DataFrame, train_end_date: datetime.datetime):
        """
        Runs the backtest
        """
        # Prepare data and train model
        data = self.prepare_data(df, train_end_date)
        model_trainer = self.train_model(data['X_train'], data['y_train'])
        
        # Run backtest on test set
        test_data = data['X_test']
        for timestamp, features in test_data.iterrows():
            # Generate signals using ML model
            features_df = pd.DataFrame([features])
            prediction = model_trainer.model.predict(features_df)[0]
            
            # Create market event
            market_data = {
                'timestamp': timestamp,
                'price': df.loc[timestamp, 'close'],
                'prediction': prediction
            }
            self.process_market_update(market_data)
            
            # Process any pending orders
            self.execute_orders()
            
        # Calculate final performance metrics
        self.calculate_performance()

    def process_market_update(self, market_data: Dict):
        """Processes new market data and generates trading signals"""
        current_price = market_data['price']
        prediction = market_data['prediction']
        
        # Generate trading signal based on model prediction
        if prediction > 0.7:  # Strong buy signal
            self.generate_order('BUY', current_price)
        elif prediction < 0.3:  # Strong sell signal
            self.generate_order('SELL', current_price)
            
        # Update portfolio value
        self.portfolio.update_value(current_price)

    def generate_order(self, direction: str, price: float):
        """Generates a new order based on signal and position sizing"""
        position_size = self.config.position_size * self.portfolio.current_value
        quantity = position_size / price
        
        order = {
            'direction': direction,
            'quantity': quantity,
            'price': price,
            'stop_loss': price * (1 - self.config.stop_loss if direction == 'BUY' 
                                else 1 + self.config.stop_loss),
            'take_profit': price * (1 + self.config.take_profit if direction == 'BUY' 
                                  else 1 - self.config.take_profit)
        }
        self.events.put(('ORDER', order))

    def execute_orders(self):
        """Processes pending orders in the event queue"""
        while not self.events.empty():
            event_type, event_data = self.events.get()
            
            if event_type == 'ORDER':
                # Apply slippage
                executed_price = event_data['price'] * (
                    1 + self.config.slippage if event_data['direction'] == 'BUY'
                    else 1 - self.config.slippage
                )
                
                # Calculate commission
                commission = executed_price * event_data['quantity'] * self.config.commission_rate
                
                # Execute trade
                self.portfolio.execute_trade(
                    direction=event_data['direction'],
                    quantity=event_data['quantity'],
                    price=executed_price,
                    commission=commission
                )
                
                # Log trade
                self.trades.append({
                    'timestamp': datetime.datetime.now(),
                    'direction': event_data['direction'],
                    'quantity': event_data['quantity'],
                    'price': executed_price,
                    'commission': commission
                })

    def calculate_performance(self):
        """Calculates final performance metrics"""
        returns = pd.Series([t['price'] for t in self.trades]).pct_change()
        
        self.performance_metrics = {
            'total_return': self.portfolio.current_value / self.config.initial_capital - 1,
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 1 else 0,
            'max_drawdown': self.portfolio.max_drawdown,
            'total_trades': len(self.trades),
            'win_rate': sum(1 for t in self.trades if t['profit'] > 0) / len(self.trades) if self.trades else 0
        }

class Portfolio:
    """Manages portfolio positions and tracks performance"""
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.current_value = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.equity_curve = []
        self.max_drawdown = 0
        
    def update_value(self, current_price: float):
        """Updates portfolio value based on current market prices"""
        total_value = self.cash
        for symbol, position in self.positions.items():
            total_value += position['quantity'] * current_price
            
        self.current_value = total_value
        self.equity_curve.append(total_value)
        
        # Update max drawdown
        peak = max(self.equity_curve)
        current_drawdown = (peak - total_value) / peak
        self.max_drawdown = max(self.max_drawdown, current_drawdown)

    def execute_trade(self, direction: str, quantity: float, price: float, commission: float):
        """Executes a trade and updates portfolio state"""
        cost = quantity * price + commission
        
        if direction == 'BUY':
            if cost <= self.cash:
                self.cash -= cost
                self.positions['ASSET'] = {
                    'quantity': quantity,
                    'entry_price': price
                }
        else:  # SELL
            if 'ASSET' in self.positions:
                self.cash += quantity * price - commission
                del self.positions['ASSET']

if __name__ == '__main__':
    # Example usage
    config = BacktestConfig(
        initial_capital=100000,
        commission_rate=0.001,
        slippage=0.0005
    )
    
    backtest = BacktestEngine(config)
    
    # Load your data
    df = pd.read_parquet('path/to/your/data.parquet')
    train_end_date = datetime.datetime(2024, 1, 1)
    
    # Run backtest
    backtest.run(df, train_end_date)
    
    # Print results
    print("\nBacktest Results:")
    for metric, value in backtest.performance_metrics.items():
        print(f"{metric}: {value:.2%}")
