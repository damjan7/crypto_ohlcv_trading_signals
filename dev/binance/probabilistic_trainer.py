import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

class ProbabilisticModelTrainer:
    def __init__(
        self,
        return_horizons: Dict[str, int] = {
            '30m': 1,
            '60m': 2,
            '12h': 24,
            '1d': 48
        },
        return_thresholds: List[float] = [0.0, 0.001, 0.002, 0.005],
        n_splits: int = 5,
        test_size: int = 1000
    ):
        self.return_horizons = return_horizons
        self.return_thresholds = return_thresholds
        self.n_splits = n_splits
        self.test_size = test_size
        self.models = {}
        
    def create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        targets = {}
        
        for horizon_name, periods in self.return_horizons.items():
            future_returns = df['close'].shift(-periods) / df['close'] - 1
            
            for threshold in self.return_thresholds:
                target_name = f'target_{horizon_name}_{threshold}'
                targets[target_name] = (future_returns > threshold).astype(int)
        
        return pd.DataFrame(targets)

    def prepare_features(self, df: pd.DataFrame):
        df = df.set_index('Date')
        df = df.sort_index()
        
        # Create all target variables
        targets = self.create_targets(df)
        
        # Select features
        feature_cols = [col for col in df.columns if col not in ['Date', 'close', 'open', 'high', 'low', 'volume']]
        X = df[feature_cols]
        
        return X, targets

    def train_models(self, X: pd.DataFrame, targets: pd.DataFrame):
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9
        }
        
        for target_col in targets.columns:
            print(f"Training model for {target_col}")
            y = targets[target_col].dropna()
            X_subset = X.loc[y.index]
            
            model = lgb.LGBMClassifier(**params)
            model.fit(X_subset, y)
            self.models[target_col] = model
            
    def predict_probabilities(self, X: pd.DataFrame) -> pd.DataFrame:
        predictions = {}
        
        for target_name, model in self.models.items():
            prob = model.predict_proba(X)[:, 1]
            predictions[target_name] = prob
            
        return pd.DataFrame(predictions, index=X.index)
    
    def evaluate_predictions(self, predictions: pd.DataFrame, targets: pd.DataFrame):
        evaluation = {}
        
        for col in predictions.columns:
            pred_probs = predictions[col]
            actual = targets[col]
            
            # Calculate metrics for different probability thresholds
            thresholds = np.arange(0.5, 1.0, 0.1)
            thresh_metrics = []
            
            for thresh in thresholds:
                pred_binary = (pred_probs > thresh).astype(int)
                accuracy = (pred_binary == actual).mean()
                precision = (pred_binary & actual).sum() / pred_binary.sum()
                
                thresh_metrics.append({
                    'threshold': thresh,
                    'accuracy': accuracy,
                    'precision': precision
                })
                
            evaluation[col] = pd.DataFrame(thresh_metrics)
            
        return evaluation

def main():
    # Load your data
    feature_df = create_feature_dataset()  # Your existing function
    
    # Initialize trainer
    trainer = ProbabilisticModelTrainer(
        return_horizons={'30m': 1, '60m': 2, '12h': 24, '1d': 48},
        return_thresholds=[0.0, 0.001, 0.002, 0.005]
    )
    
    # Prepare features and targets
    X, targets = trainer.prepare_features(feature_df)
    
    # Train models
    trainer.train_models(X, targets)
    
    # Make predictions
    predictions = trainer.predict_probabilities(X)
    
    # Evaluate
    evaluation = trainer.evaluate_predictions(predictions, targets)
    
    # Plot results for each target
    for target_name, eval_df in evaluation.items():
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=eval_df, x='threshold', y='precision')
        plt.title(f'Precision vs Probability Threshold for {target_name}')
        plt.show()

if __name__ == "__main__":
    main()


# Get predictions for new data
new_data = your_new_data
predictions = trainer.predict_probabilities(new_data)

# Example: Find opportunities with high probability of 0.5% return in 1 hour
opportunities = predictions[predictions['target_60m_0.005'] > 0.8]
