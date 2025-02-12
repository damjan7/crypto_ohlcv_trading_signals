import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import lightgbm as lgb
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from data_processor import create_feature_dataset

class ModelTrainer:
    def __init__(self, n_splits: int = 5, test_size: int = 1000):
        self.n_splits = n_splits
        self.test_size = test_size
        self.feature_importance = None
        self.performance_metrics = None
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        # Remove any future-looking features
        future_cols = [col for col in df.columns if 'future' in col.lower()]
        df = df.drop(columns=future_cols)
        
        # Create target (next period return)
        df['target'] = df['close'].shift(-1) / df['close'] - 1
        df['target'] = np.where(df['target'] > 0, 1, 0)
        
        # Drop rows with NaN values (caused by rolling windows)
        df = df.dropna()
        
        # Select features and target
        feature_cols = [col for col in df.columns if col not in ['target', 'timestamp', 'close', 'open', 'high', 'low']]
        X = df[feature_cols]
        y = df['target']
        
        return X, y
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
        return X_train_scaled, X_test_scaled
    
    def train_evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        tscv = TimeSeriesSplit(n_splits=self.n_splits, test_size=self.test_size)
        metrics = []
        feature_importance_list = []
        
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9
        }
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Scale features
            X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
            
            # Train model
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            fold_metrics = {
                'fold': fold,
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred)
            }
            metrics.append(fold_metrics)
            
            # Feature importance
            importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            })
            feature_importance_list.append(importance)
        
        # Aggregate results
        self.performance_metrics = pd.DataFrame(metrics)
        self.feature_importance = pd.concat(feature_importance_list).groupby('feature')['importance'].mean().sort_values(ascending=False)
        
        return {
            'metrics': self.performance_metrics,
            'feature_importance': self.feature_importance
        }
    
    def plot_results(self):
        # Performance metrics plot
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.performance_metrics.melt(id_vars=['fold'], 
                                                      value_vars=['precision', 'recall', 'f1']))
        plt.title('Model Performance Metrics Across Folds')
        plt.show()
        
        # Feature importance plot
        plt.figure(figsize=(12, 8))
        self.feature_importance.head(20).plot(kind='barh')
        plt.title('Top 20 Feature Importance')
        plt.show()

def main():
    # Get feature dataset
    feature_df = create_feature_dataset()
    
    # Initialize trainer
    trainer = ModelTrainer(n_splits=5, test_size=1000)
    
    # Prepare features
    X, y = trainer.prepare_features(feature_df)
    
    # Train and evaluate
    results = trainer.train_evaluate(X, y)
    
    # Plot results
    trainer.plot_results()
    
    # Print summary metrics
    print("\nAverage Performance Metrics:")
    print(results['metrics'].mean())
    
    print("\nTop 10 Most Important Features:")
    print(results['feature_importance'].head(10))

if __name__ == "__main__":
    main()
