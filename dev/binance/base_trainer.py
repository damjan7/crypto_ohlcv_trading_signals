
import numpy as np
import pandas as pd


from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit

'''
Below is a first infrastructure for Model Training and Evaluation

A BaseModel Wrapper:
A simple class that wraps a scikit‑learn model and “remembers” whether it’s 
being used for a regression (deterministic) or classification (probabilistic) task. 
This wrapper exposes a common interface for fitting and predicting.

A Model Factory:
A function that instantiates models based on a name string (or identifier) so 
that you can quickly switch between, say, a linear regression and a random forest 
regressor for continuous targets or between logistic regression and random forest 
classifiers for probabilistic outputs.

A Model Trainer Class:
An example “trainer” that takes your training/test split (made with time‑based 
splits to avoid lookahead bias) and uses time‑series cross-validation. It shows 
how you might evaluate the model using appropriate metrics (mean squared error 
for regression or ROC AUC for classification).

A Pipeline Example:
An optional pipeline (using scikit‑learn’s Pipeline) to include a feature scaling 
step before modeling. This helps ensure that your features are treated consistently.

'''


# =============================================================================
# 1. Base Model Wrapper
# =============================================================================
class BaseModel:
    """
    A simple wrapper for scikit-learn models to provide a unified interface.
    Parameters:
        model: A scikit-learn estimator.
        model_type: A string, either 'regression' or 'classification'.

    --> unifies the interface for regression and classification models
    and adds some type safety checks.
    """
    def __init__(self, model, model_type):
        self.model = model
        self.model_type = model_type
        
    def fit(self, X, y):
        self.model.fit(X, y) # since 'self' is an instance of sklearn.base.BaseModel we know that it has a fit method
        
    def predict(self, X):
        """
        For regression, returns predictions.
        For classification, returns probabilities for the positive class.
        """
        if self.model_type == 'regression':
            return self.model.predict(X)
        elif self.model_type == 'classification':
            # For classifiers that provide probabilities, return probability of class 1.
            return self.model.predict_proba(X)[:, 1]
        else:
            raise ValueError("Invalid model type specified.")
    
    def predict_class(self, X):
        """
        Returns class labels for classification models.
        """
        if self.model_type == 'classification':
            return self.model.predict(X)
        else:
            raise ValueError("predict_class is only valid for classification models.")


# =============================================================================
# Helper function to initialize BaseModel instances
# =============================================================================
def get_model(model_name: str, **kwargs):
    """
    Factory method to create model instances.
    Parameters:
        model_name: A string identifier for the model type.
        kwargs: Additional keyword arguments for the model.
    Returns:
        An instance of BaseModel.
    """
    if model_name == 'linear_regression':
        return BaseModel(LinearRegression(**kwargs), model_type='regression')
    elif model_name == 'ridge_regression':
        return BaseModel(Ridge(**kwargs), model_type='regression')
    elif model_name == 'random_forest_regression':
        return BaseModel(RandomForestRegressor(**kwargs), model_type='regression')
    elif model_name == 'logistic_regression':
        return BaseModel(LogisticRegression(**kwargs), model_type='classification')
    elif model_name == 'ridge_classification':
        return BaseModel(RidgeClassifier(**kwargs), model_type='classification')
    elif model_name == 'random_forest_classification':
        return BaseModel(RandomForestClassifier(**kwargs), model_type='classification')
    else:
        raise ValueError("Unknown model name: {}".format(model_name))


# =============================================================================
# 3. Model Trainer Class
# =============================================================================
class ModelTrainer:
    """
    Encapsulates training, cross-validation, and evaluation of a model.
    
    Parameters:
        model: An instance of BaseModel.
        X_train: Training features (pandas DataFrame or array).
        y_train: Training target (pandas Series or array).
        X_test: Test features.
        y_test: Test target.
    """
    def __init__(self, model: BaseModel, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
    def train(self):
        self.model.fit(self.X_train, self.y_train)
    
    def predict(self):
        return self.model.predict(self.X_test)
    
    def evaluate(self):
        preds = self.model.predict(self.X_test)
        if self.model.model_type == 'regression':
            mse = mean_squared_error(self.y_test, preds)
            print("Mean Squared Error:", mse)
            return mse
        elif self.model.model_type == 'classification':
            # Using ROC AUC as an example metric for classification.
            roc_auc = roc_auc_score(self.y_test, preds)
            precision = precision_score(self.y_test, preds)
            recall = recall_score(self.y_test, preds)
            f1 = f1_score(self.y_test, preds)
            return {'precision': precision, 'recall': recall, 
                    'f1': f1, 'roc_auc': roc_auc}
        else:
            raise ValueError("Invalid model type for evaluation.")
            
    def cross_validate(self, cv_splits=5):
        """
        Performs time-series cross-validation using TimeSeriesSplit.
        """
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        scores = []
        for train_index, test_index in tscv.split(self.X_train):
            X_cv_train = self.X_train.iloc[train_index]
            X_cv_test = self.X_train.iloc[test_index]
            y_cv_train = self.y_train.iloc[train_index]
            y_cv_test = self.y_train.iloc[test_index]
            
            self.model.fit(X_cv_train, y_cv_train)
            preds = self.model.predict(X_cv_test)
            
            if self.model.model_type == 'regression':
                score = mean_squared_error(y_cv_test, preds)
            elif self.model.model_type == 'classification':
                score = roc_auc_score(y_cv_test, preds)
            scores.append(score)
        
        avg_score = np.mean(scores)
        print("Cross-validated score:", avg_score)
        return avg_score
    
    def get_feature_importance(self):
        if hasattr(self.model.model, 'feature_importances_'):
            return pd.Series(
                self.model.model.feature_importances_,
                index=self.X_train.columns
            ).sort_values(ascending=False)
        elif hasattr(self.model.model, 'coef_'):
            return pd.Series(
                self.model.model.coef_[0],
                index=self.X_train.columns
            ).sort_values(ascending=False)
    
    def calculate_trading_metrics(self, initial_capital=10000):
        predictions = self.model.predict(self.X_test)
        returns = self.y_test * predictions  # Assuming y_test contains actual returns
        cumulative_returns = (1 + returns).cumprod()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)  # Annualized
        max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_return': cumulative_returns[-1] - 1
    }
    
    def save_model(self, filepath):
        import joblib
        joblib.dump(self.model, filepath)


# =============================================================================
# 4. (Optional) Pipeline Example for Scaling and Modeling
# =============================================================================
def create_pipeline(model: BaseModel):
    """
    Creates a scikit-learn Pipeline that includes standard scaling and the model.
    
    Returns:
        A new BaseModel instance wrapping the pipeline.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('estimator', model.model)  # the underlying estimator
    ])
    return BaseModel(pipeline, model.model_type)


# =============================================================================
# Example Usage
# =============================================================================
if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    # Assume df is your time-series DataFrame that includes a 'Date' column,
    # feature columns, and a 'target' column that was created using the previous
    # target creation function.
    #
    # It is essential that your DataFrame is sorted by 'Date' to avoid lookahead bias.
    # -----------------------------------------------------------------------------
    
    # Example: df = df.sort_values('Date')
    
    # Split data based on time (for instance, using a cutoff date).
    train_end_date = pd.to_datetime("2024-01-01")
    df_train_set = df[df['Date'] < train_end_date].copy()
    df_test_set = df[df['Date'] >= train_end_date].copy()
    
    # Define your feature columns. (Exclude 'Date' and 'target')
    feature_columns = [col for col in df_train_set.columns if col not in ['Date', 'target']]
    X_train = df_train_set[feature_columns]
    y_train = df_train_set['target']
    X_test = df_test_set[feature_columns]
    y_test = df_test_set['target']
    
    # ----------------------------
    # Deterministic Model Example:
    # Predicting future returns with regression.
    # ----------------------------
    # Choose a regression model (e.g., linear regression)
    reg_model_instance = get_model('linear_regression')
    # Optionally, create a pipeline that scales the features first.
    reg_pipeline = create_pipeline(reg_model_instance)
    
    # Instantiate the trainer and train/evaluate.
    reg_trainer = ModelTrainer(reg_pipeline, X_train, y_train, X_test, y_test)
    reg_trainer.train()
    reg_trainer.evaluate()
    reg_trainer.cross_validate(cv_splits=5)
    
    # ----------------------------
    # Probabilistic Model Example:
    # Predicting if the future return exceeds a threshold using classification.
    # ----------------------------
    # Choose a classification model (e.g., logistic regression)
    clf_model_instance = get_model('logistic_regression', solver='liblinear')
    # Create a pipeline for the classifier.
    clf_pipeline = create_pipeline(clf_model_instance)
    
    # Instantiate the trainer for the classification task.
    clf_trainer = ModelTrainer(clf_pipeline, X_train, y_train, X_test, y_test)
    clf_trainer.train()
    clf_trainer.evaluate()
    clf_trainer.cross_validate(cv_splits=5)


