"""
Loads single features  and combines them into a single signals.
Since we are creating signals in this file, we need to create the feature dataset for all pairs of interest.

TODO:
- normalize features

first easy method:
- create z-scores and simply combine via averaging. Then use simple sorting portfolios
- implement something weighting rules according to place in the z-score distribution
"""

import data_processor as dp
import pandas as pd
import numpy as np
from typing import List, Dict
import datetime


@dataclass
class SignalConfig:
    """Configuration parameters for signal generation"""
    window_size: int = 16  # Default window size for rolling calculations
    step_size: int = 4     # Default step size for rolling calculations
    min_periods: int = 1   # Minimum number of observations required for calculation
    top_n: int = 5         # Number of top assets to go long
    bottom_n: int = 5      # Number of bottom assets to go short


class SignalClass(ABC):
    """
    Abstract base class for signal generation.
    """
    
    def __init__(self, config: SignalConfig = SignalConfig()):
        """
        Initialize the signal generator with configuration parameters.
        Args:
            config: Configuration parameters for signal generation
        """
        self.config = config
        
    @abstractmethod
    def validate_features(self, feature_dict: Dict[str, pd.DataFrame]) -> bool:
        """
        Validate that the required features are present in the feature dictionary.
        Args:
            feature_dict: Dictionary of feature matrices
        Returns:
            bool: True if all required features are present, False otherwise
        Raises:
            ValueError: If required features are missing
        """
        pass
    
    @abstractmethod
    def preprocess_features(self, feature_dict: Dict[str, pd.DataFrame], 
                           start_date: datetime.datetime, 
                           end_date: datetime.datetime) -> pd.DataFrame:
        """
        Preprocess features before signal generation.
        
        Args:
            feature_dict: Dictionary of feature matrices
            start_date: Start date for signal generation
            end_date: End date for signal generation
            
        Returns:
            pd.DataFrame: Preprocessed feature matrix
        """
        pass
    
    @abstractmethod
    def calculate_weights(self, processed_features: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate portfolio weights based on processed features.
        
        Args:
            processed_features: Preprocessed feature matrix
            
        Returns:
            pd.DataFrame: Portfolio weights
        """
        pass

    def inspect_feature_dict(feature_dict: Dict[str, pd.DataFrame]) -> None:
    """
    Print/log information about the feature dictionary.
    For example:
      - The earliest and latest timestamp per feature
      - The frequency or time deltas
      - The pairs (columns) each feature contains
    """
    for name, df in feature_dict.items():
        print(f"--- Feature: {name} ---")
        print(f"  Start: {df.index.min()}, End: {df.index.max()}")
        print(f"  Columns (pairs): {df.columns.tolist()}")
        # You could also calculate frequency heuristics, e.g. median time delta, etc.
        print()

    
    def generate_signal(self, feature_dict: Dict[str, pd.DataFrame], 
                       start_date: datetime.datetime, 
                       end_date: datetime.datetime) -> Dict[str, pd.DataFrame]:
        """
        Generate trading signals based on features.
        This method orchestrates the signal generation process by:
        1. Validating features
        2. Preprocessing features
        3. Calculating weights
        Args:
            feature_dict: Dictionary of feature matrices
            start_date: Start date for signal generation
            end_date: End date for signal generation
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of signal matrices
        """
        # inspect feature dict
        self.inspect_feature_dict(feature_dict)
                           
        # Validate features
        self.validate_features(feature_dict)
        
        # Preprocess features
        processed_features = self.preprocess_features(feature_dict, start_date, end_date)
        
        # Calculate weights
        weights = self.calculate_weights(processed_features)
        
        # Return signals
        return {"signal": weights}


class AverageFeatureSignal(SignalClass):
    """
    Signal generator that averages normalized features and creates long-short portfolios.
    
    This implementation is based on the test_signal_1 function, which:
    1. Averages normalized features
    2. Calculates rolling means and standard deviations
    3. Ranks assets based on scores
    4. Creates long-short portfolios based on ranks
    """
    
    def __init__(self, config: SignalConfig = SignalConfig(), 
                required_features: List[str] = None,
                signal_name: str = "8hr_signal"):
        """
        Initialize the signal generator.
        
        Args:
            config: Configuration parameters for signal generation
            required_features: List of required features
            signal_name: Name of the generated signal
        """
        super().__init__(config)
        
        # Default required features if none provided
        if required_features is None:
            self.required_features = [
                "return_1h", "volume_rel_ma24", "RSI_feature", 
                "price_rel_ma_6", "price_rel_ma_12", "volume_zscore_20"
            ]
        else:
            self.required_features = required_features
            
        self.signal_name = signal_name
    
    def validate_features(self, feature_dict: Dict[str, pd.DataFrame]) -> bool:
        """
        Validate that all required features are present in the feature dictionary.
        
        Args:
            feature_dict: Dictionary of feature matrices
            
        Returns:
            bool: True if all required features are present
            
        Raises:
            ValueError: If any required feature is missing
        """
        for feature_name in self.required_features:
            if feature_name not in feature_dict.keys():
                raise ValueError(f"Feature {feature_name} not found in feature_dict")
        return True
    
    def preprocess_features(self, feature_dict: Dict[str, pd.DataFrame], 
                           start_date: datetime.datetime, 
                           end_date: datetime.datetime) -> pd.DataFrame:
        """
        Preprocess features by averaging them and calculating rolling statistics.
        
        Args:
            feature_dict: Dictionary of feature matrices
            start_date: Start date for signal generation
            end_date: End date for signal generation
            
        Returns:
            pd.DataFrame: Preprocessed feature matrix with rolling means and standard deviations
        """
        # Average features
        for id, mat in enumerate(feature_dict.values()):
            if id == 0:
                aggregated_scores = mat.copy() / len(feature_dict.keys())
            else:
                aggregated_scores += mat.copy() / len(feature_dict.keys())
                
        # Drop rows with all NA's
        aggregated_scores.dropna(how="all", inplace=True)
        
        # Filter by date range
        aggregated_scores = aggregated_scores.loc[start_date:end_date]
        
        # Calculate rolling statistics
        # Average scores across time (8 hrs (16 30min periods) with 2 hrs overlap (4 30min periods))
        aggregated_scores_means = aggregated_scores.rolling(
            min_periods=self.config.min_periods, 
            window=self.config.window_size, 
            step=self.config.step_size
        ).mean()
        
        aggregated_scores_sds = aggregated_scores.rolling(
            min_periods=self.config.min_periods, 
            window=self.config.window_size, 
            step=self.config.step_size
        ).std()
        
        # Calculate z-scores
        aggregated_scores_zscores = (aggregated_scores_means - aggregated_scores_means.mean()) / aggregated_scores_sds
        
        # Calculate ranks
        aggregated_scores_ranks = aggregated_scores_means.rank(axis=1, method="min")
        
        return aggregated_scores_ranks
    
    def calculate_weights(self, processed_features: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate portfolio weights based on ranks.
        
        Args:
            processed_features: Preprocessed feature matrix with ranks
            
        Returns:
            pd.DataFrame: Portfolio weights
        """
        # Calculate weights matrix (same dimensions as processed_features)
        weights = pd.DataFrame(np.nan, index=processed_features.index, columns=processed_features.columns)
        
        # For each row, assign weights to top and bottom N
        for idx in weights.index:
            row_ranks = processed_features.loc[idx]
            
            # Identify top and bottom N positions
            top_n = row_ranks.nlargest(self.config.top_n).index
            bottom_n = row_ranks.nsmallest(self.config.bottom_n).index
            
            # Assign positive weights to top N (proportional to their ranks)
            top_ranks = row_ranks[top_n]
            weights.loc[idx, top_n] = top_ranks / top_ranks.sum()
            
            # Assign negative weights to bottom N (proportional to their ranks)
            bottom_ranks = row_ranks[bottom_n]
            weights.loc[idx, bottom_n] = -bottom_ranks / bottom_ranks.sum()
        
        return weights
    
    def generate_signal(self, feature_dict: Dict[str, pd.DataFrame], 
                       start_date: datetime.datetime, 
                       end_date: datetime.datetime) -> Dict[str, pd.DataFrame]:
        """
        Generate trading signals based on features.
        
        Args:
            feature_dict: Dictionary of feature matrices
            start_date: Start date for signal generation
            end_date: End date for signal generation
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of signal matrices
        """
        # Validate features
        self.validate_features(feature_dict)
        
        # Preprocess features
        processed_features = self.preprocess_features(feature_dict, start_date, end_date)
        
        # Calculate weights
        weights = self.calculate_weights(processed_features)
        
        # Return signals with custom signal name
        return {self.signal_name: weights}


class WeightedFeatureSignal(AverageFeatureSignal):
    """
    Signal generator that applies custom weights to features before averaging.
    
    This extends the AverageFeatureSignal class by allowing different weights
    to be applied to different features.
    """
    
    def __init__(self, config: SignalConfig = SignalConfig(), 
                required_features: List[str] = None,
                feature_weights: Dict[str, float] = None,
                signal_name: str = "weighted_signal"):
        """
        Initialize the weighted signal generator.
        
        Args:
            config: Configuration parameters for signal generation
            required_features: List of required features
            feature_weights: Dictionary mapping feature names to weights
            signal_name: Name of the generated signal
        """
        super().__init__(config, required_features, signal_name)
        self.feature_weights = feature_weights or {feature: 1.0 for feature in self.required_features}
        
        # Normalize weights to sum to 1
        total_weight = sum(self.feature_weights.values())
        self.feature_weights = {k: v/total_weight for k, v in self.feature_weights.items()}
    
    def preprocess_features(self, feature_dict: Dict[str, pd.DataFrame], 
                           start_date: datetime.datetime, 
                           end_date: datetime.datetime) -> pd.DataFrame:
        """
        Preprocess features by applying weights before averaging.
        
        Args:
            feature_dict: Dictionary of feature matrices
            start_date: Start date for signal generation
            end_date: End date for signal generation
            
        Returns:
            pd.DataFrame: Preprocessed feature matrix with rolling means and standard deviations
        """
        # Apply weights to features before averaging
        aggregated_scores = None
        
        for feature_name, mat in feature_dict.items():
            if feature_name in self.feature_weights:
                weight = self.feature_weights[feature_name]
                if aggregated_scores is None:
                    aggregated_scores = mat.copy() * weight
                else:
                    aggregated_scores += mat.copy() * weight
                
        # Drop rows with all NA's
        aggregated_scores.dropna(how="all", inplace=True)
        
        # Filter by date range
        aggregated_scores = aggregated_scores.loc[start_date:end_date]
        
        # Calculate rolling statistics
        aggregated_scores_means = aggregated_scores.rolling(
            min_periods=self.config.min_periods, 
            window=self.config.window_size, 
            step=self.config.step_size
        ).mean()
        
        aggregated_scores_sds = aggregated_scores.rolling(
            min_periods=self.config.min_periods, 
            window=self.config.window_size, 
            step=self.config.step_size
        ).std()
        
        # Calculate z-scores
        aggregated_scores_zscores = (aggregated_scores_means - aggregated_scores_means.mean()) / aggregated_scores_sds
        
        # Calculate ranks
        aggregated_scores_ranks = aggregated_scores_means.rank(axis=1, method="min")
        
        return aggregated_scores_ranks



# old implementation
def test_signal_1(normalized_dict: Dict[str, pd.DataFrame], start_date: datetime.datetime, end_date: datetime.datetime): 
    """
    #############################################################
    Note: IT IS HARD CODED FOR 30 MINUTES DATA!!!!!!!!!!!!!!!!!!!!
    #############################################################


    Example signal:
    parameters:
    - normalized_dict: dictionary of normalized feature matrices
    - start_date: start date of the signal
    - end_date: end date of the signal

    features:
    - "return_1h", "volume_rel_ma24", "RSI_feature", "price_rel_ma_6", "price_rel_ma_12", "volume_zscore_20"
    - higher is better for all these features, so can simply average the normalized features

    """
    for feature_name in ["return_1h", "volume_rel_ma24", "RSI_feature", "price_rel_ma_6", "price_rel_ma_12", "volume_zscore_20"]:
        if feature_name not in normalized_dict.keys():
            raise ValueError(f"Feature {feature_name} not found in normalized_dict")
        
    for id, mat in enumerate(normalized_dict.values()):
        if id == 0:
            aggregated_scores = mat.copy() / len(normalized_dict.keys())
        else:
            aggregated_scores += mat.copy() / len(normalized_dict.keys())
    # drop rows with all NA's
    aggregated_scores.dropna(how="all", inplace=True)

    # filter by date range
    aggregated_scores = aggregated_scores.loc[start_date:end_date]

    # average scores across time (8 hrs (16 30min periods) with 2 hrs overlap (4 30min periods))
    aggregated_scores_means = aggregated_scores.rolling(min_periods=1, window=16, step=4).mean()
    aggregated_scores_sds = aggregated_scores.rolling(min_periods=1, window=16, step=4).std()

    # calculate z-scores
    aggregated_scores_zscores = (aggregated_scores_means - aggregated_scores_means.mean()) / aggregated_scores_sds

    # calculate ranks
    aggregated_scores_ranks = aggregated_scores_means.rank(axis=1, method="min")

    #############################################################
    # calculate simple sorting portfolios
    # long top 5, short bottom 5, proportional to rank
    #############################################################
    # calculate weights matrix (same dimensions as aggregated_scores_ranks)
    weights = pd.DataFrame(np.nan, index=aggregated_scores_ranks.index, columns=aggregated_scores_ranks.columns)
    
    # for each row, assign weights to top and bottom 5
    for idx in weights.index:
        row_ranks = aggregated_scores_ranks.loc[idx]
        
        # identify top and bottom 5 positions
        top_5 = row_ranks.nlargest(5).index
        bottom_5 = row_ranks.nsmallest(5).index
        
        # assign positive weights to top 5 (proportional to their ranks)
        top_ranks = row_ranks[top_5]
        weights.loc[idx, top_5] = top_ranks / top_ranks.sum()
        
        # assign negative weights to bottom 5 (proportional to their ranks)
        bottom_ranks = row_ranks[bottom_5]
        weights.loc[idx, bottom_5] = -bottom_ranks / bottom_ranks.sum()
    
    return {"8hr_signal": weights}
            
            
