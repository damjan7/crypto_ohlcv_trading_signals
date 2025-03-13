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
from abc import ABC, abstractmethod
from dataclasses import dataclass


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

    def _get_z_score_features(self, feature_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Get z-score features from the feature dictionary.
        """
        z_score_feature_dict = {}
        for feature_name in feature_dict.keys():
            df = feature_dict[feature_name]
            df_demeaned = (df.sub(df.mean(axis=1), axis=0))
            df_standardized = (df_demeaned.div(df_demeaned.std(axis=1), axis=0))
            z_score_feature_dict[feature_name] = df_standardized
        return z_score_feature_dict

    def inspect_feature_dict(self, feature_dict: Dict[str, pd.DataFrame]) -> None:
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

    def align_weights_to_data(self, weights: pd.DataFrame, data_index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Align signal weights to match the frequency of the original data.
        
        When window_size and step_size are not 1, the weights matrix has fewer rows
        than the original data. This method creates a new DataFrame with the same index
        as the original data and fills it with the appropriate weights.
        
        Args:
            weights: The weights DataFrame with potentially fewer rows
            data_index: The DatetimeIndex from the original data to align with
            
        Returns:
            pd.DataFrame: Aligned weights with the same index as the original data
        """
        # Create a new DataFrame with the same index as the original data
        aligned_weights = pd.DataFrame(np.nan, index=data_index, columns=weights.columns)
        
        # Fill the aligned weights with values from the original weights
        aligned_weights = aligned_weights.fillna(weights)
        
        # Forward fill to propagate weights to timestamps between signal calculations
        aligned_weights = aligned_weights.ffill()
        
        return aligned_weights

    def reduce_turnover(self, weights: pd.DataFrame, returns_data: pd.DataFrame = None, threshold: float = 0.05) -> pd.DataFrame:
        """
        Reduce trading activity by only updating weights when they change significantly.
        
        This method compares each new target weight with the current drifted weight
        (accounting for asset performance) and only updates if the absolute difference 
        exceeds the threshold. This helps reduce transaction costs by avoiding small 
        rebalancing trades.
        
        Args:
            weights: The original target weights DataFrame
            returns_data: DataFrame of asset returns between rebalancing periods (optional)
            threshold: Minimum absolute change required to update a weight (default: 0.05 or 5%)
            
        Returns:
            pd.DataFrame: Modified weights with reduced trading activity
        """
        # Create a copy of the original weights
        to_reduced_weights = weights.copy()
        
        # If returns data is provided, we can account for weight drift
        if returns_data is not None:
            # Iterate through each row (timestamp) starting from the second row
            for i in range(1, len(to_reduced_weights)):
                prev_timestamp = to_reduced_weights.index[i-1]
                curr_timestamp = to_reduced_weights.index[i]
                
                # Get the previous actual weights
                prev_weights = to_reduced_weights.iloc[i-1].copy()
                
                # Calculate drifted weights based on returns between prev_timestamp and curr_timestamp
                # Get returns for the period between the two timestamps
                period_returns = returns_data.loc[prev_timestamp:curr_timestamp].iloc[1:].fillna(0)
                
                # If we have returns data for this period
                if not period_returns.empty:
                    # Calculate cumulative returns for the period
                    cum_returns = (1 + period_returns).prod() - 1
                    
                    # Calculate drifted weights
                    drifted_weights = prev_weights.copy()
                    portfolio_return = 0
                    
                    # First pass: calculate the portfolio return
                    for asset in drifted_weights.index:
                        if not pd.isna(drifted_weights[asset]) and asset in cum_returns.index:
                            portfolio_return += drifted_weights[asset] * cum_returns[asset]
                    
                    # Second pass: calculate the drifted weights
                    for asset in drifted_weights.index:
                        if not pd.isna(drifted_weights[asset]) and asset in cum_returns.index:
                            drifted_weights[asset] *= (1 + cum_returns[asset]) / (1 + portfolio_return)
                    
                    # Compare target weights with drifted weights
                    for asset in to_reduced_weights.columns:
                        target_weight = weights.iloc[i][asset]
                        drifted_weight = drifted_weights[asset] if asset in drifted_weights.index else 0
                        
                        # Skip if target weight is NaN
                        if pd.isna(target_weight):
                            continue
                        
                        # If the absolute difference is less than the threshold, keep the drifted weight
                        if abs(target_weight - drifted_weight) < threshold:
                            to_reduced_weights.iloc[i, to_reduced_weights.columns.get_loc(asset)] = drifted_weight
                else:
                    # If no returns data, fall back to simple comparison with previous weights
                    self._simple_weight_comparison(to_reduced_weights, i, threshold)
        else:
            # If no returns data provided, fall back to simple comparison with previous weights
            for i in range(1, len(to_reduced_weights)):
                self._simple_weight_comparison(to_reduced_weights, i, threshold)
        
        return to_reduced_weights

    def _simple_weight_comparison(self, weights_df: pd.DataFrame, row_idx: int, threshold: float) -> None:
        """
        Helper method for simple weight comparison without accounting for drift.
        
        Args:
            weights_df: DataFrame of weights to modify in-place
            row_idx: Index of the current row to process
            threshold: Minimum absolute change required to update a weight
        """
        prev_row = weights_df.iloc[row_idx-1]
        curr_row = weights_df.iloc[row_idx].copy()
        
        # For each asset, check if the weight change exceeds the threshold
        for asset in weights_df.columns:
            prev_weight = prev_row[asset]
            curr_weight = curr_row[asset]
            
            # Skip if either weight is NaN
            if pd.isna(prev_weight) or pd.isna(curr_weight):
                continue
            
            # If the absolute difference is less than the threshold, keep the previous weight
            if abs(curr_weight - prev_weight) < threshold:
                weights_df.iloc[row_idx, weights_df.columns.get_loc(asset)] = prev_weight

    def generate_signal(self, feature_dict: Dict[str, pd.DataFrame], 
                       start_date: datetime.datetime, 
                       end_date: datetime.datetime,
                       returns_data: pd.DataFrame = None) -> Dict[str, pd.DataFrame]:
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
            returns_data: DataFrame of asset returns for calculating weight drift (optional)
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
        
        # Create reduced trading version of weights
        reduced_weights = self.reduce_turnover(weights, returns_data)
        
        # Store original data index for alignment
        original_index = None
        for feature_name in self.required_features:
            if feature_name in feature_dict:
                original_index = feature_dict[feature_name].loc[start_date:end_date].index
                break
        
        # Create aligned weights if window_size or step_size is not 1
        if (self.config.window_size != 1 or self.config.step_size != 1) and original_index is not None:
            aligned_weights = self.align_weights_to_data(weights, original_index)
            aligned_reduced_weights = self.align_weights_to_data(reduced_weights, original_index)
            return {
                self.signal_name: weights,  # Original sparse weights
                f"{self.signal_name}_aligned": aligned_weights,  # Aligned dense weights
                f"{self.signal_name}_reduced": reduced_weights,  # Reduced trading sparse weights
                f"{self.signal_name}_reduced_aligned": aligned_reduced_weights  # Reduced trading aligned weights
            }
        
        # Return signals with custom signal name
        return {
            self.signal_name: weights,
            f"{self.signal_name}_reduced": reduced_weights
        }


class SimpleSortingLongOnly(SignalClass):
    """
    Simple Sorting based on a signle feature
    """

    def __init__(self, config: SignalConfig = SignalConfig(), 
                signal_name: str = "simple_sorting_signal",
                feature_name: str = "close",
                feature_dict: Dict[str, pd.DataFrame] = None,
                start_date: datetime.datetime = None,
                end_date: datetime.datetime = None,
                num_quantiles: int = 5):

        super().__init__(config)
        self.signal_name = signal_name
        self.feature_name = feature_name
        self.feature_dict = feature_dict
        self.start_date = start_date
        self.end_date = end_date
        self.num_quantiles = num_quantiles

        print(f"Initialized {self.__class__.__name__} with:")
        print(f"Feature name: {feature_name}, Window size: {config.window_size}, Step size: {config.step_size}, Min periods: {config.min_periods}, Num quantiles: {num_quantiles}")

    def validate_features(self) -> bool:
        """
        Validate that the required feature is present in the feature dictionary.
        """
        if self.feature_name not in self.feature_dict.keys():
            raise ValueError(f"Feature {self.feature_name} not found in feature_dict")
        return True

    def preprocess_features(self) -> pd.DataFrame:
        """
        Preprocess the feature.
        """
        feature_df = self.feature_dict[self.feature_name]
        feature_df = feature_df.loc[self.start_date:self.end_date]
        return feature_df

    def calculate_weights(self, processed_features: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the weights for the signal.
        Returns weights for all Quintile Portfolios.
        """
        # Initialize dictionary to store quintile weights
        quintile_weights = {}
        for quintile in range(self.num_quantiles):
            quintile_name = f"quintile_{quintile+1}"
            quintile_weights[quintile_name] = pd.DataFrame(0, index=processed_features.index, columns=processed_features.columns)
        
        # Calculate ranks for all dates at once
        all_ranks = processed_features.rank(axis=1, method="first", pct=True)
        
        # Define quintile breakpoints (0.0-0.2, 0.2-0.4, etc.)
        breakpoints = [i/self.num_quantiles for i in range(self.num_quantiles+1)]
        
        # Process all quintiles at once
        for quintile in range(self.num_quantiles):
            quintile_name = f"quintile_{quintile+1}"
            
            # Create mask for this quintile (e.g., ranks between 0.0-0.2 for quintile 1)
            lower_bound = breakpoints[quintile]
            upper_bound = breakpoints[quintile+1]
            
            # For the last quintile, include the upper bound to catch any rounding issues
            if quintile == self.num_quantiles - 1:
                quintile_mask = (all_ranks > lower_bound) & (all_ranks <= upper_bound)
            else:
                quintile_mask = (all_ranks > lower_bound) & (all_ranks <= upper_bound)
            
            # Count assets in each quintile for each date
            assets_per_date = quintile_mask.sum(axis=1)
            
            # Calculate weights (1/count for each asset in the quintile)
            for idx in processed_features.index:
                count = assets_per_date[idx]
                if count > 0:
                    quintile_weights[quintile_name].loc[idx, quintile_mask.loc[idx]] = 1 / count
        
        return quintile_weights

    def generate_signal(self) -> pd.DataFrame:
        """
        Generate the signal.
        """
        processed_features = self.preprocess_features()
        weights = self.calculate_weights(processed_features)
        return weights

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
                signal_name: str = "average_score_signal",
                feature_dict: Dict[str, pd.DataFrame] = None,
                start_date: datetime.datetime = None,
                end_date: datetime.datetime = None):
        """
        Initialize the signal generator.
        Args:
            config: Configuration parameters for signal generation
            required_features: List of required features
            signal_name: Name of the generated signal
            feature_dict: Dictionary of feature matrices
            start_date: Start date for signal generation
            end_date: End date for signal generation
        """
        super().__init__(config)
        print(f"Initialized {signal_name} with:")
        print(f"Window size: {config.window_size}, Step size: {config.step_size}, Min periods: {config.min_periods}, Top n: {config.top_n}, Bottom n: {config.bottom_n}")
        
        # Default required features if none provided
        if required_features is None:
            self.required_features = [
                "return_1h", "volume_rel_ma24", "RSI_feature", 
                "price_rel_ma_6", "price_rel_ma_12", "volume_zscore_20"
            ]
        else:
            self.required_features = required_features
            
        self.signal_name = signal_name
        self.feature_dict = feature_dict
        self.start_date = start_date
        self.end_date = end_date
    
    def validate_features(self) -> bool:
        """
        Validate that all required features are present in the feature dictionary.
        Returns:
            bool: True if all required features are present
        Raises:
            ValueError: If any required feature is missing
        """
        for feature_name in self.required_features:
            if feature_name not in self.feature_dict.keys():
                raise ValueError(f"Feature {feature_name} not found in feature_dict")
        return True
    
    def preprocess_features(self) -> pd.DataFrame:
        """
        Preprocess features by averaging them and calculating rolling statistics.
        Returns:
            pd.DataFrame: Preprocessed feature matrix with rolling means and standard deviations
        """
        # get simple z-score features
        z_score_feature_dict = self._get_z_score_features(self.feature_dict)

        # get simple z-score averaging
        # Average features
        for id, mat in enumerate(z_score_feature_dict.values()):
            if id == 0:
                aggregated_scores = mat.copy() / len(z_score_feature_dict.keys())
            else:
                aggregated_scores += mat.copy() / len(z_score_feature_dict.keys())
                
        # Drop rows with all NA's
        aggregated_scores.dropna(how="all", inplace=True)
        
        # Filter by date range
        aggregated_scores = aggregated_scores.loc[self.start_date:self.end_date]
        
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
    
    def generate_signal(self) -> Dict[str, pd.DataFrame]:
        """
        Generate trading signals based on features.
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of signal matrices
        """
        # Validate features
        self.validate_features()
        
        # Preprocess features
        processed_features = self.preprocess_features()
        
        # Calculate weights
        weights = self.calculate_weights(processed_features)
        
        # Store original data index for alignment
        original_index = None
        for feature_name in self.required_features:
            if feature_name in self.feature_dict:
                original_index = self.feature_dict[feature_name].loc[self.start_date:self.end_date].index
                break
        
        # Create aligned weights if window_size or step_size is not 1
        if (self.config.window_size != 1 or self.config.step_size != 1) and original_index is not None:
            aligned_weights = self.align_weights_to_data(weights, original_index)
            return {
                self.signal_name: weights,  # Original sparse weights
                f"{self.signal_name}_aligned": aligned_weights  # Aligned dense weights
            }
        
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
                signal_name: str = "weighted_signal",
                feature_dict: Dict[str, pd.DataFrame] = None,
                start_date: datetime.datetime = None,
                end_date: datetime.datetime = None):
        """
        Initialize the weighted signal generator.
        
        Args:
            config: Configuration parameters for signal generation
            required_features: List of required features
            feature_weights: Dictionary mapping feature names to weights
            signal_name: Name of the generated signal
            feature_dict: Dictionary of feature matrices
            start_date: Start date for signal generation
            end_date: End date for signal generation
        """
        super().__init__(config, required_features, signal_name, feature_dict, start_date, end_date)

        self.feature_weights = feature_weights or {feature: 1.0 for feature in self.required_features}
        
        # Normalize weights to sum to 1
        total_weight = sum(self.feature_weights.values())
        self.feature_weights = {k: v/total_weight for k, v in self.feature_weights.items()}
    
    def preprocess_features(self) -> pd.DataFrame:
        """
        Preprocess features by applying weights before averaging.
        
        Returns:
            pd.DataFrame: Preprocessed feature matrix with rolling means and standard deviations
        """
        # Apply weights to features before averaging
        aggregated_scores = None

        # get simple z-score features
        z_score_feature_dict = self._get_z_score_features(self.feature_dict)

        for feature_name, mat in z_score_feature_dict.items():
            if feature_name in self.feature_weights:
                weight = self.feature_weights[feature_name]
                if aggregated_scores is None:
                    aggregated_scores = mat.copy() * weight
                else:
                    aggregated_scores += mat.copy() * weight
                
        # Drop rows with all NA's
        aggregated_scores.dropna(how="all", inplace=True)
        
        # Filter by date range
        aggregated_scores = aggregated_scores.loc[self.start_date:self.end_date]
        
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

    # average scores across time (8 hrs (16*30min periods) with 2 hrs overlap (4*30min periods))
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
            
            
def test_signal_2(feature_dict: Dict[str, pd.DataFrame], start_date: datetime.datetime, end_date: datetime.datetime):

    print("testing...")

    return None
