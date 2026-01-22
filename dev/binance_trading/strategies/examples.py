from strategies.base import BaseStrategy
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import datetime

class SimpleSortingStrategy(BaseStrategy):
    """
    Simple Sorting based on a single feature.
    """
    def __init__(self, feature_name: str, num_quantiles: int = 5, config: Optional[Dict] = None):
        super().__init__(config)
        self.feature_name = feature_name
        self.num_quantiles = num_quantiles

    def generate_signals(self, feature_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        if self.feature_name not in feature_dict:
            raise ValueError(f"Feature {self.feature_name} not found in feature_dict keys: {list(feature_dict.keys())}")
        
        feature_df = feature_dict[self.feature_name]
        
        # Calculate ranks for all dates at once
        all_ranks = feature_df.rank(axis=1, method="first", pct=True)
        
        weights = pd.DataFrame(0.0, index=feature_df.index, columns=feature_df.columns)
        
        # We want to create weights for Quintile 1 (Bottom) to Quintile 5 (Top)? 
        # Original code returned a dictionary of quintiles. 
        # Here we should probably return a specific portfolio (e.g. Long Top / Short Bottom or just Long Top).
        # The user's original debug.py used "quintile_1". 
        # Let's assume we want to implement a Long Top / Short Bottom or flexible strat.
        # For this example, let's implement a Long-Short Strategy: Short Bottom Quantile, Long Top Quantile.
        
        # But to match original "SimpleSortingLongOnly" name, maybe just Long Top?
        # Original code had `quintile_weights` dictionary. 
        # Let's implement logic to return weights for the 'Top' quantile as the signal.
        
        top_quantile_threshold = 1.0 - (1.0 / self.num_quantiles)
        
        # Create mask for top quantile
        target_mask = all_ranks > top_quantile_threshold
        
        # Count assets in the top quantile per row
        counts = target_mask.sum(axis=1)
        
        # Assign weights (1/count)
        # Avoid division by zero
        weights = target_mask.astype(float).div(counts.replace(0, np.nan), axis=0).fillna(0)
        
        return weights

class AverageFeatureStrategy(BaseStrategy):
    """
    Strategy that averages normalized features and creates long-short portfolios.
    """
    def __init__(self, features: List[str], top_n: int = 5, bottom_n: int = 5, config: Optional[Dict] = None):
        super().__init__(config)
        self.features = features
        self.top_n = top_n
        self.bottom_n = bottom_n

    def generate_signals(self, feature_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        # Check if all features exist
        missing = [f for f in self.features if f not in feature_dict]
        if missing:
            raise ValueError(f"Missing features: {missing}")

        # Get z-scores and average them
        aggregated_scores = None
        count = 0
        
        for feature_name in self.features:
            df = feature_dict[feature_name]
            # Standardize (Z-score) cross-sectionally
            df_demeaned = df.sub(df.mean(axis=1), axis=0)
            df_standardized = df_demeaned.div(df_demeaned.std(axis=1), axis=0)
            
            if aggregated_scores is None:
                aggregated_scores = df_standardized
            else:
                aggregated_scores += df_standardized
            count += 1
            
        aggregated_scores /= count
        aggregated_scores.dropna(how="all", inplace=True)
        
        # Generate ranks
        ranks = aggregated_scores.rank(axis=1, method="min")
        
        weights = pd.DataFrame(np.nan, index=ranks.index, columns=ranks.columns)
        
        for idx in weights.index:
            row_ranks = ranks.loc[idx]
            
            # Simple Top N / Bottom N based on rank score (which is just rank of average z-score)
            # Higher z-score -> Higher rank -> Long
            # Be careful with nlargest/nsmallest on ranks vs scores. 
            # Ranks are 1..N. Higher rank = Higher Score.
            
            top_n_idx = row_ranks.nlargest(self.top_n).index
            bottom_n_idx = row_ranks.nsmallest(self.bottom_n).index
            
            # Weighting by rank or equal weight?
            # Original code: weights proportional to ranks.
            
            top_ranks = row_ranks[top_n_idx]
            weights.loc[idx, top_n_idx] = top_ranks / top_ranks.sum()
            
            bottom_ranks = row_ranks[bottom_n_idx]
            weights.loc[idx, bottom_n_idx] = -bottom_ranks / bottom_ranks.sum()
            
        return weights.fillna(0)
