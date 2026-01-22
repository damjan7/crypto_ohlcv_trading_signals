from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Optional

class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    Strategies must implement `generate_signals` to return portfolio weights.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

    @abstractmethod
    def generate_signals(self, feature_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate target weights for the portfolio.
        
        Args:
            feature_dict: Dictionary where keys are asset symbols and values are DataFrames
                          containing features (and 'close' price).
                          
        Returns:
            pd.DataFrame: A DataFrame of weights where index is time and columns are assets.
        """
        pass
