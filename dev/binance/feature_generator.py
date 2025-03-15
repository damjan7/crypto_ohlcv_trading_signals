from typing import List, Dict, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class Config:
    """
    Configuration for (technical) feature calculations.
    
    Attributes:
        lookback_period_rsi (int): Number of periods for RSI calculation.
        lookback_period_bollinger (int): Number of periods for Bollinger Bands.
        rsi_oversold (float): RSI threshold below which the asset is considered oversold.
        rsi_overbought (float): RSI threshold above which the asset is considered overbought.
        ema_short (int): Period for the short-term EMA.
        ema_long (int): Period for the long-term EMA.
        stochastic_period (int): Period for the stochastic oscillator.
        atr_period (int): Period for ATR (Average True Range) calculation.
    """
    lookback_period_rsi: int = 14
    lookback_period_bollinger: int = 20
    rsi_oversold: float = 30
    rsi_overbought: float = 70
    ema_short: int = 9
    ema_long: int = 21
    stochastic_period: int = 14
    atr_period: int = 14

class BaseFeature(ABC):
    """
    Abstract base class for feature calculation classes.
    Each feature class must implement the calculate method to add its computed feature columns.
    """
    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate and add features to the DataFrame.
        
        Parameters:
            df (pd.DataFrame): DataFrame containing at least the 'close' price column,
                               and optionally 'high', 'low', and 'volume' columns.
        
        Returns:
            pd.DataFrame: The original DataFrame augmented with new feature columns.
        """
        pass

class RSIFeature(BaseFeature):
    """
    Calculates the Relative Strength Index (RSI) and a corresponding signal feature.
    
    RSI is a normalized momentum indicator (0-100), making it directly comparable across assets
    with different price levels.
    """
    def __init__(self, config: Config):
        self.period = config.lookback_period_rsi
        self.oversold = config.rsi_oversold
        self.overbought = config.rsi_overbought
    
    def get_names(self) -> List[str]:
        return ['RSI', 'RSI_feature']

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        # RSI signal: 1 indicates oversold (potential buy), -1 indicates overbought (potential sell)
        df['RSI_feature'] = np.where(df['RSI'] < self.oversold, 1, 
                                     np.where(df['RSI'] > self.overbought, -1, 0))
        return df

class EMAFeature(BaseFeature):
    """
    Calculates exponential moving averages (EMA) over two different periods and generates a crossover signal.
    
    Note:
        - The raw EMA values are in absolute price units (scale-dependent) and should be compared with caution.
        - The crossover signal (EMA_feature), indicating trend direction, can be more informative for cross-asset comparisons.
    """
    def __init__(self, config: Config):
        self.short_period = config.ema_short
        self.long_period = config.ema_long

    def get_names(self) -> List[str]:
        return ['EMA_short', 'EMA_long', 'EMA_feature']

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        df['EMA_short'] = df['close'].ewm(span=self.short_period, adjust=False).mean()
        df['EMA_long'] = df['close'].ewm(span=self.long_period, adjust=False).mean()
        df['EMA_feature'] = np.where(df['EMA_short'] > df['EMA_long'], 1, -1)
        return df
    
class ReturnFeatures(BaseFeature):
    """
    Calculates percentage returns over specified periods.
    """
    def __init__(self):
        self.periods = {
            '1': 1,     # 1 period
            '2': 2,     # 2 periods
            '6': 6,     # 6 periods
            '12': 12,   # 12 periods
        }

    def get_names(self) -> List[str]:
        return [f'return_{name}' for name in self.periods.keys()]

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        for name, period in self.periods.items():
            df[f'return_{name}'] = df['close'].pct_change(periods=period)
        return df

class VolatilityFeatures(BaseFeature):
    """
    Calculates rolling volatility (the standard deviation of percentage returns) over various windows.
    """
    def __init__(self):
        self.windows = [2, 6, 12, 24, 48]  # windows expressed in 30-minute periods

    def get_names(self) -> List[str]:
        return [f'volatility_{window}' for window in self.windows]

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        for window in self.windows:
            df[f'volatility_{window}'] = df['close'].pct_change().rolling(window=window).std()
        return df

class VolumeFeatures(BaseFeature):
    """
    Calculates volume-based features including moving averages and a relative volume measure.
    
    Note:
        - Raw volume moving averages (volume_ma_*) are scale-dependent.
        - The relative volume feature (volume_rel_ma24), computed as volume divided by its 24-period moving average,
          is normalized and more comparable across assets with different trading volumes.
    """
    def __init__(self):
        self.windows = [2, 6, 12, 24, 48]  # windows expressed in x num of periods

    def get_names(self) -> List[str]:
        return [f'volume_ma_{window}' for window in self.windows] + ['volume_rel_ma24']

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        # Compute moving averages for volume over different windows
        for window in self.windows:
            df[f'volume_ma_{window}'] = df['usd_volume'].rolling(window=window).mean()
        
        # Normalize the current usd_volume using the 24-period moving average for better cross-asset comparison
        df['volume_rel_ma24'] = df['usd_volume'] / df['volume_ma_24']
        return df

class PriceMAFeatures(BaseFeature):
    """
    Calculates price moving averages and relative deviations from these averages.
    
    Note:
        - Raw price moving averages (price_ma_*) are in absolute price units.
        - Relative deviations (price_rel_ma_*) are computed as the difference relative to the moving average,
          making them scale-invariant and comparable across assets.
    """
    def __init__(self):
        self.windows = [2, 6, 12, 24, 48, 96]  # windows expressed in 30-minute periods
    
    def get_names(self) -> List[str]:
        return [f'price_ma_{window}' for window in self.windows] + [f'price_rel_ma_{window}' for window in self.windows]

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        for window in self.windows:
            df[f'price_ma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'price_rel_ma_{window}'] = df['close'] / df[f'price_ma_{window}'] - 1
        return df

    

class MomentumFeatures(BaseFeature):
    """
    Calculates momentum features, including rate-of-change measures and MACD.
    
    Note:
        - The percentage-based momentum measures (e.g., pct change) are scale-invariant.
        - MACD (and its signal line) are derived from EMAs, so while the raw values are scale-dependent,
          the MACD crossover signals are useful for cross-asset trend comparisons.
    """
    def __init__(self):
        self.windows = [2, 6, 12, 24, 48]  # windows expressed in 30-minute periods
    
    def get_names(self) -> List[str]:
        return [f'momentum_{window}' for window in self.windows] + ['macd', 'macd_feature']

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        # Calculate rate-of-change momentum features
        for window in self.windows:
            df[f'momentum_{window}'] = df['close'].pct_change(periods=window)
        
        # MACD calculation: difference between 12-period and 26-period EMAs
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        # MACD signal line
        df['macd_feature'] = df['macd'].ewm(span=9, adjust=False).mean()
        return df

class BollingerBandsFeature(BaseFeature):
    """
    Calculates Bollinger Bands and a corresponding signal feature based on price deviations.
    
    Although the bands are derived from raw price data, the resulting signal (BB_feature) uses standard deviation,
    providing a normalized context that is more comparable across assets.
    """
    def __init__(self, config: Config):
        self.period = config.lookback_period_bollinger
        self.std_dev = 2
    
    def get_names(self) -> List[str]:
        return ['BB_middle', 'BB_upper', 'BB_lower', 'BB_feature']

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        df['BB_middle'] = df['close'].rolling(window=self.period).mean()
        df['BB_std'] = df['close'].rolling(window=self.period).std()
        df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * self.std_dev)
        df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * self.std_dev)
        df['BB_feature'] = np.where(df['close'] < df['BB_lower'], 1,
                                     np.where(df['close'] > df['BB_upper'], -1, 0))
        return df

class ATRFeature(BaseFeature):
    """
    Calculates the Average True Range (ATR), a measure of market volatility.
    Note:
        - ATR is calculated in absolute price units, which means it is scale-dependent.
        - For cross-asset comparisons, consider normalizing ATR by the asset's price.
    """
    def __init__(self, config: Config):
        self.period = config.atr_period

    def get_names(self) -> List[str]:
        return ['ATR']

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        # True range: maximum of (high - low, |high - previous close|, |low - previous close|)
        high_low = df['high'] - df['low']
        high_prev_close = (df['high'] - df['close'].shift(1)).abs()
        low_prev_close = (df['low'] - df['close'].shift(1)).abs()
        true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
        df['ATR'] = true_range.rolling(window=self.period).mean()
        return df

class LogReturnFeature(BaseFeature):
    """
    Calculates the log returns of the asset's close price.
    """
    def get_names(self) -> List[str]:
        return ['log_return']

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        return df

class StochasticOscillatorFeature(BaseFeature):
    """
    Calculates the Stochastic Oscillator (%K and %D).
    
    This momentum indicator is normalized (ranging from 0 to 100) and measures the close relative to its recent high-low range,
    making it directly comparable across different assets.
    """
    def __init__(self, config: Config):
        self.period = config.stochastic_period

    def get_names(self) -> List[str]:
        return ['stoch_k', 'stoch_d']

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        low_min = df['low'].rolling(window=self.period).min()
        high_max = df['high'].rolling(window=self.period).max()
        df['stoch_k'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()  # %D is the 3-period moving average of %K
        return df

class OBVFeature(BaseFeature):
    """
    Calculates the On-Balance Volume (OBV), which aggregates volume based on the direction of price movement.
    Note:
        - OBV is a cumulative measure and is sensitive to absolute volume levels.
        - This makes it less directly comparable across assets with very different trading volumes.
    """
    def get_names(self) -> List[str]:
        return ['OBV']

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        df['direction'] = np.where(df['close'].diff() >= 0, 1, -1)
        df['OBV'] = (df['usd_volume'] * df['direction']).cumsum()
        df.drop(columns=['direction'], inplace=True)
        return df

class VolumeZScoreFeature(BaseFeature):
    """
    Calculates the z-score of the trading volume over a specified rolling window.
    By standardizing volume this feature becomes scale-invariant.
    """
    def __init__(self, window: int = 20):
        self.window = window

    def get_names(self) -> List[str]:
        return [f'volume_zscore_{self.window}']

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        vol_mean = df['usd_volume'].rolling(window=self.window).mean()
        vol_std = df['usd_volume'].rolling(window=self.window).std()
        df[f'volume_zscore_{self.window}'] = (df['usd_volume'] - vol_mean) / vol_std
        return df

# ===========================================================
# Comments on Feature Comparability between Different Assets
# ===========================================================
#
# 1. Scale-Invariant Features (directly comparable across assets):
#    - Percentage Returns (from ReturnFeatures and LogReturnFeature): They express relative change.
#    - RSI and Stochastic Oscillator: Normalized momentum indicators (0-100).
#    - Volatility (based on percentage changes).
#    - Relative Price Measures (price_rel_ma) and Relative Volume (volume_rel_ma24).
#    - Volume Z-Score (volume_zscore) for standardized volume.
#
# 2. Scale-Dependent Features (require caution or additional normalization):
#    - Raw EMAs and Price Moving Averages (EMA, price_ma): Reflect absolute price levels.
#    - ATR: An absolute measure of volatility (can be normalized by price if needed).
#    - OBV: A cumulative measure that varies greatly with absolute trading volume.
#
# For cross-asset analysis, focus on the normalized or percentage-based indicators to ensure meaningful comparisons.


# Feature Generator Class
class FeatureGenerator:


    def __init__(self, config: Config = Config()):
        self.config = config
        self.features: List[BaseFeature] = []

    def add_feature(self, feature: BaseFeature) -> None:
        self.features.append(feature)

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        for feature in self.features:
            df = feature.calculate(df)
        return df

    def get_all_feature_names(self):
        all_feature_names = []
        for feature in self.features:
            all_feature_names.extend(feature.get_names())
        return all_feature_names




