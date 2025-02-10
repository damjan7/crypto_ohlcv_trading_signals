from typing import List, Dict, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class SignalConfig:
    lookback_period_rsi: int = 14
    lookback_period_bollinger: int = 20
    rsi_oversold: float = 30
    rsi_overbought: float = 70
    ema_short: int = 9
    ema_long: int = 21

class BaseFeature(ABC): 
    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

class RSIFeature(BaseFeature):
    def __init__(self, config: SignalConfig):
        self.period = config.lookback_period_rsi
        self.oversold = config.rsi_oversold
        self.overbought = config.rsi_overbought

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI_signal'] = np.where(df['RSI'] < self.oversold, 1, 
                                   np.where(df['RSI'] > self.overbought, -1, 0))
        return df

class EMAFeature(BaseFeature):
    def __init__(self, config: SignalConfig):
        self.short_period = config.ema_short
        self.long_period = config.ema_long

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        df['EMA_short'] = df['close'].ewm(span=self.short_period).mean()
        df['EMA_long'] = df['close'].ewm(span=self.long_period).mean()
        df['EMA_signal'] = np.where(df['EMA_short'] > df['EMA_long'], 1, -1)
        return df
    
class ReturnFeatures(BaseFeature):
    def __init__(self):
        self.periods = {
            '30m': 1,    # 1 period of 30min
            '1h': 2,     # 2 periods of 30min
            '6h': 12,    # 12 periods of 30min
        }

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        for name, period in self.periods.items():
            df[f'return_{name}'] = df['close'].pct_change(periods=period)
        return df

class VolatilityFeatures(BaseFeature):
    def __init__(self):
        self.windows = [2, 6, 12, 24, 48]  # 1h, 3h, 6h, 12h, 24h in 30min periods

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        for window in self.windows:
            df[f'volatility_{window}'] = df['close'].pct_change().rolling(window=window).std()
        return df

class VolumeFeatures(BaseFeature):
    def __init__(self):
        self.windows = [2, 6, 12, 24, 48]  # 1h, 3h, 6h, 12h, 24h in 30min periods

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        # Volume moving averages
        for window in self.windows:
            df[f'volume_ma_{window}'] = df['volume'].rolling(window=window).mean()
        
        # Volume relative to moving average
        df['volume_rel_ma24'] = df['volume'] / df['volume_ma_24']
        return df

class PriceMAFeatures(BaseFeature):
    def __init__(self):
        self.windows = [2, 6, 12, 24, 48, 96]  #  1h, 3h, 6h, 12h, 24h, 48h in 30min periods

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        for window in self.windows:
            df[f'price_ma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'price_rel_ma_{window}'] = df['close'] / df[f'price_ma_{window}'] - 1
        return df

class MomentumFeatures(BaseFeature):
    def __init__(self):
        self.windows = [2, 6, 12, 24, 48]  #  1h, 3h, 6h, 12h, 24h in 30min periods

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        # ROC (Rate of Change)
        for window in self.windows:
            df[f'momentum_{window}'] = df['close'].pct_change(periods=window)
        
        # MACD
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        return df

class BollingerBandsFeature(BaseFeature):
    def __init__(self, config: SignalConfig):
        self.period = config.lookback_period_bollinger
        self.std_dev = 2

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        df['BB_middle'] = df['close'].rolling(window=self.period).mean()
        df['BB_std'] = df['close'].rolling(window=self.period).std()
        df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * self.std_dev)
        df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * self.std_dev)
        df['BB_signal'] = np.where(df['close'] < df['BB_lower'], 1,
                                  np.where(df['close'] > df['BB_upper'], -1, 0))
        return df


class SignalGenerator:
    def __init__(self, config: SignalConfig = SignalConfig()):
        self.config = config
        self.features: List[BaseFeature] = []
        
    def add_feature(self, feature: BaseFeature) -> None:
        self.features.append(feature)
        
    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        for feature in self.features:
            df = feature.calculate(df)
        return df
    
    def generate_combined_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        signal_columns = [col for col in df.columns if col.endswith('_signal')]
        df['combined_signal'] = df[signal_columns].mean(axis=1)
        return df
