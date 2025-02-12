import signal_generator as sg
import pandas as pd
import polars as pl
import os
import numpy as np
import plotly.express as px
from typing import List

def create_feature_dataset(df: pd.DataFrame) -> pd.DataFrame:
    '''
    This function takes in a dataframe and returns a dataframe with all the features calculated.
    This is done for a single pair. 

    Also returns a list with all the names of the features
    '''

    # Initialize
    config = sg.SignalConfig()
    generator = sg.SignalGenerator(config)
    
    # Add all features
    generator.add_feature(sg.ReturnFeatures())
    generator.add_feature(sg.VolatilityFeatures())
    generator.add_feature(sg.VolumeFeatures())
    generator.add_feature(sg.PriceMAFeatures())
    generator.add_feature(sg.RSIFeature(config))
    generator.add_feature(sg.EMAFeature(config))
    generator.add_feature(sg.BollingerBandsFeature(config))

    for feature in generator.features:
        feature_df = feature.calculate(df)
    
    feature_names = generator.get_all_feature_names()
    return feature_df, feature_names

def create_target_dataset(
        df: pd.DataFrame,
        horizon_lst: List[int] = [1, 6, 12, 24, 48],
        target_type: str = 'regression',
        price_column: str = 'close',
        threshold: float = 0.002
        ) -> pd.DataFrame:
    '''
    Create one or multiple target variables for time series forecasting.
    Target can be regression or classification.
    '''
    df = df.copy()

    # Create target variable
    for horizon in horizon_lst:
        if target_type == 'classification':
            # create a binary variable: 1 if the return is greater than the threshold, 0 otherwise
            tmp = df[price_column].shift(-horizon) / df[price_column] - 1
            df[f'target_classif_horizon{horizon}'] = (tmp > threshold).astype(int)
        elif target_type == 'regression':
            df[f'target_regr_horizon{horizon}'] = df[price_column].shift(-horizon) / df[price_column] - 1
        else:
            raise ValueError("target_type must be either 'classification' or 'regression'")
    
    target_names = [f'target_regr_horizon{horizon}' for horizon in horizon_lst]
    target_df = df.loc[:, target_names]
    target_df['Date'] = df.loc[:, ["Date"]]
    return target_df, target_names




#def calc_feature_dataset_for_pairs() 
    

#### THIS SHOULD BE CALLED IN MAIN FILE ####
TICKER_DATA_PATH = r"C:\Users\Damja\CODING_LOCAL\trading\data\ticker_specific_data_BINANCE"
pairs = pd.read_csv(r"dev\binance\pairs.csv")
NUM_PAIRS_TO_LOAD = 100
pairs = pairs.iloc[:NUM_PAIRS_TO_LOAD, 0].values
pairs = [pair.replace("USD", "USDT") for pair in pairs]
pair = pairs[2]
df = pl.read_parquet(f'{TICKER_DATA_PATH}/{pair.replace("/", "")}.parquet').to_pandas()

df, feature_names = create_feature_dataset(df)

target_df, target_names = create_target_dataset(df, target_type='regression')

print("done")
