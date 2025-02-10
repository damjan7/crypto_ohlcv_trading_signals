import signal_generator as sg
import pandas as pd
import polars as pl
import os
import numpy as np
import plotly.express as px

def create_feature_dataset(df: pd.DataFrame) -> pd.DataFrame:
    '''
    This function takes in a dataframe and returns a dataframe with all the features calculated.
    This is done for a single pair. 
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
        feature_df = feature.calculate(feature_df)
    return feature_df


TICKER_DATA_PATH = r"C:\Users\Damja\CODING_LOCAL\trading\data\ticker_specific_data_BINANCE"
pairs = pd.read_csv(r"dev\binance\pairs.csv")
NUM_PAIRS_TO_LOAD = 100
pairs = pairs.iloc[:NUM_PAIRS_TO_LOAD, 0].values
pairs = [pair.replace("USD", "USDT") for pair in pairs]
pair = pairs[2]
df = pl.read_parquet(f'{TICKER_DATA_PATH}/{pair.replace("/", "")}.parquet').to_pandas()

create_feature_dataset(df)