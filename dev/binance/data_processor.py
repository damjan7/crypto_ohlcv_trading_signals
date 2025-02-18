import feature_generator as fg
import pandas as pd
import polars as pl
import os
import numpy as np
import plotly.express as px
from typing import List, Dict, Tuple


def create_feature_dataset(df: pd.DataFrame) -> pd.DataFrame:
    '''
    This function takes in a dataframe and returns a dataframe with all the features calculated.
    This is done for a single pair. 

    Also returns a list with all the names of the features
    '''

    df = df.copy()

    # Initialize
    config = fg.Config()
    generator = fg.FeatureGenerator(config)
    
    # Add all features
    # manual so far, but no big deal
    generator.add_feature(fg.ReturnFeatures())
    generator.add_feature(fg.VolatilityFeatures())
    generator.add_feature(fg.VolumeFeatures())
    generator.add_feature(fg.PriceMAFeatures())
    generator.add_feature(fg.RSIFeature(config))
    generator.add_feature(fg.EMAFeature(config))
    generator.add_feature(fg.BollingerBandsFeature(config))

    for feature in generator.features:
        feature_df = feature.calculate(df)
    
    feature_names = generator.get_all_feature_names()
    feature_df.index = feature_df.Date
    feature_df.drop(["Date"], axis=1, inplace=True)
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
    target_df = df.loc[:, target_names + ['Date']]
    target_df.index = df.Date
    #target_df.drop(["Date"], axis=1, inplace=True)
    return target_df, target_names

def create_target_datasets_for_pairs(
    pairs: List[str],
    input_path: str
) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    out_dict = {}
    for pair in pairs:
        df = pl.read_parquet(f'{input_path}/{pair.replace("/", "")}.parquet').to_pandas()
        out_dict[pair], target_names = create_target_dataset(df)
    return out_dict, target_names

def create_feature_ds_dict_for_pairs(
    pairs: List[str], 
    input_path: str
) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    out_dict = {}
    for pair in pairs:
        df = pl.read_parquet(f'{input_path}/{pair.replace("/", "")}.parquet').to_pandas()
        out_dict[pair], feat_names = create_feature_dataset(df)
    return out_dict, feat_names


def create_cross_sectional_ds(dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Given a feature Dictionary, creates a cross sectional dataset
    """

#def calc_feature_dataset_for_pairs() 
    

class DataClass():
    """
    Data Class that holds all the data that is then used in the signal generator

    TODO:
    - implement the above funcs in the class structure? or not?
    """
    def __init__(self, pairs: List[str], input_path: str):
        self.feature_dict, self.feature_names = create_feature_ds_dict_for_pairs(pairs=pairs, input_path=input_path)
        self.target_dict, self.target_names = create_target_datasets_for_pairs(pairs=pairs, input_path=input_path)

    def get_feature_names(self):
        return self.feature_names

    def summarize_data(self):
        pass

    def normalize_features(self, feature_names: List[str]):
        pass

