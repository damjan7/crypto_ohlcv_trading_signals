### Simple Trend Volume Strategy
# Trend needs to be positive and volume above some long term average

import sys
import os
# Add parent directory to path to find modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import feature_generator as fg
import pandas as pd
import polars as pl
import os
import numpy as np
import plotly.express as px
from typing import List, Dict, Tuple
from Backtester_simple import Backtester
import datetime
import signal_generator as sg
import data_processor as datproc


# load data
TICKER_DATA_PATH = r"data/ticker_specific_data_BINANCE"
pairs = pd.read_csv("dev/binance/pairs.csv")
NUM_PAIRS_TO_LOAD = 100
pairs = pairs.iloc[:NUM_PAIRS_TO_LOAD, 0].values
pairs = [pair.replace("USD", "USDT") for pair in pairs]
pairs = [p for p in pairs if p not in ['USDTT/USDT', 'USDT/USDT', "USDTC/USDT"]]
data = datproc.DataClass(pairs=pairs, input_path=TICKER_DATA_PATH)
print("Creating Cross Sectional Feature Matrix..")
data.create_cross_sectional_feature_matrix_dictionary()
print("All features: ", data.cross_sectional_feat_dict.keys())
print("Finished Cross Sectional Feature Matrix!")
# done loading data

# if wanted, normalize the features
#data.normalize_cross_sectional_features(feature_names=[...])


# Signal Idea
# Volume Moving Averages
# volume_ma_2 corresp. to 2 x period length (now 6hrs), hence 12hrs
# volume_ma_6 --> 36hrs, 
# volume_ma_12 --> 72hrs
# volume_ma_24 --> 144hrs = 6 days
# volume_ma_48 --> 288hrs = 12 days

# example; if 12 or 36 hrs > than 12 days times some factor, then buy
# problem, 12 hrs can be during US day time, so might need a large factor to account for this

# let's see how often this happens
(data.cross_sectional_feat_dict['volume_ma_2'] > 1.5 * data.cross_sectional_feat_dict['volume_ma_48']).sum() / (len(data.cross_sectional_feat_dict['volume_ma_2']) - np.isnan(data.cross_sectional_feat_dict['volume_ma_2']).sum())
# for factor 1.5, for BTC happens around 1/7 of the time, but probably in bull and bear markets so seems alright

# same for 12 day ma, so might use this factor
(data.cross_sectional_feat_dict['volume_ma_2'] > 1.5 * data.cross_sectional_feat_dict['volume_ma_12']).sum() / (len(data.cross_sectional_feat_dict['volume_ma_2']) - np.isnan(data.cross_sectional_feat_dict['volume_ma_2']).sum())


# now for the trend signal
# return_1, return_2, return_6, return_12
# Relative deviations (price_rel_ma_*) are computed as the difference relative to the moving average
# df['close'] / df[f'price_ma_{window}'] - 1














