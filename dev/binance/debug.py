import feature_generator as fg
import pandas as pd
import polars as pl
import os
import numpy as np
import plotly.express as px
from typing import List, Dict, Tuple

import data_processor as datproc

print(os.getcwd())

#C:\Users\Damja\CODING_LOCAL\trading\dev
#C:\Users\Damja\CODING_LOCAL\trading\dev\binance\pairs.csv

TICKER_DATA_PATH = r"data/ticker_specific_data_BINANCE"
#pairs = pd.read_csv(os.path.join(os.getcwd(), "dev", "binance", "pairs.csv")) # dev/binance/pairs.csv
pairs = pd.read_csv("dev/binance/pairs.csv")
NUM_PAIRS_TO_LOAD = 100
pairs = pairs.iloc[:NUM_PAIRS_TO_LOAD, 0].values
pairs = [pair.replace("USD", "USDT") for pair in pairs]

pairs = [p for p in pairs if p != 'USDTT/USDT']

DATA = datproc.DataClass(pairs=pairs[0:15], input_path=TICKER_DATA_PATH)

#FEATURE MATRIX DICTIONARY
DATA.create_cross_sectional_feature_matrix_dictionary()


# given FEATURE MATRIX DICTIONARY, create a signal
# lets look at volatility_2, return_1h, volume_rel_ma24, RSI_feature, price_rel_ma_6, price_rel_ma_12, volume_zscore_20

normalized_dict = {}
for feature_name in ["volatility_2", "return_1h", "volume_rel_ma24", "RSI_feature", "price_rel_ma_6", "price_rel_ma_12", "volume_zscore_20"]:
    df = DATA.cross_sectional_feat_dict[feature_name]
    df_demeaned = (df.sub(df.mean(axis=1), axis=0))
    df_standardized = (df_demeaned.div(df_demeaned.std(axis=1), axis=0))
    normalized_dict[feature_name] = df_standardized




print("done")
