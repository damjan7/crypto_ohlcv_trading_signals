import feature_generator as fg
import pandas as pd
import polars as pl
import os
import numpy as np
import plotly.express as px
from typing import List, Dict, Tuple
from Backtester_simple import Backtester
import datetime

import data_processor as datproc

print(os.getcwd())

#C:\Users\Damja\CODING_LOCAL\trading\dev
#C:\Users\Damja\CODING_LOCAL\trading\dev\binance\pairs.csv

TICKER_DATA_PATH = r"data/ticker_specific_data_BINANCE"
#pairs = pd.read_csv(os.path.join(os.getcwd(), "dev", "binance", "pairs.csv")) # dev/binance/pairs.csv
pairs = pd.read_csv("dev/binance/pairs.csv")
NUM_PAIRS_TO_LOAD = 15
pairs = pairs.iloc[:NUM_PAIRS_TO_LOAD, 0].values
pairs = [pair.replace("USD", "USDT") for pair in pairs]

pairs = [p for p in pairs if p not in ['USDTT/USDT', 'USDT/USDT', "USDTC/USDT"]]


data = datproc.DataClass(pairs=pairs, input_path=TICKER_DATA_PATH)

#FEATURE MATRIX DICTIONARY
data.create_cross_sectional_feature_matrix_dictionary()




#######################################################
### THIS IS THE OLD WAY OF CREATING SIGNALS
#######################################################
normalized_dict = {}
for feature_name in ["return_1h", "volume_rel_ma24", "RSI_feature", "price_rel_ma_6", "price_rel_ma_12", "volume_zscore_20"]:
    df = data.cross_sectional_feat_dict[feature_name]
    df_demeaned = (df.sub(df.mean(axis=1), axis=0))
    df_standardized = (df_demeaned.div(df_demeaned.std(axis=1), axis=0))
    normalized_dict[feature_name] = df_standardized
import signal_generator as sg
weights = sg.test_signal_1(normalized_dict=normalized_dict, start_date=datetime.datetime(2024, 1, 1), end_date=datetime.datetime(2024, 12, 31))
#######################################################
#######################################################
#

### Call to Backtester with signals
pairs = data.pairs

# random signals (weights) for testing
"""
signals = [ np.concatenate([np.random.dirichlet(np.ones(5)), -np.random.dirichlet(np.ones(5)), np.zeros(5)]) for i in range(252*4)]
signals = pd.DataFrame(signals)
signals.index = score_matrix.index[-252*4:]
print(f"Signals have shape {signals.shape}")
"""

# backtester
start_date = weights["8hr_signal"].index[0]
end_date = weights["8hr_signal"].index[-1]

# get df in dictionary format
data_dict = {}
data_dict['close'] = data.cross_sectional_feat_dict["close"]

data_dict_8hr = {}
roll = data.cross_sectional_feat_dict["close"].rolling(window=16, step=4, min_periods=1)
data_dict_8hr['close'] = roll.agg(lambda x : x.iloc[-1])

# data dict filtered by start and end date
data_dict_1yr = {k: v[start_date:end_date] for k, v in data_dict.items()}

# create empty weights df with same index as data_dict_1yr
weights_1yr = pd.DataFrame(np.nan,index=data_dict_1yr["close"].index, columns=data_dict_1yr["close"].columns)
weights_1yr = weights_1yr.fillna(weights["8hr_signal"])
weights_1yr = weights_1yr.ffill()

# btc 1 weight else 0
"""
weights_1yr = pd.DataFrame(np.nan,index=data_dict_1yr["close"].index, columns=data_dict_1yr["close"].columns)
weights_1yr["BTC/USDT"] = 1"""

"""sg.test_signal_2(feature_dict=data.cross_sectional_feat_dict, start_date=datetime.datetime(2023, 1, 1), end_date=datetime.datetime(2024, 12, 31))
"""
# test SignalClass
"""signal_config = sg.SignalConfig(window_size=16, step_size=4, min_periods=1, top_n=5, bottom_n=5)
signal_generator = sg.AverageFeatureSignal(config=signal_config)
signal_generator2 = sg.WeightedFeatureSignal(config=signal_config)

weights = signal_generator.generate_signal(feature_dict=data.cross_sectional_feat_dict, start_date=start_date, end_date=end_date)
weights = signal_generator2.generate_signal(feature_dict=data.cross_sectional_feat_dict, start_date=start_date, end_date=end_date)
"""

# test SimpleSortingLongOnly
signal_generator = sg.SimpleSortingLongOnly(
    config=sg.SignalConfig(window_size=16, step_size=4, min_periods=1, top_n=5, bottom_n=5), 
    signal_name="simple_sorting_signal",
    feature_name="stoch_d",
    feature_dict=data.cross_sectional_feat_dict,
    start_date=start_date,
    end_date=end_date,
    num_quantiles=5
    )
weights = signal_generator.generate_signal()

# daily weights
daily_weights = {k: w.resample("D").last() for k, w in weights.items()}


backtester = Backtester(data_dict=data_dict_1yr, weights=daily_weights['quintile_5'], start_date=start_date, end_date=end_date)
out = backtester.run_backtest()
metrics = backtester.performance_metrics()
print(metrics)
backtester.plot_results()


# test if 8h close prices or 8hr rolling returns are the same



print("done")
