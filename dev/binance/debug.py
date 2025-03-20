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

print(os.getcwd())

#C:\Users\Damja\CODING_LOCAL\trading\dev
#C:\Users\Damja\CODING_LOCAL\trading\dev\binance\pairs.csv

TICKER_DATA_PATH = r"data/ticker_specific_data_BINANCE"
#pairs = pd.read_csv(os.path.join(os.getcwd(), "dev", "binance", "pairs.csv")) # dev/binance/pairs.csv
pairs = pd.read_csv("dev/binance/pairs.csv")
NUM_PAIRS_TO_LOAD = 100
pairs = pairs.iloc[:NUM_PAIRS_TO_LOAD, 0].values
pairs = [pair.replace("USD", "USDT") for pair in pairs]

pairs = [p for p in pairs if p not in ['USDTT/USDT', 'USDT/USDT', "USDTC/USDT"]]


data = datproc.DataClass(pairs=pairs, input_path=TICKER_DATA_PATH)

# data.feature_dict contains features for each pair
# data.cross_sectional_feat_dict contains feature matrices for each feature

#FEATURE MATRIX DICTIONARY
print("Creating Cross Sectional Feature Matrix..")
data.create_cross_sectional_feature_matrix_dictionary()
print("All features: ", data.cross_sectional_feat_dict.keys())
print("Finished Cross Sectional Feature Matrix!")

# can normalize the features here
# NOT IMPLEMENTED YET
#data.normalize_cross_sectional_features()




#######################################################
### THIS IS THE OLD WAY OF CREATING SIGNALS
#######################################################
"""normalized_dict = {}
for feature_name in ["return_1h", "volume_rel_ma24", "RSI_feature", "price_rel_ma_6", "price_rel_ma_12", "volume_zscore_20"]:
    df = data.cross_sectional_feat_dict[feature_name]
    df_demeaned = (df.sub(df.mean(axis=1), axis=0))
    df_standardized = (df_demeaned.div(df_demeaned.std(axis=1), axis=0))
    normalized_dict[feature_name] = df_standardized
import signal_generator as sg
weights = sg.test_signal_1(normalized_dict=normalized_dict, start_date=datetime.datetime(2024, 1, 1), end_date=datetime.datetime(2024, 12, 31))
#######################################################"""
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

"""# backtester
start_date = weights["8hr_signal"].index[0]
end_date = weights["8hr_signal"].index[-1]

# get df in dictionary format
data_dict = {}
data_dict['close'] = data.cross_sectional_feat_dict["close"]

data_dict_8hr = {}
roll = data.cross_sectional_feat_dict["close"].rolling(window=4, step=2, min_periods=1)
data_dict_8hr['close'] = roll.agg(lambda x : x.iloc[-1])"""

# data dict filtered by start and end date
#data_dict_1yr = {k: v[start_date:end_date] for k, v in data_dict.items()}

# create empty weights df with same index as data_dict_1yr
"""weights_1yr = pd.DataFrame(np.nan,index=data_dict_1yr["close"].index, columns=data_dict_1yr["close"].columns)
weights_1yr = weights_1yr.fillna(weights["8hr_signal"])
weights_1yr = weights_1yr.ffill()
"""
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

# create some signals based on RSI, Bollinger Bands, Stoch RSI, MACD, etc.
data.cross_sectional_feat_dict["RSI"] #lower better
data.cross_sectional_feat_dict["stoch_d"] #higher better
data.cross_sectional_feat_dict["OBV"]
data.cross_sectional_feat_dict["volume_zscore_20"]# standardized cross sectionally or not?????
data.cross_sectional_feat_dict["price_rel_ma_2"]
data.cross_sectional_feat_dict["price_rel_ma_6"]
#data.cross_sectional_feat_dict["BB_feature"]

data.normalize_cross_sectional_features(feature_names=["RSI", "stoch_d", "OBV", "price_rel_ma_2", "price_rel_ma_6"])    

# i.e. high vol, high RSI buy signal, and BB_feature == 1 
# normalize RSI
data.normalized_cross_sectional_feat_dict['RSI'] = data.normalized_cross_sectional_feat_dict['RSI'].mul(-1)
# List of features to aggregate
feature_names = [
    'RSI',
    'stoch_d',
    'OBV',
    'price_rel_ma_2',
    'price_rel_ma_6'
]
# Aggregate features by summing them across the specified feature names
aggregated_features = sum(data.normalized_cross_sectional_feat_dict[feature] for feature in feature_names)

feature_dict = {"aggregated_features": aggregated_features}



# test SimpleSortingLongOnly
start_date = datetime.datetime(2024, 1, 1)
end_date = datetime.datetime(2024, 12, 31)
signal_generator = sg.SimpleSortingLongOnly(
    config=sg.SignalConfig(window_size=1, step_size=1, min_periods=1, top_n=5, bottom_n=5), 
    signal_name="simple_sorting_signal",
    feature_name="aggregated_features",
    feature_dict=feature_dict,
    start_date=start_date,
    end_date=end_date,
    num_quantiles=5
    )

print("Calculating Weights...")
weights = signal_generator.generate_signal()
print("Finished Calculating Weights!")


# daily weights
daily_weights = {k: w.resample("D").last() for k, w in weights.items()}

#######################################################
# btc only weight matrix to check performance of btc only strategy
btc_only_weights_df = pd.DataFrame(index=weights["quintile_1"].index, columns=weights["quintile_1"].columns)
btc_only_weights_df["BTC/USDT"] = 1
backtester = Backtester(data_dict=data.cross_sectional_feat_dict, weights=btc_only_weights_df, start_date=start_date, end_date=end_date)
out = backtester.run_backtest()
metrics = backtester.performance_metrics()
print(metrics)
backtester.plot_results()
###############################


# I think this makes it too complicated, 
# for now stick to the same structure without any rolling stuff...
#data_dict = {}
#roll = data.cross_sectional_feat_dict["close"].rolling(window=4, step=2, min_periods=1)
#data_dict['close'] = roll.agg(lambda x : x.iloc[-1])

backtester = Backtester(data_dict=data.cross_sectional_feat_dict, weights=weights['quintile_1'], start_date=start_date, end_date=end_date)
out = backtester.run_backtest()
metrics = backtester.performance_metrics()
print(metrics)
backtester.plot_results()


# test if 8h close prices or 8hr rolling returns are the same



print("done")
