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
import helper_functions as hf


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
# return_1, return_2, return_6, return_12 (1, 2, 6, 12 periods, i.e. 6hr periods now)
# Relative deviations (price_rel_ma_*) are computed as the difference relative to the moving average
# df['close'] / df[f'price_ma_{window}'] - 1

# plot some MA's for XRP/USDT
xrp = data.feature_dict["XRP/USDT"].loc[:, ["price_ma_2", "price_ma_6", "price_ma_12", "price_ma_24", "price_ma_48", "close"]]
fig = px.line(xrp, title="Price Moving Averages (XRP/USDT)")
fig.show()

# simple strat:
# buy price_rel_ma_24 > price_rel_ma_48 
# else go short

buy_signal = data.cross_sectional_feat_dict['price_rel_ma_24'] > data.cross_sectional_feat_dict['price_rel_ma_48']
sell_signal = data.cross_sectional_feat_dict['price_rel_ma_24'] < data.cross_sectional_feat_dict['price_rel_ma_48']

# Create equal-weight buy signal weights
# Replace True with 1/(sum of True in row) and False with np.nan
buy_signal_weights = buy_signal.copy()
for i in range(len(buy_signal_weights)):
    row_sum = buy_signal_count[i]
    if row_sum > 0:  # Avoid division by zero
        buy_signal_weights.iloc[i, :] = buy_signal_weights.iloc[i, :].map(lambda x: 1/row_sum if x else np.nan)
    else:
        buy_signal_weights.iloc[i, :] = np.nan


############# CREATE LONG SHORT WEIGHTS #############
# Count number of True values in each row of buy and sell signals
buy_signal_count = buy_signal.sum(axis=1)
sell_signal_count = sell_signal.sum(axis=1)

# Create long-short signal weights
# Buy signals: Replace True with +1/(sum of True in row) and False with 0
# Sell signals: Replace True with -1/(sum of True in row) and False with 0
long_short_weights = pd.DataFrame(0, index=buy_signal.index, columns=buy_signal.columns)

for i in range(len(long_short_weights)):
    # Process buy signals (positive weights)
    buy_row_sum = buy_signal_count[i]
    if buy_row_sum > 0:  # Avoid division by zero
        for j in range(len(long_short_weights.columns)):
            if buy_signal.iloc[i, j]:
                long_short_weights.iloc[i, j] = 1/buy_row_sum
    
    # Process sell signals (negative weights)
    sell_row_sum = sell_signal_count[i]
    if sell_row_sum > 0:  # Avoid division by zero
        for j in range(len(long_short_weights.columns)):
            if sell_signal.iloc[i, j]:
                long_short_weights.iloc[i, j] = -1/sell_row_sum

# Replace zeros with np.nan for cleaner representation (optional)
long_short_weights = long_short_weights.replace(0, np.nan)
#####################################################


############# CREATE FILTERED LONG ONLY WEIGHTS #############
# Create a new long-only table with minimum participation requirement
# Only add weights if at least 10% of non-NA assets have buy signals
# First, identify non-NA assets in each row
valid_assets = ~data.cross_sectional_feat_dict["close"].isna()
valid_count_per_row = valid_assets.sum(axis=1)

# Calculate buy signals
buy_signal = data.cross_sectional_feat_dict['price_rel_ma_24'] > data.cross_sectional_feat_dict['price_rel_ma_48']

# Count buy signals per row
buy_signal_count = buy_signal.sum(axis=1)

# Calculate percentage of non-NA assets with buy signals
buy_signal_percentage = buy_signal_count / valid_count_per_row

# Create the filtered long-only weights
filtered_long_weights = buy_signal.copy()

# Process each row
for i in range(len(filtered_long_weights)):
    # Check if at least 10% of non-NA assets have buy signals
    if buy_signal_percentage[i] >= 0.10:
        # If yes, distribute weights equally among buy signals
        row_sum = buy_signal_count[i]
        if row_sum > 0:  # Avoid division by zero (should always be true given the 10% check)
            filtered_long_weights.iloc[i, :] = filtered_long_weights.iloc[i, :].map(lambda x: 1/row_sum if x else np.nan)
    else:
        # If less than 10% have buy signals, set all weights to NaN (no trading)
        filtered_long_weights.iloc[i, :] = np.nan
#####################################################


############# CREATE FILTERED LONG SHORT WEIGHTS #############
# Create a filtered long-short table with minimum participation requirement
# Only add weights if at least 10% of non-NA assets have signals (buy or sell)

# First, identify non-NA assets in each row
valid_assets = ~data.cross_sectional_feat_dict["close"].isna()
valid_count_per_row = valid_assets.sum(axis=1)

# Calculate buy and sell signals
buy_signal = data.cross_sectional_feat_dict['price_rel_ma_24'] > data.cross_sectional_feat_dict['price_rel_ma_48']
sell_signal = data.cross_sectional_feat_dict['price_rel_ma_24'] < data.cross_sectional_feat_dict['price_rel_ma_48']

# Count buy and sell signals per row
buy_signal_count = buy_signal.sum(axis=1)
sell_signal_count = sell_signal.sum(axis=1)

# Calculate percentage of non-NA assets with buy or sell signals
buy_signal_percentage = buy_signal_count / valid_count_per_row
sell_signal_percentage = sell_signal_count / valid_count_per_row

# Create the filtered long-short weights
filtered_long_short_weights = pd.DataFrame(0, index=buy_signal.index, columns=buy_signal.columns)

# Process each row
for i in range(len(filtered_long_short_weights)):
    # Check if at least 10% of non-NA assets have buy signals
    buy_percentage = buy_signal_percentage[i]
    sell_percentage = sell_signal_percentage[i]
    
    # Only proceed if either buy or sell signals meet the 10% threshold
    if buy_percentage >= 0.10 or sell_percentage >= 0.10:
        # Process buy signals (positive weights) if they meet threshold
        if buy_percentage >= 0.10:
            buy_row_sum = buy_signal_count[i]
            for j in range(len(filtered_long_short_weights.columns)):
                if buy_signal.iloc[i, j]:
                    filtered_long_short_weights.iloc[i, j] = 1/buy_row_sum
        
        # Process sell signals (negative weights) if they meet threshold
        if sell_percentage >= 0.10:
            sell_row_sum = sell_signal_count[i]
            for j in range(len(filtered_long_short_weights.columns)):
                if sell_signal.iloc[i, j]:
                    filtered_long_short_weights.iloc[i, j] = -1/sell_row_sum
    else:
        # If neither buy nor sell signals meet the 10% threshold, set all weights to NaN (no trading)
        filtered_long_short_weights.iloc[i, :] = np.nan

# Replace zeros with np.nan for cleaner representation
filtered_long_short_weights = filtered_long_short_weights.replace(0, np.nan)
#####################################################


# run simple backtest
start_date = datetime.datetime(2020, 1, 1)
end_date = datetime.datetime(2024, 12, 31)
backtester = Backtester(data_dict=data.cross_sectional_feat_dict, weights=buy_signal_weights, start_date=start_date, end_date=end_date)
out = backtester.run_backtest()
metrics = backtester.performance_metrics()
print(metrics)
backtester.plot_results()

# run long short backtest
backtester = Backtester(data_dict=data.cross_sectional_feat_dict, weights=long_short_weights, start_date=start_date, end_date=end_date)
out = backtester.run_backtest()
metrics = backtester.performance_metrics()
print(metrics)
backtester.plot_results()

# btc only
btc_only_weights = pd.DataFrame(index=buy_signal_weights.index, columns=buy_signal_weights.columns)
btc_only_weights["BTC/USDT"] = 1
backtester = Backtester(data_dict=data.cross_sectional_feat_dict, weights=btc_only_weights, start_date=start_date, end_date=end_date)
out = backtester.run_backtest()
metrics = backtester.performance_metrics()
print(metrics)
backtester.plot_results()

# run filtered long only backtest
backtester = Backtester(data_dict=data.cross_sectional_feat_dict, weights=filtered_long_weights, start_date=start_date, end_date=end_date)
out = backtester.run_backtest()
metrics = backtester.performance_metrics()
print(metrics)
backtester.plot_results()

# run filtered long short backtest
backtester = Backtester(data_dict=data.cross_sectional_feat_dict, weights=filtered_long_short_weights, start_date=start_date, end_date=end_date)
out = backtester.run_backtest()
metrics = backtester.performance_metrics()
print(metrics)
backtester.plot_results()


#### Same analysis but with single token, XRP/USDT
buy_signal_xrp = buy_signal.copy()['XRP/USDT']
sell_signal_xrp = sell_signal.copy()['XRP/USDT']

# Create equal-weight buy signal weights
xrp_only_weights = pd.DataFrame(index=buy_signal.index, columns=buy_signal.columns)
xrp_only_weights["XRP/USDT"] = np.where(buy_signal_xrp, 1, np.nan)

# run xrp only backtest
backtester = Backtester(data_dict=data.cross_sectional_feat_dict, weights=xrp_only_weights, start_date=start_date, end_date=end_date)
out = backtester.run_backtest()
metrics = backtester.performance_metrics()
print(metrics)
backtester.plot_results()

# compare to xrp constant 100% weights
xrp_only_weights_100 = pd.DataFrame(index=buy_signal.index, columns=buy_signal.columns)
xrp_only_weights_100["XRP/USDT"] = 1
backtester = Backtester(data_dict=data.cross_sectional_feat_dict, weights=xrp_only_weights_100, start_date=start_date, end_date=end_date)
out = backtester.run_backtest()
metrics = backtester.performance_metrics()
print(metrics)
backtester.plot_results()


# trade asset j if:
# 1. price_rel_ma_12 > 0
# 2. price_rel_ma_4 > 0
# 2. volume_ma_2 > 1.5 * volume_ma_48
# 3. volume_ma_2 > 1.5 * volume_ma_12


# plot vol features to see spikes in volume and maybe trend
hf.plot_timeseries(data.cross_sectional_feat_dict, title="Volume Moving Average 2", height=600, width=1000)

# plot volume ma 2, 6, 12, 24, 48 for XRP USDT
xrp = data.feature_dict["XRP/USDT"].loc[:, ["volume_ma_2", "volume_ma_6", "volume_ma_12", "volume_ma_24", "volume_ma_48", "close"]]

# normalize volumes by close
xrp_norm = xrp.copy()
vol_cols = ["volume_ma_2", "volume_ma_6", "volume_ma_12", "volume_ma_24", "volume_ma_48"]
xrp_norm[vol_cols] = xrp[vol_cols].div(xrp["close"], axis=0)
fig = px.line(xrp_norm, title="Volume Moving Average 2 for XRPUSDT")
fig.show()

# z-score normalization
xrp_norm2 = xrp.copy()
for col in vol_cols:
    xrp_norm2[col] = (xrp[col] - xrp[col].mean()) / xrp[col].std()
fig2 = px.line(xrp_norm2, vol_cols, title="Z-Score Normalized Volume MAs (XRP/USDT)")
fig2.show()


#  
















