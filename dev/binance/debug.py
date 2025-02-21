import feature_generator as fg
import pandas as pd
import polars as pl
import os
import numpy as np
import plotly.express as px
from typing import List, Dict, Tuple

import data_processor as datproc

print(os.getcwd())

print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

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

print("done")
