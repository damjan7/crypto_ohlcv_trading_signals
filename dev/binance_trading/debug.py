import sys
import os
import datetime
import pandas as pd

# Add the current directory to sys.path to ensure imports work if run from here
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.loader import BinanceLoader
loader = BinanceLoader()
data = loader.load_data(pairs = None, interval='1d')

from data.pipeline import DataPipeline  
pipeline = DataPipeline()


print("")

# Create Cross-Sectional Data Dictionary
print("Creating Cross-Sectional Data...")
cs_data = pipeline.create_data_dict_for_backtest(data)
cs_data['returns'] = cs_data['close'].pct_change(fill_method=None) 

# year filter tmp
yr_2025_mask = (1+cs_data['returns']['BTCUSDT']).index.year == 2025
yr_2021_mask = (1+cs_data['returns']['BTCUSDT']).index.year == 2021