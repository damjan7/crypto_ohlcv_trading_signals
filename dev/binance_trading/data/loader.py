import os
import pandas as pd
from typing import List, Optional
import datetime

class BinanceLoader:
    def __init__(self, data_dir: str = "data_storage"):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

    def load_data(
        self, 
        pairs: List[str] = None, 
        interval: str = "1h"
        ) -> pd.DataFrame:
        """
        Load data from disk. If pairs is None, finds all available pairs.
        """
        data = {}
        
        # Expected structure: {data_dir}/spot/monthly/klines/{pair}/{interval}
        # But we need to support recursive search if structure varies.
        base_path = os.path.join("data_storage", "spot", "monthly", "klines")
        target_pairs = None
        if target_pairs is None:
            if os.path.exists(base_path):
                 target_pairs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
            else:
                 # Try to find any folder if base path doesn't exist (maybe different version)
                 target_pairs = []
                 print(f"Base path {base_path} not found. Searching recursively...")
                 # This is a fallback to find folders that look like tickers
                 pass

        if not target_pairs:
             print("No pairs found or provided.")
             
        print(f"Loading data for {len(target_pairs)} pairs...")
        for pair in target_pairs:
            # Construct possible path
            pair_path = os.path.join(base_path, pair, interval)
            
            # DFS for CSVs
            found_csvs = []
            if os.path.exists(pair_path):
                 for f in sorted(os.listdir(pair_path)):
                      if f.endswith(".csv"):
                           found_csvs.append(os.path.join(pair_path, f))
            
            if not found_csvs:
                 # Fallback search if path is different
                 # print(f"No CSVs found in standard path for {pair}, searching...")
                 pass
            
            dfs = []
            for f in found_csvs:
                try:
                    # Generic headerless load
                    df_chunk = pd.read_csv(f, header=None)
                    if len(df_chunk.columns) == 12:
                        df_chunk.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', "close_time", "quote_asset_volume", 
                        "num_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"]
                        dfs.append(df_chunk)
                    else:
                        AssertionError("Data Shapes have changed from 12 to something else, please check.")
                except Exception as e:
                    AssertionError("Error in data loading forat------")
                    pass
            
            if dfs:
                final_df = pd.concat(dfs)
                # there's datetime objects that are not in correct format (i.e. not ms)
                # timestamps larger than 10^14 are likely microseconds (16 digits)
                # Current time in ms is ~1.7e12, so 1e14 is a safe cutoff for the next few thousand years.
                microsecond_mask = final_df['open_time'] > 1e14
                # Divide only the microsecond rows by 1000 to normalize them to ms
                # We use // for integer division to keep the data type consistent
                final_df.loc[microsecond_mask, 'open_time'] = final_df.loc[microsecond_mask, 'open_time'] // 1000
                final_df['open_time'] = pd.to_datetime(final_df['open_time'], unit='ms')
                final_df.set_index('open_time', inplace=True)
                final_df.sort_index(inplace=True)
                final_df = final_df[~final_df.index.duplicated(keep='first')]
                # Numeric conversion
                for col in ['open', 'high', 'low', 'close', 'volume', "quote_asset_volume", 
                        "num_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume",]:
                    final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
                
                data[pair] = final_df
                
        return data
