import os
import datetime
from typing import Optional

try:
    from binance_historical_data import BinanceDataDumper
    HAS_BINANCE_DUMPER = True
except ImportError:
    HAS_BINANCE_DUMPER = False
    print("Warning: binance-historical-data package not installed. Data downloading will be disabled.")

class BinanceDownloader:
    def __init__(self, data_dir: str = "data_storage"):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

    def update_all_data(
        self, 
        start_date: datetime.date,
        end_date: datetime.date,
        interval: str = "1h"  
        ):
        """
        Updates all USDT pairs to the latest available data.
        """
        if not HAS_BINANCE_DUMPER:
            raise ImportError("binance-historical-data package is required to download data.")
            
        # Initialize Dumper
        # asset_class="spot", data_type="klines" come from original plan/docs
        dumper = BinanceDataDumper(
            path_dir_where_to_dump=self.data_dir,
            asset_class="spot",
            data_type="klines",
            data_frequency=interval,
        )
        
        print("Updating all USDT pairs (this may take time)...")
        # dump_data with is_to_update_existing=True handles the "from existing to today" logic
        dumper.dump_data(
            date_start=start_date.date() if isinstance(start_date, datetime.datetime) else start_date,
            date_end=end_date.date() if isinstance(end_date, datetime.datetime) else end_date,
            tickers=None, # None = All USDT pairs
            is_to_update_existing=True
        )
        
        # Clean up outdated daily files as recommended
        dumper.delete_outdated_daily_results()
        print("Update complete.")
