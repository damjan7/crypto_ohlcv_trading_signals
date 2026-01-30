import pandas as pd
from typing import List, Dict, Tuple
import datetime
from data.loader import BinanceLoader
from features.indicators import FeatureGenerator

class DataPipeline:
    def __init__(self, data_dir: str = "data_storage"):
        self.loader = BinanceLoader(data_dir=data_dir)
        self.feature_generator = FeatureGenerator()

    def load_and_process(self, 
                         pairs: List[str] = None, 
                         start_date: datetime.datetime = None, 
                         end_date: datetime.datetime = None, 
                         interval: str = "1d",
                         update_data: bool = False
                        ) -> Dict[str, pd.DataFrame]:
        
        print(f"Processing for interval {interval}..")

        if update_data:
            self.loader.update_all_data(
                start_date=start_date,
                end_date=end_date,
                interval=interval
                )
            
        # Load raw data
        raw_data = self.loader.load_data(pairs, interval)
        
        # Process features
        processed_data_dict = {}
        for pair, df in raw_data.items():
            # Filter by date if provided (though loader might have loaded all)
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
                
            if df.empty:
                print(f"Warning: Data for {pair} is empty after filtering.")
                continue
                
            # Rename volume for features if needed (loader uses 'volume', features check for it)
            # FeatureGenerator expects dataframe
            df_features = self.feature_generator.process_data(df)
            
            # Ensure proper typing/cleaning
            df_features = df_features.astype(float)
            
            processed_data_dict[pair] = df_features
            
        return processed_data_dict

    def create_data_dict_for_backtest(self, processed_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Convert dict of {pair: df_with_all_features} to 
        dict of {feature_name: df_with_pairs_as_cols}.
        Backtester and Strategy often expect this cross-sectional format.
        """
        if not processed_data:
            return {}
            
        features = list(processed_data.values())[0].columns
        
        cross_sectional_dict = {}
        pairs = list(processed_data.keys())
        
        for feature in features:
            # Gather series for this feature from all pairs
            series_list = []
            for pair in pairs:
                if feature in processed_data[pair]:
                    s = processed_data[pair][feature]
                    s.name = pair
                    series_list.append(s)
            
            if series_list:
                df_feature = pd.concat(series_list, axis=1)
                cross_sectional_dict[feature] = df_feature
                
        return cross_sectional_dict
