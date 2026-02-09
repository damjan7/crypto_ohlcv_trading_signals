import os
import pandas as pd
import numpy as np
from datetime import date, timedelta
import glob
import json

def process_six_data():
    # Configuration
    start_date = date(2026, 1, 12)
    today = date.today()
    downloads_dir = "downloads"
    processed_dir = "processed"
    os.makedirs(processed_dir, exist_ok=True)

    # Columns to pivot into cross-sectional dataframes
    # These are the columns of interest for analysis
    pivot_columns = [
        'ClosingPrice', 'DailyHighPrice', 'DailyLowPrice', 'OpeningPrice',
        'OnMarketVolume', 'TotalVolume', 'OffBookVolume', 'SwissAtMidVolume',
        'PreviousClosingPrice', 'LastDate'
    ]
    
    # Store daily DataFrames
    all_daily_data = []
    
    # Track missing data for reporting
    missing_files = []
    date_mismatches = []
    
    current_date = start_date
    print(f"Processing data from {start_date} to {today}...")

    while current_date <= today:
        if current_date.weekday() < 5:  # Monday=0, Friday=4
            date_str = current_date.strftime("%Y-%m-%d")
            
            # File paths
            blue_chip_file = os.path.join(downloads_dir, f"swiss_blue_chip_shares_{date_str}.csv")
            mid_small_file = os.path.join(downloads_dir, f"mid_and_small_caps_swiss_shares_{date_str}.csv")
            
            daily_dfs = []
            
            # Read Blue Chip
            if os.path.exists(blue_chip_file):
                try:
                    df_bc = pd.read_csv(blue_chip_file, sep=';')
                    daily_dfs.append(df_bc)
                except Exception as e:
                    print(f"Error reading {blue_chip_file}: {e}")
            else:
                missing_files.append(f"Blue Chip: {date_str}")

            # Read Mid/Small Cap
            if os.path.exists(mid_small_file):
                try:
                    df_ms = pd.read_csv(mid_small_file, sep=';')
                    daily_dfs.append(df_ms)
                except Exception as e:
                    print(f"Error reading {mid_small_file}: {e}")
            else:
                missing_files.append(f"Mid/Small: {date_str}")
            
            if daily_dfs:
                # Concatenate the daily files
                daily_combined = pd.concat(daily_dfs, ignore_index=True)
                daily_combined['Date'] = pd.to_datetime(date_str)
                
                # Validation: Check if LastDate matches the file date
                # Convert LastDate to string for comparison if needed, though exact semantic match is tricky with potentially different formats
                # Here we'll just store semantic checks for the report
                # LastDate format in CSV seems to be YYYYMMDD based on inspection (e.g., 20260112)
                
                # Simple check for any obviously wrong dates (e.g. from future or distant past)
                # We can also check consistency
                
                all_daily_data.append(daily_combined)

        current_date += timedelta(days=1)

    if not all_daily_data:
        print("No data found to process.")
        return

    # Combine all data
    full_df = pd.concat(all_daily_data, ignore_index=True)
    
    # Convert LastDate to datetime for better handling
    # The format in CSV inspection was YYYYMMDD (integer or string)
    full_df['LastDate_Parsed'] = pd.to_datetime(full_df['LastDate'], format='%Y%m%d', errors='coerce')
    
    # Generate Cross-Sectional DataFrames
    print("\nGenerating cross-sectional DataFrames...")
    
    validation_report = []
    
    for col in pivot_columns:
        if col not in full_df.columns:
            print(f"Warning: Column {col} not found in data.")
            continue
            
        # Pivot: Index=Date, Columns=ISIN, Values=Col
        # Using ISIN as the unique identifier for stocks
        try:
            pivot_df = full_df.pivot_table(index='Date', columns='ISIN', values=col)
            
            # Save to CSV
            output_file = os.path.join(processed_dir, f"{col}.csv")
            pivot_df.to_csv(output_file)
            print(f"Saved {output_file}")
            
            # Validation for this metric
            missing_val_count = pivot_df.isna().sum().sum()
            total_cells = pivot_df.size
            missing_pct = (missing_val_count / total_cells) * 100
            
            validation_report.append({
                'Metric': col,
                'Shape': pivot_df.shape,
                'Missing Values': missing_val_count,
                'Missing %': f"{missing_pct:.2f}%"
            })
            
        except Exception as e:
            print(f"Error pivoting {col}: {e}")

    # --- Validation Summary ---
    print("\n" + "="*30)
    print("      DATA VALIDATION REPORT      ")
    print("="*30)
    
    print(f"\nTime Range: {start_date} to {today}")
    print(f"Total Daily Files Processed: {len(all_daily_data)}")
    
    if missing_files:
        print("\nMISSING FILES (Weekdays):")
        for f in missing_files:
            print(f"  - {f}")
    else:
        print("\nNo missing weekday files found.")

    # Check download log for historical holes
    log_file = "download_log.json"
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                log_data = json.load(f)
                missing_dates_log = log_data.get("missing_dates", [])
                if missing_dates_log:
                    print("\nKNOWN DATA HOLES (from download log):")
                    for d in missing_dates_log:
                        print(f"  - {d}")
                else:
                    print("\nNo known data holes in download log.")
        except Exception as e:
            print(f"Error reading download log: {e}")
        
    print("\nCROSS-SECTIONAL DATAFRAME STATS:")
    report_df = pd.DataFrame(validation_report)
    if not report_df.empty:
        print(report_df.to_string(index=False))
        
    # Check for Date Mismatches (LastDate vs File Date)
    # File Date is in 'Date' column
    # LastDate from csv is in 'LastDate_Parsed'
    # We check if they differ
    
    mismatches = full_df[full_df['Date'] != full_df['LastDate_Parsed']]
    if not mismatches.empty:
        print(f"\nDATE MISMATCHES WARNING: Found {len(mismatches)} rows where 'LastDate' in CSV != File Date.")
        print("This might happen if there was no trading for a stock on that specific day (stale price).")
        # Sample
        print("Sample mismatches:")
        print(mismatches[['Date', 'LastDate', 'ISIN', 'ShortName']].head().to_string(index=False))
    else:
        print("\nDates consistency check: OK (All LastDate entries match the file date).")

    print("\nProcessing complete.")

if __name__ == "__main__":
    process_six_data()
