import os
import requests
from datetime import date, timedelta, datetime
import time
import json
import glob

DOWNLOADS_DIR = "downloads"
LOG_FILE = "download_log.json"

def get_start_date(base_start_date):
    """Determine the start date based on existing files."""
    files = glob.glob(os.path.join(DOWNLOADS_DIR, "*_20*.csv"))
    if not files:
        return base_start_date
    
    dates = []
    for f in files:
        try:
            # Extract date from filename (e.g., ..._2026-01-12.csv)
            date_str = f.split('_')[-1].replace('.csv', '')
            dates.append(datetime.strptime(date_str, "%Y-%m-%d").date())
        except ValueError:
            continue
            
    if not dates:
        return base_start_date
        
    last_date = max(dates)
    next_date = last_date + timedelta(days=1)
    
    # If next date is in future, stop
    if next_date > date.today():
        return None
        
    return next_date

def load_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            return json.load(f)
    return {"missing_dates": []}

def save_log(log_data):
    with open(LOG_FILE, 'w') as f:
        json.dump(log_data, f, indent=4, default=str)

def fetch_six_data():
    base_start = date(2026, 1, 12)
    start_date = get_start_date(base_start)
    today = date.today()
    
    if start_date is None:
        print("Data is up to date.")
        return

    # URL patterns with placeholders for the date
    url_patterns = [
        "http://www.six-group.com/sheldon/historical_prices/v1/download/swiss_blue_chip_shares_{date_str}.csv",
        "https://www.six-group.com/sheldon/historical_prices/v1/download/mid_and_small_caps_swiss_shares_{date_str}.csv"
    ]
    
    os.makedirs(DOWNLOADS_DIR, exist_ok=True)
    log_data = load_log()
    missing_dates = set(log_data.get("missing_dates", []))
    
    print(f"Starting download from {start_date} to {today}...")
    
    current_date = start_date
    while current_date <= today:
        # Check if it's a weekday (Monday=0, Sunday=6)
        if current_date.weekday() < 5:
            date_str = current_date.strftime("%Y-%m-%d")
            print(f"Processing date: {date_str}")
            
            day_missing = False
            
            for pattern in url_patterns:
                url = pattern.format(date_str=date_str)
                filename = url.split('/')[-1]
                filepath = os.path.join(DOWNLOADS_DIR, filename)
                
                if os.path.exists(filepath):
                    print(f"  Skipping {filename} (already exists)")
                    continue
                
                print(f"  Downloading {url}...")
                try:
                    response = requests.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        with open(filepath, 'wb') as f:
                            f.write(response.content)
                        print(f"  Saved to {filepath}")
                    elif response.status_code == 404:
                         print(f"  File not found on server (404): {url}")
                         day_missing = True
                    else:
                        print(f"  Failed with status code {response.status_code}: {url}")
                    
                    # Be polite to the server
                    time.sleep(0.5)
                    
                except requests.exceptions.RequestException as e:
                    print(f"  Request Error for {url}: {e}")
                except Exception as e:
                    print(f"  Unexpected Error for {url}: {e}")

            if day_missing:
                missing_dates.add(date_str)
        
        current_date += timedelta(days=1)
    
    log_data["missing_dates"] = sorted(list(missing_dates))
    log_data["last_run"] = datetime.now().isoformat()
    save_log(log_data)
    
    print("Download process completed.")

if __name__ == "__main__":
    fetch_six_data()
