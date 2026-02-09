import os
import requests
from datetime import date, timedelta
import time

def fetch_six_data():
    # Start date as specified
    start_date = date(2026, 1, 12)
    today = date.today()
    
    # URL patterns with placeholders for the date
    url_patterns = [
        "http://www.six-group.com/sheldon/historical_prices/v1/download/swiss_blue_chip_shares_{date_str}.csv",
        "https://www.six-group.com/sheldon/historical_prices/v1/download/mid_and_small_caps_swiss_shares_{date_str}.csv"
    ]
    
    # Create a directory to store the downloads
    output_dir = "downloads"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting download from {start_date} to {today}...")
    
    current_date = start_date
    while current_date <= today:
        # Check if it's a weekday (Monday=0, Sunday=6)
        if current_date.weekday() < 5:
            date_str = current_date.strftime("%Y-%m-%d")
            print(f"Processing date: {date_str}")
            
            for pattern in url_patterns:
                url = pattern.format(date_str=date_str)
                filename = url.split('/')[-1]
                filepath = os.path.join(output_dir, filename)
                
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
                    else:
                        print(f"  Failed with status code {response.status_code}: {url}")
                    
                    # Be polite to the server
                    time.sleep(0.5)
                    
                except requests.exceptions.RequestException as e:
                    print(f"  Request Error for {url}: {e}")
                except Exception as e:
                    print(f"  Unexpected Error for {url}: {e}")
        
        current_date += timedelta(days=1)
    
    print("Download process completed.")

if __name__ == "__main__":
    fetch_six_data()
