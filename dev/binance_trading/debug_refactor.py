
try:
    from data.loader import BinanceLoader
    from data.downloader import BinanceDownloader
    from data.pipeline import DataPipeline
    print("Imports successful.")
except ImportError as e:
    print(f"Import failed: {e}")
    exit(1)

try:
    loader = BinanceLoader()
    print("BinanceLoader instantiated.")
    downloader = BinanceDownloader()
    print("BinanceDownloader instantiated.")
    pipeline = DataPipeline()
    print("DataPipeline instantiated.")
except Exception as e:
    print(f"Instantiation failed: {e}")
    exit(1)

print("Verification complete.")
