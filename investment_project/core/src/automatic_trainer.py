import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'investment_project.settings')
django.setup()
from prediction_model import train_and_save_model
import time

# List of tickers you want to train models for
TICKERS_TO_TRAIN = ["AAPL", "MSFT", "AMZN", "TSLA"]

if __name__ == '__main__':
    start_time = time.time()
    print("--- Starting Offline Model Training Job ---")
    
    for ticker in TICKERS_TO_TRAIN:
        try:
            train_and_save_model(ticker)
        except Exception as e:
            print(f"❌ Failed to train model for {ticker}. Error: {e}")
            
    end_time = time.time()
    print(f"--- Offline Training Job Finished in {end_time - start_time:.2f} seconds ---")
