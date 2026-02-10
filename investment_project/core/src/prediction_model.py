import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
import os
from django.conf import settings # Import Django settings

# Update MODEL_DIR to use absolute path from settings
MODEL_DIR = os.path.join(settings.BASE_DIR, 'models')

# Ensure a 'models' directory exists to save/load models
if not os.path.exists('models'):
    os.makedirs('models')

# --- Model Loading & Prediction Logic ---
MODEL_DIR = "models"
# The Ticker map is now centralized in app.py, so it's removed from here.

# FIX: The function now accepts a ticker symbol directly.
def load_and_predict(ticker, days_ahead=7):
    """
    Loads a pre-trained model for a given ticker and predicts future stock prices.
    """
    if not ticker:
        raise ValueError("A valid ticker symbol must be provided.")

    model_path = os.path.join(MODEL_DIR, f"{ticker}_model.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model found for {ticker}. Please run train_models.py")

    # Load the trained model and its scaler
    model = load_model(model_path)
    scaler = MinMaxScaler(feature_range=(0, 1))

    # --- Fetch latest data for prediction input ---
    today = pd.to_datetime('today').strftime('%Y-%m-%d')
    # Use the provided ticker to download data
    data = yf.download(ticker, start="2018-01-01", end=today)
    if data.empty:
        raise ValueError(f"Could not fetch data for {ticker}")

    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    dataset = data[features]
    
    scaler.fit(dataset)
    scaled_data = scaler.transform(dataset)

    time_step = 60
    if len(scaled_data) < time_step:
        raise ValueError("Not enough historical data to make a prediction.")
        
    last_60_days = scaled_data[-time_step:]

    # --- Prediction Logic ---
    current_seq = last_60_days
    future_predictions_scaled = []
    n_features = len(features)

    for _ in range(days_ahead):
        X_future = current_seq.reshape(1, time_step, n_features)
        next_pred_scaled = model.predict(X_future, verbose=0)[0, 0]
        future_predictions_scaled.append(next_pred_scaled)

        new_row = current_seq[-1].copy()
        new_row[3] = next_pred_scaled # Update 'Close' price
        current_seq = np.vstack((current_seq[1:], new_row))

    # Inverse transform the 'Close' price predictions
    dummy_future = np.zeros((len(future_predictions_scaled), n_features))
    dummy_future[:, 3] = future_predictions_scaled
    future_prices = scaler.inverse_transform(dummy_future)[:, 3]

    return future_prices.tolist()

def create_dataset_multivariate(dataset, time_step=60):
    """Creates a dataset for a multivariate time series model."""
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step)])
        y.append(dataset[i + time_step, 3])  # index 3 = 'Close'
    return np.array(X), np.array(y)

def train_and_save_model(ticker):
    """
    Fetches data, trains a multivariate LSTM model, and saves it to a file.
    This is intended to be run offline periodically.
    """
    print(f"--- Starting training for {ticker} ---")
    
    # 1. Fetch and prepare data
    data = yf.download(ticker, start="2018-01-01", end=pd.to_datetime('today').strftime('%Y-%m-%d'))
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    dataset = data[features]
    
    # Normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # 2. Create training data
    time_step = 60
    X_train, y_train = create_dataset_multivariate(scaled_data, time_step)

    if len(X_train) == 0:
        print(f"Not enough data to train a model for {ticker}. Skipping.")
        return

    # 3. Build and train LSTM model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(64, return_sequences=False),
        Dense(32),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_train, y_train, batch_size=32, epochs=20, verbose=1)

    # 4. Save the trained model
    model_path = f'models/{ticker}_model.h5'
    model.save(model_path)
    print(f"✅ Model for {ticker} trained and saved to {model_path}")