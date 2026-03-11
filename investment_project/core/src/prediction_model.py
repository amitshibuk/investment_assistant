import yfinance as yf
import pandas as pd
import numpy as np
import os
import joblib  # NEW: For saving/loading the scaler
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
from django.conf import settings

# --- Configuration ---
# Use absolute path to ensure models are found regardless of where code is run
MODEL_DIR = os.path.join(settings.BASE_DIR, 'models')

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# ==========================================
#  TRAINING LOGIC
# ==========================================

def create_dataset(dataset, time_step=60):
    """Creates X, y datasets from a single column (Univariate)."""
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

def train_and_save_model(ticker):
    """
    Trains a Univariate LSTM model (Close price only) and saves
    BOTH the model (.h5) and the scaler (.pkl).
    """
    print(f"--- Starting training for {ticker} ---")
    
    # 1. Fetch Data
    # Download more history to ensure better training
    data = yf.download(ticker, start="2015-01-01", end=pd.to_datetime('today').strftime('%Y-%m-%d'))
    
    if len(data) < 200:
        print(f"❌ Not enough data for {ticker}. Skipping.")
        return

    # 2. Prepare Data (UNIVARIATE FIX)
    # We only use 'Close' price. This fixes the recursion issue.
    dataset = data[['Close']].values.astype(float)
    
    # 3. Scale Data & SAVE THE SCALER (SCALER FIX)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Save the scaler so we can reproduce this scale during prediction
    scaler_path = os.path.join(MODEL_DIR, f"{ticker}_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"   Scaler saved to {scaler_path}")

    # 4. Create Training Splits
    time_step = 60
    X, y = create_dataset(scaled_data, time_step)

    # Reshape for LSTM [samples, time steps, features]
    # Features = 1 because we are only using 'Close'
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # 5. Build LSTM Model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train
    model.fit(X, y, batch_size=64, epochs=10, verbose=1)

    # 6. Save Model
    model_path = os.path.join(MODEL_DIR, f"{ticker}_model.h5")
    model.save(model_path)
    print(f"✅ Model for {ticker} trained and saved to {model_path}")


# ==========================================
#  PREDICTION LOGIC
# ==========================================

def load_and_predict(ticker, days_ahead=7):
    """
    Loads the model + scaler and predicts future prices recursively.
    """
    if not ticker:
        raise ValueError("Ticker symbol required.")

    model_path = os.path.join(MODEL_DIR, f"{ticker}_model.h5")
    scaler_path = os.path.join(MODEL_DIR, f"{ticker}_scaler.pkl")

    # 1. Check if files exist — do NOT train on the fly; let the caller handle it
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"No trained model found for {ticker}. Please add it to your portfolio and use Auto-Train.")

    # 2. Load Resources
    model = load_model(model_path)
    scaler = joblib.load(scaler_path) # Load the EXACT scaler used in training

    # 3. Get Recent Data
    # Fetch enough past data to cover the lookback window (60 days)
    # We fetch 100 days to be safe
    end_date = pd.to_datetime('today')
    start_date = end_date - pd.Timedelta(days=150)
    
    data = yf.download(ticker, start=start_date, end=end_date)
    
    if data.empty:
        raise ValueError(f"No live data found for {ticker}")

    # Prepare input (Must match training structure: 'Close' only)
    dataset = data[['Close']].values.astype(float)
    
    # Scale using the LOADED scaler
    scaled_data = scaler.transform(dataset)

    time_step = 60
    if len(scaled_data) < time_step:
        raise ValueError(f"Not enough recent data for {ticker} (Need 60 days).")

    # Get the last 60 days as the starting point
    current_batch = scaled_data[-time_step:].reshape(1, time_step, 1)
    future_predictions_scaled = []

    # 4. Recursive Prediction Loop
    for _ in range(days_ahead):
        # next_pred_scaled comes out as a scalar or shape (1,)
        next_pred_scaled = model.predict(current_batch, verbose=0)[0, 0]
        
        future_predictions_scaled.append(next_pred_scaled)

        # CORRECTED DIMENSION HANDLING:
        # 1. Reshape the prediction to match the feature dimension: (1, 1, 1)
        new_step = np.array(next_pred_scaled).reshape(1, 1, 1)
        
        # 2. Concatenate along the time axis (axis=1) 
        # current_batch[:, 1:, :] removes the oldest day, keeping it (1, 59, 1)
        # np.append or np.concatenate adds the new day, making it (1, 60, 1) again
        current_batch = np.concatenate([current_batch[:, 1:, :], new_step], axis=1)

    # 5. Inverse Transform to get actual Dollar prices
    # Convert the list of predictions to a NumPy array and reshape to 2D (N rows, 1 column)
    predictions_reshaped = np.array(future_predictions_scaled).reshape(-1, 1)
    
    # Now inverse_transform will work correctly
    future_prices = scaler.inverse_transform(predictions_reshaped)

    # Flatten back to a simple list for the JSON response: [150.5, 151.2, ...]
    return future_prices.flatten().tolist()