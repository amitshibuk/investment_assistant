import plotly.graph_objects as go
import plotly.io as pio
import yfinance as yf
import pandas as pd
import json
from datetime import timedelta

def fetch_historical_context(ticker, period='6mo'):
    data = yf.download(
        ticker,
        period=period,
        auto_adjust=False
    )
    return data

def generate_interactive_chart(ticker, predicted_prices):

    history = yf.download(ticker, period='6mo', auto_adjust=False)

    if history.empty:
        return None

    # 🔥 Flatten MultiIndex columns
    if isinstance(history.columns, pd.MultiIndex):
        history.columns = history.columns.get_level_values(0)

    history = history.reset_index()

    print("DEBUG CLOSE MIN MAX:",
      history['Close'].min(),
      history['Close'].max())

    print("DEBUG FIRST 5 CLOSE VALUES:",
      history['Close'].head().tolist())


    last_real_date = history['Date'].iloc[-1]
    future_dates = [last_real_date + timedelta(days=i+1) for i in range(len(predicted_prices))]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=history['Date'].tolist(),
        y=history['Close'].tolist(),
        mode='lines',
        name='Real Historical Price',
        line=dict(color='#22d3ee', width=2)
    ))

    predict_x = [last_real_date] + future_dates
    predict_y = [history['Close'].iloc[-1]] + predicted_prices

    fig.add_trace(go.Scatter(
        x=predict_x,
        y=predict_y,
        mode='lines+markers',
        name='AI Forecast',
        line=dict(color='#f97316', width=2, dash='dot'),
        marker=dict(size=4)
    ))

    fig.update_layout(
        title=f'{ticker} Analysis: History vs. AI Forecast',
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x unified"
    )

    import json
    from plotly.utils import PlotlyJSONEncoder

    return json.loads(
        json.dumps(fig, cls=PlotlyJSONEncoder)
    )

def generate_candlestick_chart(ticker):
    """
    Generates a Plotly Candlestick chart for the portfolio view.
    NO PREDICTIONS. Just historical data representation.
    """
    print(f"DEBUG: Generating chart for {ticker}")
    import datetime
    try:
        # Download data with a small buffer to ensure we catch today
        # period='6mo' usually works, but '1d' interval might lag by a day if not 'prepost'
        # Let's try adding prepost=True to gauge if that helps, but usually standard download is fine.
        # The issue might be dropna() removing the current partial day.
        history = yf.download(ticker, period='6mo', interval='1d', auto_adjust=False)

        if history.empty:
            print(f"DEBUG: Empty history for {ticker}")
            return None

        # Flatten columns if MultiIndex
        if isinstance(history.columns, pd.MultiIndex):
            history.columns = history.columns.get_level_values(0)

        # Reset index to get Date column
        history = history.reset_index()

        # Clean Data: Only drop if critical price data is missing
        history = history.dropna(subset=['Open', 'High', 'Low', 'Close'])

        # Explicit Type Conversion to native Python types
        dates = history['Date'].dt.strftime('%Y-%m-%d').tolist()
        opens = history['Open'].astype(float).tolist()
        highs = history['High'].astype(float).tolist()
        lows = history['Low'].astype(float).tolist()
        closes = history['Close'].astype(float).tolist()

        if not dates:
             return None

        print(f"DEBUG: {ticker} - {len(dates)} rows. Last Date: {dates[-1]}")

        fig = go.Figure()

        fig.add_trace(go.Candlestick(
            x=dates,
            open=opens,
            high=highs,
            low=lows,
            close=closes,
            name=ticker,
            increasing_line_color='#22c55e',
            decreasing_line_color='#ef4444'
        ))

        fig.update_layout(
            title=f'{ticker} Analysis (6 Months)',
            template="plotly_dark",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            margin=dict(l=50, r=50, t=50, b=50),
            height=600,
            xaxis=dict(
                type='date',  # Explicitly tell Plotly this is a date axis
                autorange=True
            ),
            yaxis=dict(autorange=True)
        )

        return json.loads(fig.to_json())

    except Exception as e:
        print(f"DEBUG: Error generating chart: {e}")
        import traceback
        traceback.print_exc()
        return None


