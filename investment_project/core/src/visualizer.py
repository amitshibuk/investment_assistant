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


