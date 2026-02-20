import plotly.graph_objects as go
import plotly.io as pio
import yfinance as yf
import pandas as pd
import json
import numpy as np
from datetime import timedelta


def fetch_historical_context(ticker, period='6mo'):
    data = yf.download(
        ticker,
        period=period,
        auto_adjust=False
    )
    return data


def generate_interactive_chart(ticker, predicted_prices):
    """
    Generates an enhanced prediction chart with:
    - Filled area gradient behind historical close line
    - 50-day Moving Average line
    - Shaded forecast zone
    - Seamless connection between history and prediction
    """
    history = yf.download(ticker, period='6mo', auto_adjust=False)

    if history.empty:
        return None

    # Flatten MultiIndex columns
    if isinstance(history.columns, pd.MultiIndex):
        history.columns = history.columns.get_level_values(0)

    history = history.reset_index()

    historical_dates = history['Date'].tolist()
    historical_close = history['Close'].tolist()

    last_real_date = history['Date'].iloc[-1]
    future_dates = [last_real_date + timedelta(days=i + 1) for i in range(len(predicted_prices))]

    # Build prediction line starting from the last known close
    predict_x = [last_real_date] + future_dates
    predict_y = [history['Close'].iloc[-1]] + predicted_prices

    # 50-day Moving Average
    close_series = history['Close'].astype(float)
    ma50 = close_series.rolling(window=50).mean().tolist()

    fig = go.Figure()

    # --- Historical Close (filled area) ---
    fig.add_trace(go.Scatter(
        x=historical_dates,
        y=historical_close,
        mode='lines',
        name='Close Price',
        line=dict(color='#22d3ee', width=2),
        fill='tozeroy',
        fillcolor='rgba(34, 211, 238, 0.08)',
    ))

    # --- 50-day MA ---
    fig.add_trace(go.Scatter(
        x=historical_dates,
        y=ma50,
        mode='lines',
        name='50-Day MA',
        line=dict(color='#a78bfa', width=1.5, dash='dot'),
        opacity=0.7,
    ))

    # --- Forecast shaded zone ---
    upper_band = [p * 1.03 for p in predict_y]
    lower_band = [p * 0.97 for p in predict_y]

    fig.add_trace(go.Scatter(
        x=predict_x + predict_x[::-1],
        y=upper_band + lower_band[::-1],
        fill='toself',
        fillcolor='rgba(249, 115, 22, 0.10)',
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=False,
        hoverinfo='skip',
        name='Forecast Range',
    ))

    # --- AI Forecast line ---
    fig.add_trace(go.Scatter(
        x=predict_x,
        y=predict_y,
        mode='lines+markers',
        name='AI Forecast',
        line=dict(color='#f97316', width=2.5),
        marker=dict(size=5, color='#f97316', symbol='circle'),
    ))

    fig.update_layout(
        title=dict(text=f'{ticker} — History & AI Forecast', font=dict(size=14, color='#e5e7eb')),
        template="plotly_dark",
        paper_bgcolor='rgba(17,24,39,1)',
        plot_bgcolor='rgba(17,24,39,1)',
        xaxis=dict(
            title="Date",
            gridcolor='rgba(55,65,81,0.5)',
            showgrid=True,
        ),
        yaxis=dict(
            title="Price (USD)",
            gridcolor='rgba(55,65,81,0.5)',
            showgrid=True,
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=50, r=20, t=60, b=50),
    )

    from plotly.utils import PlotlyJSONEncoder
    return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))


def generate_candlestick_chart(ticker):
    """
    Generates a polished Plotly Candlestick chart for the portfolio view,
    with gradient shading and a 20-day MA overlay.
    """
    import datetime
    try:
        history = yf.download(ticker, period='6mo', interval='1d', auto_adjust=False)

        if history.empty:
            return None

        if isinstance(history.columns, pd.MultiIndex):
            history.columns = history.columns.get_level_values(0)

        history = history.reset_index()
        history = history.dropna(subset=['Open', 'High', 'Low', 'Close'])

        dates = history['Date'].dt.strftime('%Y-%m-%d').tolist()
        opens = history['Open'].astype(float).tolist()
        highs = history['High'].astype(float).tolist()
        lows = history['Low'].astype(float).tolist()
        closes = history['Close'].astype(float).tolist()
        volumes = history['Volume'].astype(float).tolist() if 'Volume' in history.columns else []

        if not dates:
            return None

        # 20-day MA
        close_series = pd.Series(closes)
        ma20 = close_series.rolling(window=20).mean().tolist()

        fig = go.Figure()

        # --- Volume bars (secondary y-axis) ---
        if volumes:
            vol_colors = [
                'rgba(34,197,94,0.25)' if closes[i] >= opens[i] else 'rgba(239,68,68,0.25)'
                for i in range(len(closes))
            ]
            fig.add_trace(go.Bar(
                x=dates,
                y=volumes,
                name='Volume',
                marker_color=vol_colors,
                yaxis='y2',
                showlegend=False,
            ))

        # --- Candlestick ---
        fig.add_trace(go.Candlestick(
            x=dates,
            open=opens,
            high=highs,
            low=lows,
            close=closes,
            name=ticker,
            increasing_line_color='#22c55e',
            increasing_fillcolor='rgba(34,197,94,0.6)',
            decreasing_line_color='#ef4444',
            decreasing_fillcolor='rgba(239,68,68,0.6)',
        ))

        # --- 20-day MA overlay ---
        fig.add_trace(go.Scatter(
            x=dates,
            y=ma20,
            mode='lines',
            name='20-Day MA',
            line=dict(color='#f59e0b', width=1.5, dash='dot'),
            opacity=0.85,
        ))

        # --- Close line with gradient fill ---
        fig.add_trace(go.Scatter(
            x=dates,
            y=closes,
            mode='lines',
            name='Close',
            line=dict(color='rgba(99,179,237,0.6)', width=1),
            fill='tozeroy',
            fillcolor='rgba(99,179,237,0.04)',
            showlegend=False,
        ))

        fig.update_layout(
            title=dict(text=f'{ticker} — 6 Month Chart', font=dict(size=14, color='#e5e7eb')),
            template="plotly_dark",
            paper_bgcolor='rgba(17,24,39,1)',
            plot_bgcolor='rgba(17,24,39,1)',
            xaxis=dict(
                title="Date",
                type='date',
                autorange=True,
                gridcolor='rgba(55,65,81,0.4)',
                showgrid=True,
                rangeslider=dict(visible=False),
            ),
            yaxis=dict(
                title="Price (USD)",
                autorange=True,
                gridcolor='rgba(55,65,81,0.4)',
                showgrid=True,
                domain=[0.2, 1],
            ),
            yaxis2=dict(
                title="Volume",
                overlaying='y',
                side='right',
                showgrid=False,
                domain=[0, 0.18],
            ),
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
            margin=dict(l=50, r=60, t=60, b=50),
        )

        return json.loads(fig.to_json())

    except Exception as e:
        print(f"DEBUG: Error generating chart: {e}")
        import traceback
        traceback.print_exc()
        return None
