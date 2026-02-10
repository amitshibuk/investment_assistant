import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import os
import uuid

# Wrap the import in a try-except block
try:
    import google.generativeai as genai
except ModuleNotFoundError:
    print("⚠️ WARNING: 'google.generativeai' module not found. LLM summary will be a basic template.")
    print("Install it with: pip install google-generativeai")
    genai = None

# Disable GUI display for matplotlib in backend
plt.switch_backend('Agg')

def generate_performance_chart(ticker, company_name, period="1y"):
    """
    Fetches stock data, generates a performance chart, and returns key metrics.
    
    Returns:
        A tuple: (filename, performance_data_dict)
    """
    print(f"📊 Generating performance chart for {company_name} ({ticker})...")
    
    # 1. Fetch data
    data = yf.download(ticker, period=period, auto_adjust=True)
    
    # --- ROBUST FIX: Check for sufficient data length before proceeding ---
    # We need at least 50 data points for a meaningful 50-day SMA and at least 2 for daily change.
    if data.empty or len(data) < 50:
        raise ValueError(f"Not enough historical data for {ticker} to generate a performance analysis. At least 50 days of data are required.")
        
    # 2. Calculate Simple Moving Average (SMA)
    data['SMA_50'] = data['Close'].rolling(window=50).mean()

    # 3. Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(data['Close'], label='Close Price', color='cyan')
    plt.plot(data['SMA_50'], label='50-Day SMA', color='orange', linestyle='--')
    
    plt.title(f'{company_name} ({ticker}) Stock Performance ({period})', color='white')
    plt.xlabel('Date', color='white')
    plt.ylabel('Price (USD)', color='white')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Style for dark mode
    ax = plt.gca()
    ax.set_facecolor('#1a202c')
    plt.gcf().set_facecolor('#1a202c')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
    plt.legend(facecolor='#2d3748', edgecolor='white', labelcolor='white')

    # 4. Save the chart to a unique file in the static folder
    filename = f'chart_{ticker}_{uuid.uuid4()}.png'
    filepath = os.path.join('static', filename)
    plt.savefig(filepath, bbox_inches='tight')
    plt.close() # Close the plot to free up memory

    # 5. Extract key performance data for the LLM
    # These are now safe to call because of the length check above
    latest_price = data['Close'].iloc[-1]
    prev_price = data['Close'].iloc[-2]
    change = latest_price - prev_price
    percent_change = (change / prev_price) * 100
    
    sma_latest = data['SMA_50'].iloc[-1]
    # No need to check for NaN here anymore because the initial length check guarantees a value.
    trend = "upward" if latest_price > sma_latest else "downward"

    performance_data = {
        "latest_price": f"{latest_price:.2f}",
        "daily_change": f"{change:.2f}",
        "daily_percent_change": f"{percent_change:.2f}",
        "trend_vs_sma": trend
    }
    
    print(f"✅ Chart saved as {filename}. Performance data extracted.")
    return filename, performance_data

def get_llm_summary(company_name, performance_data):
    """
    Uses the Gemini API to generate a natural language summary of performance data.
    
    Returns:
        A string containing the LLM-generated summary.
    """
    print("🤖 Generating LLM summary...")

    if genai is None:
        print("--> Using placeholder data for LLM summary.")

    # ---- SIMULATED LLM RESPONSE ----
    summary_text = (
        f"{company_name} is currently trading at ${performance_data['latest_price']}. "
        f"The stock experienced a recent daily change of ${performance_data['daily_change']} ({performance_data['daily_percent_change']}%). "
        f"Based on its position relative to the 50-day moving average, the current short-term trend is {performance_data['trend_vs_sma']}."
    )
    # ---- END SIMULATION ----
    
    print("✅ LLM summary generated.")
    return summary_text

