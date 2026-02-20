import json
import os
import pytz
from datetime import datetime, time as dt_time
import google.generativeai as genai
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.contrib.auth.decorators import login_required
from dotenv import load_dotenv
import yfinance as yf

# --- Imports from your src folder ---
from .src.prediction_model import load_and_predict, train_and_save_model
from .src.visualizer import generate_interactive_chart, generate_candlestick_chart

# --- Configuration ---
load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    print("Warning: GEMINI_API_KEY is not set.")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash-lite')

TICKER_MAP = {
    "Apple": "AAPL", "Tesla": "TSLA", "Google": "GOOGL",
    "Amazon": "AMZN", "Microsoft": "MSFT", "Meta": "META", "Netflix": "NFLX"
}

# Top 15 stocks by market cap (approximate)
TOP_15_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META",
    "TSLA", "BRK-B", "AVGO", "JPM", "LLY", "V",
    "UNH", "XOM", "MA"
]

# ==========================================
#  HELPER: Entity Extraction
# ==========================================
def get_entities_with_llm(query):
    """
    Uses Gemini to extract structured task, company, and time entities from the query.
    """
    system_prompt = """
    You are an expert at understanding financial queries. Extract these entities:
    - task: The user's goal. If the user mentions a company name but no specific verb, assume 'predict'.
    - metric: The financial metric (e.g., 'stock price').
    - company: The company name.
    - time_period: A dictionary with 'value' (int) and 'unit' (string).
    
    Return ONLY a JSON object. If an entity is missing, omit it.
    """
    try:
        response = model.generate_content(
            f"{system_prompt}\n\nQuery: {query}",
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json"
            )
        )
        return json.loads(response.text)
    except Exception as e:
        return {"error": str(e)}

# ==========================================
#  HELPER: Summarization
# ==========================================
def summarize_predictions_with_llm(company, ticker, days_ahead, predictions_data):
    """
    Generates a natural language summary of the AI's forecast.
    """
    start_price = predictions_data[0]['predicted_close']
    end_price = predictions_data[-1]['predicted_close']
    change = end_price - start_price
    percent_change = (change / start_price) * 100 if start_price != 0 else 0
    
    trend = "an upward" if change > 0 else "a downward"
    if abs(percent_change) < 1: trend = "relatively stable"

    prediction_details = (
        f"Company: {company} ({ticker})\n"
        f"Period: {days_ahead} days\n"
        f"Trend: {trend} ({percent_change:.2f}%)\n"
        f"Start: ${start_price:.2f}, End: ${end_price:.2f}\n"
    )

    prompt = f"""
    You are a financial analyst. Summarize these stock predictions.
    Data: {prediction_details}
    
    Requirements:
    1. Length: Moderate (3 to 5 sentences).
    2. Formatting: Use **bold** for key prices and percentages.
    3. Content: Explain the trend clearly.
    4. MANDATORY: End with "Disclaimer: This is an AI-generated prediction and not financial advice."
    """
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return "Could not generate summary."


# ==========================================
#  MAIN VIEWS
# ==========================================

@login_required
def landing(request):
    """Renders the dashboard landing page."""
    return render(request, 'core/landing.html')

@login_required
def home(request):
    """Renders the main chat interface."""
    return render(request, 'core/index.html')

@csrf_exempt 
def predict(request):
    """
    Handles Chat, Prediction, and Visualization requests.
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)

        query = data.get('query')
        if not query:
            return JsonResponse({"error": "Query required"}, status=400)
        
        # 1. Retrieve History (Session Memory)
        chat_history = request.session.get('gemini_chat_history', [])

        # 2. Extract Entities
        entities = get_entities_with_llm(query)
        task = entities.get('task')
        company = entities.get('company')
        time_period = entities.get('time_period')

        # 3. Determine Intent
        is_prediction_task = (task == 'predict') or (company and not task) or (company and task == 'unknown')

        # --- BRANCH A: Prediction & Visualization ---
        if is_prediction_task and company:
            ticker = TICKER_MAP.get(company.capitalize())
            
            if ticker:
                # Resolve Timeframe (Session Memory)
                last_days_ahead = request.session.get('last_days_ahead', 7)
                days_ahead = last_days_ahead

                if time_period:
                    unit = time_period.get('unit', 'day')
                    val = time_period.get('value', 1)
                    if 'week' in unit: days_ahead = val * 7
                    elif 'month' in unit: days_ahead = val * 30
                    else: days_ahead = val

                request.session['last_days_ahead'] = days_ahead

                # A. Run Prediction Model (Actual LSTM)
                try:
                    predictions_list = load_and_predict(ticker, days_ahead)
                except Exception as e:
                     return JsonResponse({"error": f"Prediction Error: {str(e)}"}, status=500)

                # B. Generate Interactive Chart
                chart_data = generate_interactive_chart(ticker, predictions_list)

                formatted_predictions = [
                    {"day": i+1, "predicted_close": round(p, 2)}
                    for i, p in enumerate(predictions_list)
                ]

                # C. Generate Summary
                summary = summarize_predictions_with_llm(company, ticker, days_ahead, formatted_predictions)

                # Update History
                chat_history.append({"role": "user", "parts": [query]})
                chat_history.append({"role": "model", "parts": [summary]})
                request.session['gemini_chat_history'] = chat_history

                return JsonResponse({
                    "company": company,
                    "ticker": ticker,
                    "days_ahead": days_ahead,
                    "predictions": formatted_predictions,
                    "summary": summary,
                    "chart_data": chart_data,
                    "is_follow_up": True 
                })

        # --- BRANCH B: General Chat Fallback ---
        try:
            chat = model.start_chat(history=chat_history)
            final_query = f"{query} (Please keep the answer moderate in length, approx 3-5 sentences)"
            
            response = chat.send_message(final_query)
            
            # Manually serialize history for session
            updated_history = []
            for m in chat.history:
                updated_history.append({
                    'role': m.role, 
                    'parts': [part.text for part in m.parts]
                })
            request.session['gemini_chat_history'] = updated_history
            
            return JsonResponse({
                "summary": response.text,
                "is_follow_up": True
            })


        except Exception as e:
            return JsonResponse({"error": f"Chat Error: {str(e)}"}, status=500)

    return JsonResponse({"error": "POST method required"}, status=405)

# ==========================================
#  PORTFOLIO VIEWS
# ==========================================

from .models import Portfolio

@login_required
def dashboard(request):
    """
    Renders the unified Dashboard (Chat + Portfolio).
    """
    user_portfolio = Portfolio.objects.filter(user=request.user).order_by('-added_at')
    portfolio_tickers = list(user_portfolio.values_list('ticker', flat=True))
    return render(request, 'core/dashboard.html', {
        'portfolio': user_portfolio,
        'portfolio_tickers_json': json.dumps(portfolio_tickers),
    })

@login_required
def portfolio(request):
    """
    Legacy Portfolio View - kept for reference or direct access if needed, 
    but Dashboard is now the primary interface.
    """
    user_portfolio = Portfolio.objects.filter(user=request.user).order_by('-added_at')
    return render(request, 'core/portfolio.html', {'portfolio': user_portfolio})

@csrf_exempt
@login_required
def add_to_portfolio(request):
    """Adds a ticker to the user's portfolio."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            ticker = data.get('ticker')
            if not ticker:
                return JsonResponse({"error": "Ticker required"}, status=400)
            
            # Check if already exists
            portfolio_item, created = Portfolio.objects.get_or_create(
                user=request.user, 
                ticker=ticker.upper()
            )
            
            if created:
                return JsonResponse({"message": f"{ticker} added to portfolio", "status": "added"})
            else:
                return JsonResponse({"message": f"{ticker} already in portfolio", "status": "exists"})
                
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
            
    return JsonResponse({"error": "POST method required"}, status=405)

@csrf_exempt
@login_required
def remove_from_portfolio(request):
    """Removes a ticker from the user's portfolio."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            ticker = data.get('ticker')
            if not ticker:
                return JsonResponse({"error": "Ticker required"}, status=400)
            
            deleted_count, _ = Portfolio.objects.filter(
                user=request.user,
                ticker=ticker.upper()
            ).delete()
            
            if deleted_count > 0:
                return JsonResponse({"message": f"{ticker} removed from portfolio", "status": "removed"})
            else:
                return JsonResponse({"message": f"{ticker} not in portfolio", "status": "not_found"})
                
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "POST method required"}, status=405)

@login_required
def get_portfolio_data(request, ticker):
    """
    API to fetch chart data for a specific ticker.
    Uses Candlestick chart and skips AI prediction.
    """
    try:
        chart_data = generate_candlestick_chart(ticker)
        
        if not chart_data:
             return JsonResponse({"error": "No data found"}, status=404)

        return JsonResponse({
            "ticker": ticker,
            "chart_data": chart_data
        })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

@login_required
def get_portfolio_detail(request, ticker):
    """
    Returns enriched stock details: open, prev_close, volume, 
    current price, change%, and 30-day sparkline data.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
        prev_close = info.get('previousClose') or info.get('regularMarketPreviousClose', 0)
        open_price = info.get('open') or info.get('regularMarketOpen', 0)
        volume = info.get('volume') or info.get('regularMarketVolume', 0)
        market_cap = info.get('marketCap', 0)
        fifty_two_week_high = info.get('fiftyTwoWeekHigh', 0)
        fifty_two_week_low = info.get('fiftyTwoWeekLow', 0)

        change = current_price - prev_close if current_price and prev_close else 0
        change_pct = (change / prev_close * 100) if prev_close else 0

        # 30-day sparkline
        hist = stock.history(period="30d")
        sparkline = []
        if not hist.empty:
            sparkline = hist['Close'].tolist()

        def fmt_num(n):
            if n >= 1_000_000_000_000:
                return f"${n/1_000_000_000_000:.2f}T"
            elif n >= 1_000_000_000:
                return f"${n/1_000_000_000:.2f}B"
            elif n >= 1_000_000:
                return f"${n/1_000_000:.2f}M"
            elif n >= 1_000:
                return f"{n/1_000:.1f}K"
            return str(n)

        return JsonResponse({
            "ticker": ticker,
            "current_price": round(current_price, 2) if current_price else None,
            "prev_close": round(prev_close, 2) if prev_close else None,
            "open_price": round(open_price, 2) if open_price else None,
            "volume": fmt_num(volume) if volume else "N/A",
            "market_cap": fmt_num(market_cap) if market_cap else "N/A",
            "fifty_two_week_high": round(fifty_two_week_high, 2) if fifty_two_week_high else None,
            "fifty_two_week_low": round(fifty_two_week_low, 2) if fifty_two_week_low else None,
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
            "sparkline": [round(p, 2) for p in sparkline],
        })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

# ==========================================
#  MARKET DATA APIs
# ==========================================

def get_market_status(request):
    """Returns current NYSE/NASDAQ market open/close status."""
    try:
        et_tz = pytz.timezone('America/New_York')
        now_et = datetime.now(et_tz)
        weekday = now_et.weekday()  # 0=Mon, 6=Sun
        current_time = now_et.time()

        market_open = dt_time(9, 30)
        market_close = dt_time(16, 0)

        is_open = (
            weekday < 5 and
            market_open <= current_time <= market_close
        )

        return JsonResponse({
            "is_open": is_open,
            "status": "Open" if is_open else "Closed",
            "current_time_et": now_et.strftime("%I:%M %p ET"),
        })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


def get_stock_news(request, ticker):
    """Returns top 5 recent news items for a ticker from yfinance."""
    try:
        stock = yf.Ticker(ticker)
        raw_news = stock.news or []
        news_items = []
        for item in raw_news[:6]:
            content = item.get('content', {})
            title = content.get('title', item.get('title', ''))
            summary = content.get('summary', '')
            pub_date = content.get('pubDate', '')
            provider = content.get('provider', {})
            if isinstance(provider, dict):
                source = provider.get('displayName', '')
            else:
                source = ''
            # Get canonical URL
            canonical = content.get('canonicalUrl', {})
            url = canonical.get('url', '') if isinstance(canonical, dict) else ''

            if title:
                news_items.append({
                    "title": title,
                    "summary": summary[:200] if summary else '',
                    "url": url,
                    "source": source,
                    "published": pub_date[:10] if pub_date else '',
                })
        return JsonResponse({"ticker": ticker, "news": news_items})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


def get_top_stocks(request):
    """Returns top 15 stocks by market cap with price and daily change."""
    try:
        results = []
        for ticker_sym in TOP_15_TICKERS:
            try:
                stock = yf.Ticker(ticker_sym)
                info = stock.info
                current = info.get('currentPrice') or info.get('regularMarketPrice', 0)
                prev = info.get('previousClose') or info.get('regularMarketPreviousClose', 0)
                change_pct = ((current - prev) / prev * 100) if prev else 0
                market_cap = info.get('marketCap', 0)
                name = info.get('shortName', ticker_sym)

                results.append({
                    "ticker": ticker_sym,
                    "name": name,
                    "price": round(current, 2) if current else 0,
                    "change_pct": round(change_pct, 2),
                    "market_cap": market_cap,
                })
            except Exception:
                pass

        results.sort(key=lambda x: x['market_cap'], reverse=True)
        return JsonResponse({"stocks": results})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@login_required
def get_live_tickers(request):
    """Returns live price + change% for the user's portfolio tickers."""
    try:
        user_portfolio = Portfolio.objects.filter(user=request.user)
        tickers = [item.ticker for item in user_portfolio]

        results = []
        for ticker_sym in tickers:
            try:
                stock = yf.Ticker(ticker_sym)
                info = stock.info
                current = info.get('currentPrice') or info.get('regularMarketPrice', 0)
                prev = info.get('previousClose') or info.get('regularMarketPreviousClose', 0)
                change_pct = ((current - prev) / prev * 100) if prev else 0

                results.append({
                    "ticker": ticker_sym,
                    "price": round(current, 2) if current else 0,
                    "change_pct": round(change_pct, 2),
                    "is_up": change_pct >= 0,
                })
            except Exception:
                pass

        return JsonResponse({"tickers": results})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@login_required
def auto_train(request, ticker):
    """Triggers model training for a specific ticker in the background."""
    import threading

    def train_in_background(t):
        try:
            train_and_save_model(t)
            print(f"Auto-train complete for {t}")
        except Exception as e:
            print(f"Auto-train failed for {t}: {e}")

    ticker = ticker.upper()
    thread = threading.Thread(target=train_in_background, args=(ticker,), daemon=True)
    thread.start()

    return JsonResponse({
        "status": "training_started",
        "message": f"Training started for {ticker} in background."
    })
