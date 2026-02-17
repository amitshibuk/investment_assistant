import json
import os
import google.generativeai as genai
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from dotenv import load_dotenv

# --- Imports from your src folder ---
# These are your core logic modules for the LSTM predictions and Plotly visualizations.
from .src.prediction_model import load_and_predict
from .src.visualizer import generate_interactive_chart

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

from django.contrib.auth.decorators import login_required

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
                    # predictions_list contains the raw future prices
                    predictions_list = load_and_predict(ticker, days_ahead)
                except Exception as e:
                     return JsonResponse({"error": f"Prediction Error: {str(e)}"}, status=500)

                # B. Generate Interactive Chart
                # We pass the predicted prices to the visualizer to split the lines correctly
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
                    "chart_data": chart_data, # Sends Plotly JSON to Frontend
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