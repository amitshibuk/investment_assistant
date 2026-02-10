from django.shortcuts import render

# Create your views here.
# core/views.py
import json
import os
import google.generativeai as genai
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

# Import your src modules
# Note: Ensure core/src has an __init__.py file, or adjust import path
from .src.prediction_model import load_and_predict

# --- Configuration ---
genai.configure(api_key=settings.GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash-lite')

TICKER_MAP = {
    "Apple": "AAPL", "Tesla": "TSLA", "Google": "GOOGL",
    "Amazon": "AMZN", "Microsoft": "MSFT", "Meta": "META", "Netflix": "NFLX"
}

# --- Helper Functions (Same as Flask) ---
def get_entities_with_llm(query):
    system_prompt = """
    You are an expert at understanding financial queries. Extract these entities:
    - task: The user's goal (e.g., 'predict', 'compare').
    - metric: The financial metric (e.g., 'stock price').
    - company: The company name.
    - time_period: A dictionary with 'value' (int) and 'unit' (string).
    Return ONLY a JSON object.
    """
    try:
        response = model.generate_content(
            f"{system_prompt}\n\nQuery: {query}",
            generation_config=genai.GenerationConfig(response_mime_type="application/json")
        )
        return json.loads(response.text)
    except Exception as e:
        return {"error": str(e)}

def summarize_predictions_with_llm(company, ticker, days_ahead, predictions_data):
    start_price = predictions_data[0]['predicted_close']
    end_price = predictions_data[-1]['predicted_close']
    change = end_price - start_price
    percent_change = (change / start_price) * 100 if start_price != 0 else 0
    
    trend = "an upward" if change > 0 else "a downward"
    if abs(percent_change) < 1: trend = "a relatively stable"

    prediction_details = (
        f"Company: {company} ({ticker})\n"
        f"Period: {days_ahead} days\n"
        f"Trend: {trend} ({percent_change:.2f}%)\n"
        f"Start: ${start_price:.2f}, End: ${end_price:.2f}\n"
    )

    prompt = f"""
    You are a financial analyst. Summarize these stock predictions concisely:
    {prediction_details}
    Requirements:
    1. Mention the trend, start/end prices, and % change.
    2. Tone: Informative but cautious.
    3. MANDATORY: End with "Disclaimer: This is an AI-generated prediction and not financial advice."
    """
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return "Could not generate summary."

# --- Views ---

def home(request):
    return render(request, 'core/index.html')

# We use csrf_exempt for simplicity in migration, 
# but for production, you should pass the CSRF token in the JS fetch headers.
@csrf_exempt 
def predict(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)

        query = data.get('query')
        if not query:
            return JsonResponse({"error": "Query required"}, status=400)
        
        # 1. Extract Entities
        entities = get_entities_with_llm(query)
        if "error" in entities:
            return JsonResponse(entities, status=500)

        task = entities.get('task')
        metric = entities.get('metric')
        company = entities.get('company')
        time_period = entities.get('time_period')

        if task == 'predict' and 'stock' in str(metric) and company and time_period:
            ticker = TICKER_MAP.get(company.capitalize())
            if not ticker:
                return JsonResponse({"error": f"Company '{company}' not supported."}, status=400)

            # Parse days
            days_ahead = 1
            if time_period:
                unit = time_period.get('unit', 'day')
                val = time_period.get('value', 1)
                if 'week' in unit: days_ahead = val * 7
                elif 'month' in unit: days_ahead = val * 30
                else: days_ahead = val
            
            # 3. Get Prediction
            try:
                predictions_list = load_and_predict(ticker, days_ahead)
            except Exception as e:
                 return JsonResponse({"error": str(e)}, status=500)

            formatted_predictions = [
                {"day": i+1, "predicted_close": round(p, 2)}
                for i, p in enumerate(predictions_list)
            ]

            # 4. Summarize
            summary = summarize_predictions_with_llm(company, ticker, days_ahead, formatted_predictions)

            return JsonResponse({
                "company": company,
                "ticker": ticker,
                "days_ahead": days_ahead,
                "predictions": formatted_predictions,
                "summary": summary
            })

        return JsonResponse({"message": "Query not understood.", "entities": entities}, status=400)
    
    return JsonResponse({"error": "POST method required"}, status=405)