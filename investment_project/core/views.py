import json
import os
import google.generativeai as genai
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from .src.prediction_model import load_and_predict

# --- Configuration ---
genai.configure(api_key=settings.GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash-lite')

TICKER_MAP = {
    "Apple": "AAPL", "Tesla": "TSLA", "Google": "GOOGL",
    "Amazon": "AMZN", "Microsoft": "MSFT", "Meta": "META", "Netflix": "NFLX"
}

def get_entities_with_llm(query):
    """
    Extracts structured entities to determine if the user wants a specific prediction task.
    """
    system_prompt = """
    You are an expert at understanding financial queries. Extract these entities:
    - task: The user's goal. If the user mentions a company name but no specific verb, assume 'predict'.
    - metric: The financial metric (e.g., 'stock price').
    - company: The company name.
    - time_period: A dictionary with 'value' (int) and 'unit' (string).
    
    Return ONLY a JSON object.
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

def summarize_predictions_with_llm(company, ticker, days_ahead, predictions_data):
    """
    Generates a natural language summary of the LSTM model's output.
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
    1. Length: Moderate (3 to 5 sentences). Do not be too brief, but do not ramble.
    2. Formatting: Use bolding for key figures (prices, percentages).
    3. Content: Explain the trend and the start/end values clearly.
    4. MANDATORY: End with "Disclaimer: This is an AI-generated prediction and not financial advice."
    """
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return "Could not generate summary."

# --- Views ---
def home(request):
    return render(request, 'core/index.html')

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
        
        chat_history = request.session.get('gemini_chat_history', [])

        # 1. Extract Entities
        entities = get_entities_with_llm(query)
        
        task = entities.get('task')
        company = entities.get('company')
        time_period = entities.get('time_period')

        # 2. Check if we should trigger the LSTM Model
        is_prediction_task = (task == 'predict') or (company and not task) or (company and task == 'unknown')

        if is_prediction_task and company:
            ticker = TICKER_MAP.get(company.capitalize())
            
            if ticker:
                # --- NEW FIX: Session Memory for Timeframe ---
                # Retrieve the last used timeframe, defaulting to 7 if it's the first query
                last_days_ahead = request.session.get('last_days_ahead', 7)
                days_ahead = last_days_ahead

                # Override ONLY if the user explicitly provided a new timeframe in this query
                if time_period:
                    unit = time_period.get('unit', 'day')
                    val = time_period.get('value', 1)
                    if 'week' in unit: days_ahead = val * 7
                    elif 'month' in unit: days_ahead = val * 30
                    else: days_ahead = val

                # Save the timeframe back to the session for the next follow-up
                request.session['last_days_ahead'] = days_ahead
                # ----------------------------------------------

                try:
                    predictions_list = load_and_predict(ticker, days_ahead)
                except Exception as e:
                     return JsonResponse({"error": str(e)}, status=500)

                formatted_predictions = [
                    {"day": i+1, "predicted_close": round(p, 2)}
                    for i, p in enumerate(predictions_list)
                ]

                summary = summarize_predictions_with_llm(company, ticker, days_ahead, formatted_predictions)

                # Update conversation history with structured data context
                chat_history.append({"role": "user", "parts": [query]})
                chat_history.append({"role": "model", "parts": [summary]})
                request.session['gemini_chat_history'] = chat_history

                return JsonResponse({
                    "company": company,
                    "ticker": ticker,
                    "days_ahead": days_ahead,
                    "predictions": formatted_predictions,
                    "summary": summary,
                    "is_follow_up": True 
                })

        # 3. Standard Chat Fallback (for non-prediction questions)
        try:
            chat = model.start_chat(history=chat_history)
            
            # Helper to enforce length on general chat too
            final_query = f"{query} (Please keep the answer moderate in length, approx 3-5 sentences)"
            
            response = chat.send_message(final_query)
            
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