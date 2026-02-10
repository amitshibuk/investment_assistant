import re
import spacy
from spellchecker import SpellChecker
from difflib import get_close_matches

# --- Step 1: Setup ---
spell = SpellChecker()
# Load the small English model for spaCy, disabling components we don't need for speed.
nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger"])

# --- Step 2: Domain Dictionaries ---
COMPANIES = ["Apple", "Tesla", "Google", "Amazon", "Microsoft", "Meta", "Netflix"]
METRICS = ["growth", "revenue", "profit", "loss", "market share", "stock price", "valuation"]

# --- Step 3: Helper Functions ---

def preprocess(query):
    """Corrects spelling and converts the query to lowercase."""
    corrected = " ".join([spell.correction(word) or word for word in query.split()])
    return corrected.lower()

def get_intent(query):
    """Determines the user's intent based on keywords."""
    intents = {
        "predict": ["forecast", "predict", "estimate"],
        "compare": ["compare", "versus", "vs"],
        "visualize": ["graph", "chart", "plot", "visualize"],
        "summarize": ["summary", "summarize"]
    }
    for intent, keywords in intents.items():
        if any(keyword in query for keyword in keywords):
            return intent
    return "unknown"

def extract_entities(query):
    """
    Extracts company, time period, metric, and task from the query using a more
    robust, direct-matching approach.
    """
    entities = {"company": None, "time_period": None, "task": None, "metric": None}
    
    # 1. Extract Task
    entities["task"] = get_intent(query)

    # 2. Extract Company
    # This direct check is more reliable than fuzzy matching for this specific list.
    for company in COMPANIES:
        if company.lower() in query:
            entities["company"] = company
            break

    # 3. Extract Time Period
    # This regex looks for patterns like "7 days", "next week", "for 1 month"
    time_match = re.search(r"(\d+)?\s*(day|week|month|year)s?", query, re.IGNORECASE)
    if time_match:
        value_str, unit = time_match.groups()
        entities["time_period"] = {
            "value": int(value_str) if value_str and value_str.isdigit() else 1,
            "unit": unit.lower()
        }

    # 4. Extract Metric
    # Check for "stock price" or just "stock"
    if "stock price" in query or "stock" in query:
        entities["metric"] = "stock price"
    else:
        for metric in METRICS:
            if metric in query:
                entities["metric"] = metric
                break
                
    return entities

def intent_classification(query):
    """
    Main function to process a query and return all extracted entities.
    """
    processed_query = preprocess(query)
    entities = extract_entities(processed_query)
    print(entities)
    return entities

