# utils/ticker_loader.py

import os
import json
from datetime import datetime
from utils.screener_scraper import fetch_tickers_from_screener

CACHE_DIR = "config/tickers"

def get_today_str():
    return datetime.today().strftime("%Y-%m-%d")

def get_cache_filepath(scope):
    return os.path.join(CACHE_DIR, f"{scope}.json")

def load_tickers(scope):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = get_cache_filepath(scope)

    # Load tickers from cache if today's date matches
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            data = json.load(f)
            if data.get("date") == get_today_str():
                return data.get("tickers", [])

    # Fetch tickers if cache is missing or stale
    tickers = fetch_tickers_from_screener(scope)
    if tickers:
        with open(cache_file, "w") as f:
            json.dump({
                "date": get_today_str(),
                "tickers": tickers
            }, f, indent=2)
    return tickers