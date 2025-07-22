# ticker_loader.py

import os
import json
from datetime import datetime, date
from utils.constants import BASE_CACHE_DIR, SCOPE_URL_MAP
from utils.screener_scraper import fetch_tickers_from_screener

def get_cache_file(scope: str) -> str:
    return os.path.join(BASE_CACHE_DIR, f"{scope}.json")

def is_cache_valid(json_data: dict) -> bool:
    try:
        cache_date = datetime.strptime(json_data.get("date", ""), "%Y-%m-%d").date()
        return cache_date == date.today() and "tickers" in json_data
    except Exception as e:
        print(f"[WARN] Invalid date format in cache: {e}")
        return False


def load_tickers(scope: str) -> list:
    # Ensure the base cache directory exists
    os.makedirs(BASE_CACHE_DIR, exist_ok=True)
    file_path = get_cache_file(scope)

    # Attempt to read from cache if the file exists
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            # Return cached tickers if the data is valid and fresh
            if is_cache_valid(data):
                print(f"[CACHE HIT] Using cached tickers for scope: {scope}")
                return data["tickers"]
        except Exception as e:
            print(f"[WARN] Corrupted cache file for {scope}: {e}")

    # Cache miss or invalid data; fetch tickers from Screener
    tickers = fetch_tickers_from_screener(scope)
    today = datetime.today().strftime("%Y-%m-%d")

    # Save the newly fetched tickers to cache
    with open(file_path, "w") as f:
        json.dump({"date": today, "tickers": tickers}, f, indent=2)

    print(f"[CACHE MISS] Fetched and cached tickers for scope: {scope}")
    return tickers
