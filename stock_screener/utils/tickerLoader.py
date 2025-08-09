# tickerLoader.py
"""
This module provides functions to load and cache stock tickers for different scopes (indices/sectors).
It supports cache validation and fetching from Screener.in if cache is missing or outdated.
"""

import os
import json
from datetime import datetime, date
from stock_screener.core.constants import BASE_CACHE_DIR, SCOPE_URL_MAP
from stock_screener.utils.screenerScraper import fetch_tickers_from_screener

def get_cache_file(scope: str) -> str:
    """
    Returns the cache file path for a given scope.
    """
    return os.path.join(BASE_CACHE_DIR, f"{scope}.json")

# Add mapping for sector name variations
SECTOR_NAME_MAP = {
    # Index mappings
    'nifty_50': 'nifty_50',
    'nifty_100': 'nifty_100',
    'nifty_200': 'nifty_200',
    'nifty_500': 'nifty_500',
    'nifty_midcap_100': 'midcap_100',
    'nifty_smallcap_100': 'smallcap_100',

    # Sector mappings
    'nifty_it': 'it_sector',
    'nifty_pharma': 'pharma_sector',
    'nifty_auto': 'auto_sector',
    'nifty_bank': 'banking_sector',
    'nifty_fmcg': 'fmcg_sector',
    'nifty_metal': 'metal_sector',
    'nifty_oil_gas': 'oil_gas_sector',
    'nifty_reality': 'realty_sector',
    'nifty_energy': 'energy_sector',
    'nifty_infra': 'infra_sector',
    'nifty_chemicals': 'chemicals_sector'
}

def normalize_sector_name(scope: str) -> str:
    """
    Normalize sector names to match SCOPE_URL_MAP keys.

    Args:
        scope (str): The sector name from config

    Returns:
        str: Normalized sector name that matches SCOPE_URL_MAP
    """
    return SECTOR_NAME_MAP.get(scope, scope)

def is_cache_valid(json_data: dict) -> bool:
    """
    Checks if the cache data is valid for today and contains tickers.
    """
    try:
        cache_date = datetime.strptime(json_data.get("date", ""), "%Y-%m-%d").date()
        return cache_date == date.today() and "tickers" in json_data
    except Exception as e:
        print(f"[WARN] Invalid date format in cache: {e}")
        return False


def load_tickers(scope_or_file_path: str) -> list:
    """
    Load tickers for a given scope or from a direct file path.

    Args:
        scope_or_file_path (str): The index/sector scope identifier or direct file path.

    Returns:
        list: List of ticker symbols.
    """
    # Check if it's a direct file path
    if scope_or_file_path.endswith('.json') or '/' in scope_or_file_path:
        try:
            with open(scope_or_file_path, 'r') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, dict) and 'tickers' in data:
                return data['tickers']
            elif isinstance(data, list):
                return data
            else:
                print(f"[WARN] Unknown JSON structure in {scope_or_file_path}")
                return []
                
        except Exception as e:
            print(f"[ERROR] Could not load file {scope_or_file_path}: {e}")
            return []
    
    # Original scope-based loading logic
    scope = scope_or_file_path
    """
    Loads tickers for a given scope from cache if valid, otherwise fetches from Screener.in and updates cache.

    Args:
        scope (str): The index or sector scope identifier.

    Returns:
        list: List of ticker symbols.
    """
    # Normalize the sector name
    normalized_scope = normalize_sector_name(scope)

    # Ensure the base cache directory exists
    os.makedirs(BASE_CACHE_DIR, exist_ok=True)
    file_path = get_cache_file(scope)  # Use original scope for cache file name

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

    # Cache miss or invalid data; fetch tickers from Screener using normalized scope
    tickers = fetch_tickers_from_screener(normalized_scope)
    today = datetime.today().strftime("%Y-%m-%d")

    # Save the newly fetched tickers to cache
    with open(file_path, "w") as f:
        json.dump({"date": today, "tickers": tickers}, f, indent=2)

    print(f"[CACHE MISS] Fetched and cached tickers for scope: {scope}")
    return tickers
