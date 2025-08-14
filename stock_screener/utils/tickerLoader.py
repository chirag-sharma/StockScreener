#!/usr/bin/env python3
"""
Ticker Loading and Caching Module
=================================

This module provides functions to load and cache stock tickers for different 
scopes (indices/sectors). It supports cache validation, fetching from Screener.in 
if cache is missing or outdated, and provides robust fallback mechanisms.

Features:
- Smart caching system with date validation
- Multiple data source support (JSON files, web scraping)
- Comprehensive error handling and fallback mechanisms  
- Sector name normalization
- Performance optimization through caching
- Detailed logging for debugging

Usage:
    from stock_screener.utils.tickerLoader import load_tickers
    
    tickers = load_tickers('nifty_50')
    tickers = load_tickers('/path/to/custom.json')
"""

import os
import json
from datetime import datetime, date
from pathlib import Path
import time
from stock_screener.core.constants import BASE_CACHE_DIR, SCOPE_URL_MAP
from stock_screener.utils.screenerScraper import fetch_tickers_from_screener
from stock_screener.utils.logging_config import get_logger, log_execution_start, log_execution_end, log_progress

# Initialize module logger
logger = get_logger(__name__)

def get_cache_file(scope: str) -> str:
    """
    Get the cache file path for a given scope.
    
    Args:
        scope (str): The scope identifier (e.g., 'nifty_50', 'banking_sector')
        
    Returns:
        str: Full path to the cache file
    """
    cache_file = os.path.join(BASE_CACHE_DIR, f"{scope}.json")
    logger.debug(f"Cache file path for '{scope}': {cache_file}")
    return cache_file

# Sector name mapping for backward compatibility and flexibility
SECTOR_NAME_MAP = {
    # Index mappings - Direct match to SCOPE_URL_MAP
    'nifty_50': 'nifty_50',
    'nifty_100': 'nifty_100',
    'nifty_200': 'nifty_200',
    'nifty_500': 'nifty_500',
    'nifty_midcap_100': 'midcap_100',
    'nifty_smallcap_100': 'smallcap_100',

    # Sector mappings - Map user-friendly names to SCOPE_URL_MAP keys
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
    
    This function maps user-friendly sector names to the standardized
    keys used in SCOPE_URL_MAP for consistent URL resolution.

    Args:
        scope (str): The sector name from configuration or user input

    Returns:
        str: Normalized sector name that matches SCOPE_URL_MAP keys
    """
    normalized = SECTOR_NAME_MAP.get(scope, scope)
    if normalized != scope:
        logger.debug(f"Sector name normalized: '{scope}' -> '{normalized}'")
    return normalized


def is_cache_valid(json_data: dict) -> bool:
    """
    Check if the cache data is valid for today and contains tickers.
    
    Args:
        json_data (dict): JSON data loaded from cache file
        
    Returns:
        bool: True if cache is valid for today and has ticker data
    """
    try:
        cache_date_str = json_data.get("date", "")
        if not cache_date_str:
            logger.debug("Cache missing date field")
            return False
            
        cache_date = datetime.strptime(cache_date_str, "%Y-%m-%d").date()
        is_today = cache_date == date.today()
        has_tickers = "tickers" in json_data and json_data["tickers"]
        
        logger.debug(f"Cache validation - Date: {cache_date}, Today: {is_today}, Has tickers: {has_tickers}")
        
        return is_today and has_tickers
        
    except ValueError as e:
        logger.warning(f"Invalid date format in cache: {cache_date_str} - {e}")
        return False
    except Exception as e:
        logger.error(f"Error validating cache: {e}")
        return False


def _load_from_file_path(file_path: str) -> list:
    """
    Load tickers from a direct JSON file path.
    
    Args:
        file_path (str): Path to JSON file containing ticker data
        
    Returns:
        list: List of ticker symbols from the file
    """
    try:
        logger.info(f"Loading tickers from file: {file_path}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, dict) and 'tickers' in data:
            tickers = data['tickers']
        elif isinstance(data, list):
            tickers = data
        else:
            logger.warning(f"Unknown JSON structure in {file_path}, trying to extract data")
            tickers = []
        
        logger.info(f"Successfully loaded {len(tickers)} tickers from file")
        logger.debug(f"First few tickers: {tickers[:3]}{'...' if len(tickers) > 3 else ''}")
        
        return tickers
        
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {file_path}: {e}")
        return []
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}")
        return []


def _load_from_cache(scope: str) -> tuple:
    """
    Attempt to load tickers from cache.
    
    Args:
        scope (str): Scope identifier for cache lookup
        
    Returns:
        tuple: (tickers_list, cache_status) where cache_status is 'hit', 'miss', or 'invalid'
    """
    file_path = get_cache_file(scope)
    
    if not os.path.exists(file_path):
        logger.debug(f"Cache file does not exist: {file_path}")
        return [], 'miss'
    
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        
        if is_cache_valid(data):
            tickers = data["tickers"]
            logger.info(f"Cache hit for scope '{scope}': {len(tickers)} tickers")
            return tickers, 'hit'
        
        # Cache exists but date is old - update date if tickers exist
        if "tickers" in data and data["tickers"]:
            logger.info(f"Cache outdated but valid for scope '{scope}', updating date")
            today = datetime.today().strftime("%Y-%m-%d")
            data["date"] = today
            
            # Save the updated date back to the file
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
            
            return data["tickers"], 'hit'
        
        logger.warning(f"Cache file exists but is invalid for scope '{scope}'")
        return [], 'invalid'
        
    except json.JSONDecodeError as e:
        logger.error(f"Corrupted cache file for '{scope}': {e}")
        return [], 'invalid'
    except Exception as e:
        logger.error(f"Error reading cache for '{scope}': {e}")
        return [], 'invalid'


def _fetch_and_cache_tickers(scope: str) -> list:
    """
    Fetch tickers from web source and cache them.
    
    Args:
        scope (str): Scope identifier for ticker fetching
        
    Returns:
        list: List of fetched ticker symbols
    """
    try:
        # Normalize scope for web fetching
        normalized_scope = normalize_sector_name(scope)
        
        logger.info(f"Fetching tickers from web for scope: {scope} (normalized: {normalized_scope})")
        
        # Fetch tickers from Screener using normalized scope
        tickers = fetch_tickers_from_screener(normalized_scope)
        
        if not tickers:
            logger.warning(f"No tickers fetched for scope: {scope}")
            return []
        
        # Cache the results
        today = datetime.today().strftime("%Y-%m-%d")
        cache_data = {"date": today, "tickers": tickers}
        
        file_path = get_cache_file(scope)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, "w") as f:
            json.dump(cache_data, f, indent=2)
        
        logger.info(f"Successfully fetched and cached {len(tickers)} tickers for scope: {scope}")
        return tickers
        
    except Exception as e:
        logger.error(f"Failed to fetch and cache tickers for scope '{scope}': {e}")
        return []


def load_tickers(scope_or_file_path: str) -> list:
    """
    Load tickers for a given scope or from a direct file path.
    
    This is the main entry point for loading ticker symbols. It supports:
    - Direct JSON file paths (ending in .json or containing '/')
    - Scope-based loading with intelligent caching
    - Fallback mechanisms for robust data loading
    
    Args:
        scope_or_file_path (str): The index/sector scope identifier or direct file path.

    Returns:
        list: List of ticker symbols.
    """
    start_time = time.time()
    log_execution_start(__name__, "load_tickers", scope_or_file_path=scope_or_file_path)
    
    try:
        # Check if it's a direct file path
        if scope_or_file_path.endswith('.json') or '/' in scope_or_file_path:
            tickers = _load_from_file_path(scope_or_file_path)
            duration = time.time() - start_time
            log_execution_end(__name__, "load_tickers", duration, f"Loaded {len(tickers)} from file")
            return tickers
        
        # Scope-based loading with caching
        scope = scope_or_file_path
        logger.info(f"Loading tickers for scope: {scope}")
        
        # Ensure cache directory exists
        os.makedirs(BASE_CACHE_DIR, exist_ok=True)
        
        # Try loading from cache first
        tickers, cache_status = _load_from_cache(scope)
        
        if cache_status == 'hit':
            duration = time.time() - start_time
            log_execution_end(__name__, "load_tickers", duration, f"Cache hit: {len(tickers)} tickers")
            return tickers
        
        # Cache miss or invalid - fetch from web
        tickers = _fetch_and_cache_tickers(scope)
        
        if not tickers:
            logger.warning(f"No tickers available for scope: {scope}")
        
        duration = time.time() - start_time
        log_execution_end(__name__, "load_tickers", duration, f"Fetched {len(tickers)} tickers")
        
        return tickers
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Failed to load tickers for '{scope_or_file_path}': {e}")
        log_execution_end(__name__, "load_tickers", duration, f"Failed: {str(e)}")
        return []


# Module initialization
logger.info("Ticker loader module initialized")
logger.debug(f"Available sector mappings: {list(SECTOR_NAME_MAP.keys())}")
logger.debug(f"Cache directory: {BASE_CACHE_DIR}")
