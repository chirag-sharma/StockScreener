# src/constants.py

"""
This module contains all static constants used across the stock screener project.
Includes mapping of scope identifiers to Screener.in URLs for indices and sectors.
"""
import configparser
import os

# Scope to Screener URL map for index-based and sector-based stock groupings
SCOPE_URL_MAP = {
    # Index-based scopes
    "nifty_50": "https://www.screener.in/screens/507420/nifty-50-companies/",
    "nifty_100": "https://www.screener.in/screens/1263240/nifty-100/",
    "nifty_200": "https://www.screener.in/screens/1263241/nifty-200/",
    "nifty_500": "https://www.screener.in/screens/1263242/nifty-500/",
    "midcap_100": "https://www.screener.in/screens/1263243/nifty-midcap-100/",
    "smallcap_100": "https://www.screener.in/screens/1263244/nifty-smallcap-100/",

    # Sector-based scopes
    "auto_sector": "https://www.screener.in/screens/1263245/auto-sector/",
    "banking_sector": "https://www.screener.in/screens/1263246/banking-sector/",
    "it_sector": "https://www.screener.in/screens/1263247/it-sector/",
    "fmcg_sector": "https://www.screener.in/screens/1263248/fmcg-sector/",
    "pharma_sector": "https://www.screener.in/screens/1263249/pharma-sector/",
    "metal_sector": "https://www.screener.in/screens/1263250/metal-sector/",
    "oil_gas_sector": "https://www.screener.in/screens/1263251/oil-and-gas-sector/",
    "realty_sector": "https://www.screener.in/screens/1263252/realty-sector/",
    "energy_sector": "https://www.screener.in/screens/1263253/energy-sector/",
    "infra_sector": "https://www.screener.in/screens/1263254/infrastructure-sector/",
    "chemicals_sector": "https://www.screener.in/screens/1263255/chemicals-sector/",
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # directory of constants.py
BASE_CACHE_DIR = os.path.join("..", "data", "input", "tickers")
BASE_CACHE_DIR = os.path.abspath(BASE_CACHE_DIR)

config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'screener_config.properties')
config = configparser.ConfigParser()
config.read(config_path)

# Thresholds from config
ROE_MIN = float(config['DEFAULT'].get('roe_min', 15))
PE_RATIO_MAX = float(config['DEFAULT'].get('pe_ratio_max', 20))
DE_RATIO_MAX = float(config['DEFAULT'].get('de_ratio_max', 1))
CURRENT_RATIO_MIN = float(config['DEFAULT'].get('current_ratio_min', 1.5))
PRICE_TO_BOOK_MAX = float(config['DEFAULT'].get('price_to_book_max', 3))
PROMOTER_HOLDING_MIN = float(config['DEFAULT'].get('promoter_holding_min', 50))

THRESHOLDS = {
    'roe_min': ROE_MIN,
    'pe_ratio_max': PE_RATIO_MAX,
    'debt_to_equity_max': DE_RATIO_MAX,
    'current_ratio_min': CURRENT_RATIO_MIN,
    'price_to_book_max': PRICE_TO_BOOK_MAX,
    'promoter_holding_min': PROMOTER_HOLDING_MIN
}


