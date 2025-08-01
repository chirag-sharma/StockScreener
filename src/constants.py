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
EPS_GROWTH_MIN = float(config['DEFAULT'].get('eps_growth_min', 10))
ROA_MIN = float(config['DEFAULT'].get('roa_min', 5))
DIVIDEND_YIELD_MIN = float(config['DEFAULT'].get('dividend_yield_min', 1))
FREE_CASH_FLOW_MIN = float(config['DEFAULT'].get('free_cash_flow_min', 0))
PRICE_TO_CASH_FLOW_MAX = float(config['DEFAULT'].get('price_to_cash_flow_max', 15))
ENTERPRISE_VALUE_MIN = float(config['DEFAULT'].get('enterprise_value_min', 0))
EV_EBITDA_MAX = float(config['DEFAULT'].get('ev_ebitda_max', 15))
INTEREST_COVERAGE_MIN = float(config['DEFAULT'].get('interest_coverage_min', 3))
QUICK_RATIO_MIN = float(config['DEFAULT'].get('quick_ratio_min', 1))
NET_PROFIT_MARGIN_MIN = float(config['DEFAULT'].get('net_profit_margin_min', 10))
OPERATING_MARGIN_MIN = float(config['DEFAULT'].get('operating_margin_min', 10))
RSI_MIN = float(config['DEFAULT'].get('rsi_min', 30))
RSI_MAX = float(config['DEFAULT'].get('rsi_max', 70))
CASH_CONVERSION_RATIO_MIN = float(config['DEFAULT'].get('cash_conversion_ratio_min', 1))
DIVIDEND_PAYOUT_RATIO_MAX = float(config['DEFAULT'].get('dividend_payout_ratio_max', 60))
MARKET_CAP_MIN = float(config['DEFAULT'].get('market_cap_min', 5000))
MARKET_CAP_MAX = float(config['DEFAULT'].get('market_cap_max', 1000000))

THRESHOLDS = {
    'roe_min': ROE_MIN,
    'pe_ratio_max': PE_RATIO_MAX,
    'debt_to_equity_max': DE_RATIO_MAX,
    'current_ratio_min': CURRENT_RATIO_MIN,
    'price_to_book_max': PRICE_TO_BOOK_MAX,
    'promoter_holding_min': PROMOTER_HOLDING_MIN,
    'eps_growth_min': EPS_GROWTH_MIN,
    'roa_min': ROA_MIN,
    'dividend_yield_min': DIVIDEND_YIELD_MIN,
    'free_cash_flow_min': FREE_CASH_FLOW_MIN,
    'price_to_cash_flow_max': PRICE_TO_CASH_FLOW_MAX,
    'enterprise_value_min': ENTERPRISE_VALUE_MIN,
    'ev_ebitda_max': EV_EBITDA_MAX,
    'interest_coverage_min': INTEREST_COVERAGE_MIN,
    'quick_ratio_min': QUICK_RATIO_MIN,
    'net_profit_margin_min': NET_PROFIT_MARGIN_MIN,
    'operating_margin_min': OPERATING_MARGIN_MIN,
    'rsi_range': (RSI_MIN, RSI_MAX),
    'cash_conversion_ratio_min': CASH_CONVERSION_RATIO_MIN,
    'dividend_payout_ratio_max': DIVIDEND_PAYOUT_RATIO_MAX,
    'market_cap_range': (MARKET_CAP_MIN, MARKET_CAP_MAX)
}



