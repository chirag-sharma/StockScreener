# src/constants.py

"""
This module defines static constants and configuration values used throughout the stock screener project.
It includes mappings for index and sector scopes, directory paths, and threshold values loaded from configuration.
"""
import configparser
import os

# Mapping of scope identifiers to Screener.in URLs for index-based and sector-based stock groupings
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

# Base directory of the current file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Project root directory (two levels up from core)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

# Data directories - point to root-level data directory
BASE_CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "input", "tickers")
BASE_INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "input")  
BASE_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "output")

# Load configuration from properties file
config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'screener_config.properties')
config = configparser.ConfigParser()
config.read(config_path)

# Thresholds for financial metrics, with defaults if not specified in config
ROE_MIN = float(config['DEFAULT'].get('roe_min', 15))  # Minimum Return on Equity (%)
PE_RATIO_MAX = float(config['DEFAULT'].get('pe_ratio_max', 20))  # Maximum Price/Earnings Ratio
DE_RATIO_MAX = float(config['DEFAULT'].get('de_ratio_max', 1))  # Maximum Debt/Equity Ratio
CURRENT_RATIO_MIN = float(config['DEFAULT'].get('current_ratio_min', 1.5))  # Minimum Current Ratio
PRICE_TO_BOOK_MAX = float(config['DEFAULT'].get('price_to_book_max', 3))  # Maximum Price/Book Ratio
PROMOTER_HOLDING_MIN = float(config['DEFAULT'].get('promoter_holding_min', 50))  # Minimum Promoter Holding (%)
PRICE_TO_CASH_FLOW_MAX = float(config['DEFAULT'].get('price_to_cash_flow_max', 15))  # Maximum Price/Cash Flow Ratio
QUICK_RATIO_MIN = float(config['DEFAULT'].get('quick_ratio_min', 1.0))  # Minimum Quick Ratio
INTEREST_COVERAGE_MIN = float(config['DEFAULT'].get('interest_coverage_min', 2.0))  # Minimum Interest Coverage Ratio
FREE_CASH_FLOW_MIN = float(config['DEFAULT'].get('free_cash_flow_min', 0))  # Minimum Free Cash Flow
EPS_GROWTH_MIN = float(config['DEFAULT'].get('eps_growth_min', 10))  # Minimum EPS Growth (%)
ROA_MIN = float(config['DEFAULT'].get('roa_min', 10))  # Minimum Return on Assets (%)
NET_PROFIT_MARGIN_MIN = float(config['DEFAULT'].get('net_profit_margin_min', 10))  # Minimum Net Profit Margin (%)
OPERATING_MARGIN_MIN = float(config['DEFAULT'].get('operating_holding_min', 15))  # Minimum Operating Margin (%)
CASH_CONVERSION_MIN = float(config['DEFAULT'].get('cash_conversion_min', 1.0))  # Minimum Cash Conversion Cycle
EV_EBITDA_MAX = float(config['DEFAULT'].get('ev_ebitda_max', 10))  # Maximum EV/EBITDA Ratio
MARKET_CAP_MIN = float(config['DEFAULT'].get('market_cap_min', 1000))  # Minimum Market Capitalization (in crores)
PLEDGED_SHARES_MAX = float(config['DEFAULT'].get('pledge_min', 10))  # Maximum Pledged Shares (%)
REVENUE_GROWTH_MIN = float(config['DEFAULT'].get('revenue_growth_min', 10))  # Minimum Revenue Growth (%)
DIVIDEND_YIELD_MIN = float(config['DEFAULT'].get('dividend_yield_min', 2.0))  # Minimum Dividend Yield (%)
DIVIDEND_PAYOUT_MIN = float(config['DEFAULT'].get('dividend_payout_min', 40))  # Minimum Dividend Payout Ratio (%)


THRESHOLDS = {
    'roe_min': ROE_MIN,
    'pe_ratio_max': PE_RATIO_MAX,
    'debt_to_equity_max': DE_RATIO_MAX,
    'current_ratio_min': CURRENT_RATIO_MIN,
    'price_to_book_max': PRICE_TO_BOOK_MAX,
    'promoter_holding_min': PROMOTER_HOLDING_MIN,
    'price_to_cash_flow_max': PRICE_TO_CASH_FLOW_MAX,
    'quick_ratio_min': QUICK_RATIO_MIN,
    'interest_coverage_min': INTEREST_COVERAGE_MIN,
    'free_cash_flow_min': FREE_CASH_FLOW_MIN,
    'eps_growth_min': EPS_GROWTH_MIN,
    'roa_min': ROA_MIN,
    'net_profit_margin_min': NET_PROFIT_MARGIN_MIN,
    'operating_margin_min': OPERATING_MARGIN_MIN,
    'cash_conversion_min': CASH_CONVERSION_MIN,
    'ev_ebitda_max': EV_EBITDA_MAX,
    'market_cap_min': MARKET_CAP_MIN,
    'pledged_shares_max': PLEDGED_SHARES_MAX,
    'revenue_growth_min': REVENUE_GROWTH_MIN,
    'dividend_yield_min' : DIVIDEND_YIELD_MIN,
    'dividend_payout_min' : DIVIDEND_PAYOUT_MIN
}

# Nifty indices for the screener scraper
NIFTY_INDICES = [
    "nifty_50", "nifty_100", "nifty_200", "nifty_500",
    "nifty_midcap_50", "nifty_midcap_150", 
    "nifty_smallcap_50", "nifty_smallcap_250",
    "nifty_microcap_250"
]

# Sectoral indices for the screener scraper
SECTORAL_INDICES = [
    "nifty_it", "nifty_pharma", "nifty_auto", "nifty_fmcg",
    "nifty_bank", "nifty_financial_services", "nifty_energy",
    "nifty_psu_bank", "nifty_reality", "nifty_media",
    "nifty_commodities", "nifty_metal", "nifty_healthcare"
]
