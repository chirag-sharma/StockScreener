#!/usr/bin/env python3
"""
Constants and Configuration Module
==================================

This module defines static constants and configuration values used throughout 
the stock screener project. It provides centralized access to:
- URL mappings for index and sector-based stock groupings
- Directory paths for data and output
- Financial threshold values from configuration
- Environment and system configuration

All paths and configurations are loaded from external files to maintain
flexibility and separation of concerns.
"""

import configparser
import os
from pathlib import Path
from stock_screener.utils.logging_config import get_logger

# Initialize module logger
logger = get_logger(__name__)

# Mapping of scope identifiers to Screener.in URLs for index-based and sector-based stock groupings
SCOPE_URL_MAP = {
    # Index-based scopes - Major market indices
    "nifty_50": "https://www.screener.in/screens/507420/nifty-50-companies/",
    "nifty_100": "https://www.screener.in/screens/1263240/nifty-100/",
    "nifty_200": "https://www.screener.in/screens/1263241/nifty-200/",
    "nifty_500": "https://www.screener.in/screens/1263242/nifty-500/",
    "midcap_100": "https://www.screener.in/screens/1263243/nifty-midcap-100/",
    "smallcap_100": "https://www.screener.in/screens/1263244/nifty-smallcap-100/",

    # Sector-based scopes - Industry classifications
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

def _get_project_paths():
    """
    Calculate and return project directory paths.
    
    Returns:
        tuple: (base_dir, project_root, base_cache_dir, base_input_dir, base_output_dir)
    """
    # Base directory of the current file
    base_dir = Path(__file__).parent.absolute()
    
    # Project root directory (two levels up from core)
    project_root = base_dir.parent.parent
    
    # Data directories - point to root-level data directory
    base_cache_dir = project_root / "data" / "input" / "tickers"
    base_input_dir = project_root / "data" / "input"  
    base_output_dir = project_root / "data" / "output"
    
    # Create directories if they don't exist
    for directory in [base_cache_dir, base_input_dir, base_output_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    logger.debug(f"Project paths initialized - Root: {project_root}")
    
    return base_dir, project_root, base_cache_dir, base_input_dir, base_output_dir

# Initialize paths
BASE_DIR, PROJECT_ROOT, BASE_CACHE_DIR, BASE_INPUT_DIR, BASE_OUTPUT_DIR = _get_project_paths()

# Convert to strings for backward compatibility
BASE_DIR = str(BASE_DIR)
PROJECT_ROOT = str(PROJECT_ROOT)
BASE_CACHE_DIR = str(BASE_CACHE_DIR)
BASE_INPUT_DIR = str(BASE_INPUT_DIR)
BASE_OUTPUT_DIR = str(BASE_OUTPUT_DIR)

def _load_thresholds():
    """
    Load financial thresholds from configuration file.
    
    Returns:
        dict: Dictionary of financial thresholds with their values
    """
    config_path = Path(__file__).parent.parent.parent / "config" / "screener_config.properties"
    config = configparser.ConfigParser()
    
    try:
        config.read(config_path)
        logger.info(f"Configuration loaded from: {config_path}")
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        logger.warning("Using default threshold values")
    
    # Load thresholds with defaults
    thresholds = {
        'ROE_MIN': float(config['DEFAULT'].get('roe_min', 15)),  # Minimum Return on Equity (%)
        'PE_RATIO_MAX': float(config['DEFAULT'].get('pe_ratio_max', 20)),  # Maximum Price/Earnings Ratio
        'DE_RATIO_MAX': float(config['DEFAULT'].get('de_ratio_max', 1)),  # Maximum Debt/Equity Ratio
        'CURRENT_RATIO_MIN': float(config['DEFAULT'].get('current_ratio_min', 1.5)),  # Minimum Current Ratio
        'PRICE_TO_BOOK_MAX': float(config['DEFAULT'].get('price_to_book_max', 3)),  # Maximum Price/Book Ratio
        'PROMOTER_HOLDING_MIN': float(config['DEFAULT'].get('promoter_holding_min', 50)),  # Minimum Promoter Holding (%)
        'PRICE_TO_CASH_FLOW_MAX': float(config['DEFAULT'].get('price_to_cash_flow_max', 15)),  # Maximum Price/Cash Flow Ratio
        'QUICK_RATIO_MIN': float(config['DEFAULT'].get('quick_ratio_min', 1.0)),  # Minimum Quick Ratio
        'INTEREST_COVERAGE_MIN': float(config['DEFAULT'].get('interest_coverage_min', 2.0)),  # Minimum Interest Coverage Ratio
        'FREE_CASH_FLOW_MIN': float(config['DEFAULT'].get('free_cash_flow_min', 0)),  # Minimum Free Cash Flow
        'EPS_GROWTH_MIN': float(config['DEFAULT'].get('eps_growth_min', 10)),  # Minimum EPS Growth (%)
        'ROA_MIN': float(config['DEFAULT'].get('roa_min', 10)),  # Minimum Return on Assets (%)
        'NET_PROFIT_MARGIN_MIN': float(config['DEFAULT'].get('net_profit_margin_min', 10)),  # Minimum Net Profit Margin (%)
        'OPERATING_MARGIN_MIN': float(config['DEFAULT'].get('operating_margin_min', 15)),  # Minimum Operating Margin (%)
        'CASH_CONVERSION_MIN': float(config['DEFAULT'].get('cash_conversion_min', 1.0)),  # Minimum Cash Conversion Cycle
        'EV_EBITDA_MAX': float(config['DEFAULT'].get('ev_ebitda_max', 10)),  # Maximum EV/EBITDA Ratio
        'MARKET_CAP_MIN': float(config['DEFAULT'].get('market_cap_min', 1000)),  # Minimum Market Capitalization (in crores)
        'PLEDGED_SHARES_MAX': float(config['DEFAULT'].get('pledged_shares_max', 10)),  # Maximum Pledged Shares (%)
        'REVENUE_GROWTH_MIN': float(config['DEFAULT'].get('revenue_growth_min', 10)),  # Minimum Revenue Growth (%)
        'DIVIDEND_YIELD_MIN': float(config['DEFAULT'].get('dividend_yield_min', 2.0)),  # Minimum Dividend Yield (%)
        'DIVIDEND_PAYOUT_MIN': float(config['DEFAULT'].get('dividend_payout_min', 40)),  # Minimum Dividend Payout Ratio (%)
    }
    
    logger.debug(f"Loaded {len(thresholds)} financial thresholds from configuration")
    return thresholds

# Load thresholds and create individual constants for backward compatibility
_THRESHOLDS = _load_thresholds()

# Create individual constants (for backward compatibility)
ROE_MIN = _THRESHOLDS['ROE_MIN']
PE_RATIO_MAX = _THRESHOLDS['PE_RATIO_MAX']
DE_RATIO_MAX = _THRESHOLDS['DE_RATIO_MAX']
CURRENT_RATIO_MIN = _THRESHOLDS['CURRENT_RATIO_MIN']
PRICE_TO_BOOK_MAX = _THRESHOLDS['PRICE_TO_BOOK_MAX']
PROMOTER_HOLDING_MIN = _THRESHOLDS['PROMOTER_HOLDING_MIN']
PRICE_TO_CASH_FLOW_MAX = _THRESHOLDS['PRICE_TO_CASH_FLOW_MAX']
QUICK_RATIO_MIN = _THRESHOLDS['QUICK_RATIO_MIN']
INTEREST_COVERAGE_MIN = _THRESHOLDS['INTEREST_COVERAGE_MIN']
FREE_CASH_FLOW_MIN = _THRESHOLDS['FREE_CASH_FLOW_MIN']
EPS_GROWTH_MIN = _THRESHOLDS['EPS_GROWTH_MIN']
ROA_MIN = _THRESHOLDS['ROA_MIN']
NET_PROFIT_MARGIN_MIN = _THRESHOLDS['NET_PROFIT_MARGIN_MIN']
OPERATING_MARGIN_MIN = _THRESHOLDS['OPERATING_MARGIN_MIN']
CASH_CONVERSION_MIN = _THRESHOLDS['CASH_CONVERSION_MIN']
EV_EBITDA_MAX = _THRESHOLDS['EV_EBITDA_MAX']
MARKET_CAP_MIN = _THRESHOLDS['MARKET_CAP_MIN']
PLEDGED_SHARES_MAX = _THRESHOLDS['PLEDGED_SHARES_MAX']
REVENUE_GROWTH_MIN = _THRESHOLDS['REVENUE_GROWTH_MIN']
DIVIDEND_YIELD_MIN = _THRESHOLDS['DIVIDEND_YIELD_MIN']
DIVIDEND_PAYOUT_MIN = _THRESHOLDS['DIVIDEND_PAYOUT_MIN']

# Also provide as a dictionary for backward compatibility and easy iteration
THRESHOLDS = _THRESHOLDS.copy()

# Log successful initialization
logger.info("Constants module initialized successfully")
logger.debug(f"Available scopes: {list(SCOPE_URL_MAP.keys())}")
logger.debug(f"Data directories: Input={BASE_INPUT_DIR}, Output={BASE_OUTPUT_DIR}, Cache={BASE_CACHE_DIR}")

# Legacy THRESHOLDS dictionary format (for backward compatibility)
LEGACY_THRESHOLDS = {
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
    'dividend_yield_min': DIVIDEND_YIELD_MIN,
    'dividend_payout_min': DIVIDEND_PAYOUT_MIN
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
