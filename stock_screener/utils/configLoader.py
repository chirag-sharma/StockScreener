"""
This module provides utilities to load stock symbols from configuration files.
"""

from configparser import ConfigParser
from stock_screener.utils.tickerLoader import load_tickers


def load_symbols_from_config():
    """
    Loads stock symbols for a given sector from the configuration file.

    Returns:
        list: List of ticker symbols.

    Raises:
        ValueError: If the sector is not specified in the config.
    """
    config = ConfigParser()
    import os
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'screener_config.properties')
    config.read(config_path)
    sector = config.get('DEFAULT', 'sector', fallback=None)

    if not sector:
        raise ValueError("Sector not specified in config.")

    # Load tickers for the specified sector
    symbols = load_tickers(sector)
    return symbols
