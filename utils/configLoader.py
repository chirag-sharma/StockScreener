from configparser import ConfigParser
from utils.tickerLoader import load_tickers


def load_symbols_from_config():
    """
    Loads stock symbols for a given sector from the configuration file.

    Returns:
        list: List of ticker symbols.

    Raises:
        ValueError: If the sector is not specified in the config.
    """
    config = ConfigParser()
    config.read('../config/screener_config.properties')
    sector = config.get('DEFAULT', 'sector', fallback=None)

    if not sector:
        raise ValueError("Sector not specified in config.")

    symbols = load_tickers(sector)
    return symbols
