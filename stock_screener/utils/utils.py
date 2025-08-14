"""
This module provides utility functions for loading tickers from properties files.
"""

def load_tickers_from_properties(filepath: str) -> list:
    """
    Loads ticker symbols from a properties file.

    Args:
        filepath (str): Path to the properties file.

    Returns:
        list: List of ticker symbols.
    """
    tickers = []
    try:
        with open(filepath, 'r') as file:
            for line in file:
                if line.startswith("tickers"):
                    _, tickers_str = line.strip().split('=')
                    tickers = [t.strip() for t in tickers_str.split(',')]
    except Exception as e:
        print(f"Error reading tickers from {filepath}: {e}")
    return tickers
