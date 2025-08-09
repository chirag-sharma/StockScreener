# src/indicators.py
"""
This module provides technical indicator calculation utilities for stock analysis.
Currently includes Relative Strength Index (RSI) calculation.
"""

import pandas as pd

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI) for a given price series.

    Args:
        series (pd.Series): Series of prices (e.g., closing prices).
        period (int): Number of periods to use for RSI calculation (default: 14).

    Returns:
        pd.Series: RSI values for the input series.
    """
    # Calculate price differences between consecutive periods
    delta = series.diff()
    # Separate gains (positive deltas) and losses (negative deltas)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    # Calculate rolling average gain and loss
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    # Compute the relative strength (RS)
    rs = avg_gain / avg_loss
    # Calculate RSI using the RS value
    rsi = 100 - (100 / (1 + rs))
    return rsi
