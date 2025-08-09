# StockScreener Package
"""
A comprehensive value investing stock screener with weighted scoring system.

This package provides tools for analyzing stocks based on fundamental value investing
principles, implementing weighted scoring algorithms inspired by Warren Buffett,
Benjamin Graham, Peter Lynch, and Joel Greenblatt.
"""

__version__ = "1.0.0"
__author__ = "Chirag Sharma"
__email__ = "chirag@example.com"

# Core components
from src.screener import main as run_screener

# Services
from services.stockAnalyzer import StockAnalyzer
from services.excelExporter import ExcelExporter

# Utilities
from utils.tickerLoader import load_tickers
from utils.configLoader import load_symbols_from_config

__all__ = [
    'run_screener',
    'StockAnalyzer', 
    'ExcelExporter',
    'load_tickers',
    'load_symbols_from_config'
]
