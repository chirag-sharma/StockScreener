# tests/conftest.py
"""
Pytest configuration and shared fixtures for the test suite.
"""
import pytest
import sys
import os
import json
from unittest.mock import MagicMock

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

@pytest.fixture
def mock_stock_data():
    """Mock stock data for testing."""
    return {
        'symbol': 'TEST.NS',
        'trailingPE': 15.5,
        'debtToEquity': 0.5,
        'returnOnEquity': 0.18,
        'currentRatio': 2.1,
        'priceToBook': 2.5,
        'heldPercentInsiders': 0.65,
        'quickRatio': 1.8,
        'freeCashflow': 1000000000,
        'earningsQuarterlyGrowth': 0.15,
        'returnOnAssets': 0.12,
        'operatingMargins': 0.20,
        'profitMargins': 0.15,
        'marketCap': 50000000000,
        'totalRevenue': 10000000000,
        'netIncomeToCommon': 1500000000,
        'enterpriseToEbitda': 8.5,
        'revenueGrowth': 0.12
    }

@pytest.fixture
def mock_ticker_list():
    """Mock ticker list for testing."""
    return ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS']

@pytest.fixture
def sample_analysis_result():
    """Sample analysis result for testing."""
    return {
        'Symbol': 'TEST.NS',
        'PE Ratio': 15.5,
        'Debt/Equity': 0.5,
        'ROE': 18.0,
        'Current Ratio': 2.1,
        'Price to Book': 2.5,
        'Value Score': 75.5,
        'Investment Recommendation': 'Buy'
    }

@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary config file for testing."""
    config_content = """[DEFAULT]
sector=nifty_50
roe_min = 15
pe_ratio_max = 20
de_ratio_max = 1
current_ratio_min = 1.5
price_to_book_max = 3
"""
    config_file = tmp_path / "test_config.properties"
    config_file.write_text(config_content)
    return str(config_file)

@pytest.fixture
def temp_ticker_file(tmp_path):
    """Create a temporary ticker file for testing."""
    ticker_data = {
        "date": "2024-01-01",
        "tickers": ["TEST.NS", "SAMPLE.NS", "MOCK.NS"]
    }
    ticker_file = tmp_path / "test_tickers.json"
    ticker_file.write_text(json.dumps(ticker_data))
    return str(ticker_file)
