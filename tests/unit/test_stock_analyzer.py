#!/usr/bin/env python3
"""
Unit tests for StockAnalyzer class
"""
import sys
import os
import unittest
from unittest.mock import MagicMock, patch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from services.stockAnalyzer import StockAnalyzer

class TestStockAnalyzer(unittest.TestCase):
    """Test cases for StockAnalyzer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = StockAnalyzer("TEST.NS")
        self.mock_data = {
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
    
    def test_initialization(self):
        """Test analyzer initialization"""
        self.assertEqual(self.analyzer.symbol, "TEST.NS")
        self.assertEqual(self.analyzer.info, {})
        self.assertEqual(self.analyzer.analysis, {})
    
    @patch('yfinance.Ticker')
    def test_fetch_data_success(self, mock_ticker):
        """Test successful data fetching"""
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = self.mock_data
        mock_ticker.return_value = mock_ticker_instance
        
        self.analyzer.fetch_data()
        
        self.assertEqual(self.analyzer.info, self.mock_data)
        mock_ticker.assert_called_once_with("TEST.NS")
    
    @patch('yfinance.Ticker')
    def test_fetch_data_failure(self, mock_ticker):
        """Test data fetching with exception"""
        mock_ticker.side_effect = Exception("API Error")
        
        self.analyzer.fetch_data()
        
        self.assertEqual(self.analyzer.info, {})
    
    def test_analyze_no_data(self):
        """Test analysis with no data"""
        result = self.analyzer.analyze()
        self.assertIsNone(result)
    
    @patch('yfinance.Ticker')
    def test_analyze_with_data(self, mock_ticker):
        """Test analysis with mock data"""
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = self.mock_data
        mock_ticker.return_value = mock_ticker_instance
        
        self.analyzer.fetch_data()
        result = self.analyzer.analyze()
        
        self.assertIsNotNone(result)
        self.assertIn('Symbol', result)
        self.assertIn('Value Score', result)
        self.assertIn('Investment Recommendation', result)
        self.assertEqual(result['Symbol'], "TEST.NS")
        
        # Check that Value Score is a valid number between 0 and 100
        score = result['Value Score']
        self.assertIsInstance(score, (int, float))
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)
    
    def test_score_metric_pe_ratio(self):
        """Test PE ratio scoring"""
        # Initialize analyzer with mock data
        self.analyzer.info = self.mock_data
        
        # Test different PE ratio values
        self.assertEqual(self.analyzer._score_metric('PE Ratio', 10), 100.0)
        self.assertEqual(self.analyzer._score_metric('PE Ratio', 15), 80.0)
        self.assertEqual(self.analyzer._score_metric('PE Ratio', 20), 60.0)
        self.assertEqual(self.analyzer._score_metric('PE Ratio', 25), 40.0)
        self.assertEqual(self.analyzer._score_metric('PE Ratio', 30), 20.0)
        self.assertEqual(self.analyzer._score_metric('PE Ratio', 35), 0.0)
        self.assertEqual(self.analyzer._score_metric('PE Ratio', 0), 0.0)
        self.assertEqual(self.analyzer._score_metric('PE Ratio', -5), 0.0)
    
    def test_score_metric_debt_equity(self):
        """Test Debt/Equity ratio scoring"""
        self.analyzer.info = self.mock_data
        
        self.assertEqual(self.analyzer._score_metric('Debt/Equity', 0.3), 100.0)
        self.assertEqual(self.analyzer._score_metric('Debt/Equity', 0.5), 80.0)
        self.assertEqual(self.analyzer._score_metric('Debt/Equity', 0.8), 60.0)
        self.assertEqual(self.analyzer._score_metric('Debt/Equity', 1.0), 40.0)
        self.assertEqual(self.analyzer._score_metric('Debt/Equity', 1.5), 20.0)
        self.assertEqual(self.analyzer._score_metric('Debt/Equity', 2.0), 0.0)
    
    def test_score_metric_roe(self):
        """Test ROE scoring"""
        self.analyzer.info = self.mock_data
        
        self.assertEqual(self.analyzer._score_metric('ROE', 25), 100.0)
        self.assertEqual(self.analyzer._score_metric('ROE', 20), 80.0)
        self.assertEqual(self.analyzer._score_metric('ROE', 15), 60.0)
        self.assertEqual(self.analyzer._score_metric('ROE', 10), 40.0)
        self.assertEqual(self.analyzer._score_metric('ROE', 5), 20.0)
        self.assertEqual(self.analyzer._score_metric('ROE', 0), 0.0)
    
    def test_score_metric_invalid_values(self):
        """Test scoring with invalid values"""
        self.analyzer.info = self.mock_data
        
        self.assertEqual(self.analyzer._score_metric('PE Ratio', 'N/A'), 0.0)
        self.assertEqual(self.analyzer._score_metric('PE Ratio', None), 0.0)
        self.assertEqual(self.analyzer._score_metric('Unknown Metric', 15), 50.0)
    
    def test_calculate_weighted_score(self):
        """Test weighted score calculation"""
        self.analyzer.info = self.mock_data
        self.analyzer.analysis = {
            'PE Ratio': 15.0,
            'Price to Book': 2.0,
            'EV/EBITDA': 8.0,
            'Margin of Safety (%)': 10.0,
            'ROE': 20.0,
            'Net Profit Margin': 15.0,
            'Operating Margin': 18.0,
            'Return on Assets (ROA)': 12.0,
            'Current Ratio': 2.0,
            'Debt/Equity': 0.5,
            'Quick Ratio': 1.8,
            'Interest Coverage Ratio': 5.0,
            'Free Cash Flow': 1000000000,
            'EPS Growth (%)': 15.0,
            'Revenue Growth (%)': 12.0
        }
        
        score = self.analyzer._calculate_weighted_score()
        
        self.assertIsInstance(score, (int, float))
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)
    
    def test_check_method(self):
        """Test the _check method"""
        self.analyzer.analysis = {'Test Metric': 15.0}
        
        # Test less than condition
        self.assertTrue(self.analyzer._check('Test Metric', '<', 20))
        self.assertFalse(self.analyzer._check('Test Metric', '<', 10))
        
        # Test greater than condition
        self.assertTrue(self.analyzer._check('Test Metric', '>', 10))
        self.assertFalse(self.analyzer._check('Test Metric', '>', 20))
        
        # Test with missing value
        self.assertFalse(self.analyzer._check('Missing Metric', '<', 20))

if __name__ == '__main__':
    unittest.main()
