#!/usr/bin/env python3
"""
Unit tests for Config Loader utilities
"""
import sys
import os
import unittest
import tempfile
from unittest.mock import patch, mock_open
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.configLoader import load_symbols_from_config

class TestConfigLoader(unittest.TestCase):
    """Test cases for config loading functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_config = """[DEFAULT]
sector=nifty_50
roe_min = 15
pe_ratio_max = 20
"""
    
    @patch('utils.configLoader.load_tickers')
    @patch('configparser.ConfigParser.read')
    @patch('configparser.ConfigParser.get')
    def test_load_symbols_success(self, mock_get, mock_read, mock_load_tickers):
        """Test successful symbol loading"""
        mock_get.return_value = 'nifty_50'
        mock_load_tickers.return_value = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS']
        
        result = load_symbols_from_config()
        
        self.assertEqual(result, ['RELIANCE.NS', 'TCS.NS', 'INFY.NS'])
        mock_get.assert_called_once_with('DEFAULT', 'sector', fallback=None)
        mock_load_tickers.assert_called_once_with('nifty_50')
    
    @patch('configparser.ConfigParser.read')
    @patch('configparser.ConfigParser.get')
    def test_load_symbols_no_sector(self, mock_get, mock_read):
        """Test loading with no sector specified"""
        mock_get.return_value = None
        
        with self.assertRaises(ValueError) as context:
            load_symbols_from_config()
        
        self.assertIn("Sector not specified", str(context.exception))
    
    @patch('utils.configLoader.load_tickers')
    @patch('configparser.ConfigParser.read')
    @patch('configparser.ConfigParser.get')
    def test_load_symbols_empty_sector(self, mock_get, mock_read, mock_load_tickers):
        """Test loading with empty sector string"""
        mock_get.return_value = ''
        
        with self.assertRaises(ValueError) as context:
            load_symbols_from_config()
        
        self.assertIn("Sector not specified", str(context.exception))

if __name__ == '__main__':
    unittest.main()
