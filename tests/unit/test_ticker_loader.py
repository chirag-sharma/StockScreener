#!/usr/bin/env python3
"""
Unit tests for Ticker Loader utilities
"""
import sys
import os
import unittest
import tempfile
import json
from unittest.mock import patch, mock_open
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.tickerLoader import load_tickers, is_cache_valid, normalize_sector_name

class TestTickerLoader(unittest.TestCase):
    """Test cases for ticker loading functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_ticker_data = {
            "date": "2024-01-01",
            "tickers": ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
        }
    
    def test_normalize_sector_name(self):
        """Test sector name normalization"""
        self.assertEqual(normalize_sector_name('nifty_50'), 'nifty_50')
        self.assertEqual(normalize_sector_name('nifty_it'), 'it_sector')
        self.assertEqual(normalize_sector_name('nifty_pharma'), 'pharma_sector')
        self.assertEqual(normalize_sector_name('unknown_sector'), 'unknown_sector')
    
    def test_is_cache_valid_recent(self):
        """Test cache validation with recent data"""
        from datetime import datetime
        today = datetime.today().strftime("%Y-%m-%d")
        
        recent_data = {"date": today, "tickers": ["TEST.NS"]}
        self.assertTrue(is_cache_valid(recent_data))
    
    def test_is_cache_valid_old(self):
        """Test cache validation with old data"""
        old_data = {"date": "2020-01-01", "tickers": ["TEST.NS"]}
        self.assertFalse(is_cache_valid(old_data))
    
    def test_is_cache_valid_invalid_format(self):
        """Test cache validation with invalid format"""
        invalid_data = {"tickers": ["TEST.NS"]}  # Missing date
        self.assertFalse(is_cache_valid(invalid_data))
        
        invalid_data2 = {"date": "2024-01-01"}  # Missing tickers
        self.assertFalse(is_cache_valid(invalid_data2))
    
    def test_load_tickers_direct_file_path(self):
        """Test loading tickers from a direct file path"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.sample_ticker_data, f)
            temp_file_path = f.name
        
        try:
            result = load_tickers(temp_file_path)
            self.assertEqual(result, ["RELIANCE.NS", "TCS.NS", "INFY.NS"])
        finally:
            os.unlink(temp_file_path)
    
    def test_load_tickers_invalid_file(self):
        """Test loading tickers from invalid file"""
        result = load_tickers("/nonexistent/file.json")
        # Should handle gracefully and return empty list or appropriate default
        self.assertIsInstance(result, list)
    
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_load_tickers_file_read_error(self, mock_file, mock_exists):
        """Test handling of file read errors"""
        mock_exists.return_value = True
        mock_file.side_effect = IOError("Cannot read file")
        
        # Should handle the error gracefully
        try:
            result = load_tickers("test.json")
            self.assertIsInstance(result, list)
        except Exception as e:
            self.fail(f"load_tickers should handle file errors gracefully: {e}")
    
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_load_tickers_json_decode_error(self, mock_file, mock_exists):
        """Test handling of JSON decode errors"""
        mock_exists.return_value = True
        mock_file.return_value.read.return_value = "invalid json"
        
        # Should handle the error gracefully
        try:
            result = load_tickers("test.json")
            self.assertIsInstance(result, list)
        except Exception as e:
            self.fail(f"load_tickers should handle JSON errors gracefully: {e}")

if __name__ == '__main__':
    unittest.main()
