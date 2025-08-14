#!/usr/bin/env python3
"""
Unit tests for Excel Exporter
"""
import sys
import os
import unittest
import tempfile
import pandas as pd
from unittest.mock import patch, MagicMock
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from services.excelExporter import ExcelExporter

class TestExcelExporter(unittest.TestCase):
    """Test cases for ExcelExporter class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)
        self.temp_file.close()
        self.exporter = ExcelExporter(self.temp_file.name)
        
        self.sample_data = [
            {
                'Symbol': 'TEST.NS',
                'PE Ratio': 15.5,
                'Debt/Equity': 0.5,
                'ROE': 18.0,
                'Current Ratio': 2.1,
                'Value Score': 75.5,
                'Investment Recommendation': 'Buy'
            },
            {
                'Symbol': 'SAMPLE.NS',
                'PE Ratio': 25.0,
                'Debt/Equity': 1.5,
                'ROE': 8.0,
                'Current Ratio': 1.0,
                'Value Score': 35.2,
                'Investment Recommendation': 'Avoid'
            }
        ]
    
    def tearDown(self):
        """Clean up test files"""
        try:
            os.unlink(self.temp_file.name)
        except:
            pass
    
    def test_initialization(self):
        """Test exporter initialization"""
        self.assertEqual(self.exporter.output_path, self.temp_file.name)
    
    def test_write_data_success(self):
        """Test successful data writing"""
        try:
            self.exporter.write_data(self.sample_data)
            
            # Check if file was created
            self.assertTrue(os.path.exists(self.temp_file.name))
            
            # Read back the data to verify
            df = pd.read_excel(self.temp_file.name)
            self.assertEqual(len(df), len(self.sample_data))
            self.assertIn('Symbol', df.columns)
            self.assertIn('Value Score', df.columns)
            
        except Exception as e:
            self.fail(f"write_data raised an exception: {e}")
    
    def test_write_empty_data(self):
        """Test writing empty data"""
        try:
            self.exporter.write_data([])
            # Should not raise an exception
        except Exception as e:
            self.fail(f"write_data with empty list raised an exception: {e}")
    
    def test_write_data_with_none_values(self):
        """Test writing data with None values"""
        data_with_none = [
            {
                'Symbol': 'TEST.NS',
                'PE Ratio': None,
                'Debt/Equity': 0.5,
                'ROE': 18.0,
                'Value Score': 75.5,
                'Investment Recommendation': 'Buy'
            }
        ]
        
        try:
            self.exporter.write_data(data_with_none)
            
            # Read back and check None handling
            df = pd.read_excel(self.temp_file.name)
            self.assertEqual(len(df), 1)
            
        except Exception as e:
            self.fail(f"write_data with None values raised an exception: {e}")
    
    @patch('pandas.DataFrame.to_excel')
    def test_write_data_exception_handling(self, mock_to_excel):
        """Test exception handling in write_data"""
        mock_to_excel.side_effect = Exception("Write failed")
        
        # Should not raise exception due to try-catch
        try:
            self.exporter.write_data(self.sample_data)
        except Exception as e:
            self.fail(f"write_data should handle exceptions gracefully: {e}")

if __name__ == '__main__':
    unittest.main()
