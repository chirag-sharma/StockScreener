"""
Unit Tests for Base Prediction Model
===================================

Tests the abstract base class and common functionality.
"""

import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from datetime import datetime

from stock_screener.prediction_models.base_model import BasePredictionModel


class ConcretePredictionModel(BasePredictionModel):
    """Concrete implementation for testing the abstract base class."""
    
    def predict(self):
        return {
            "predicted_price": 1000.0,
            "confidence": 0.5,
            "method": "Test Model"
        }
    
    def get_confidence(self):
        return 0.5


class TestBasePredictionModel(unittest.TestCase):
    """Test cases for BasePredictionModel."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample historical data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        self.sample_data = pd.DataFrame({
            'Open': np.random.uniform(100, 110, 100),
            'High': np.random.uniform(110, 120, 100),
            'Low': np.random.uniform(90, 100, 100),
            'Close': np.random.uniform(100, 110, 100),
            'Volume': np.random.uniform(1000000, 5000000, 100)
        }, index=dates)
        
        self.symbol = "TEST.NS"
        self.model = ConcretePredictionModel(self.symbol, self.sample_data)
    
    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.symbol, "TEST.NS")
        self.assertIsInstance(self.model.historical_data, pd.DataFrame)
        self.assertEqual(len(self.model.historical_data), 100)
        self.assertIsNotNone(self.model.logger)
    
    def test_validate_inputs_valid(self):
        """Test input validation with valid data."""
        self.assertTrue(self.model.validate_inputs())
    
    def test_validate_inputs_empty_data(self):
        """Test input validation with empty data."""
        empty_model = ConcretePredictionModel("TEST", pd.DataFrame())
        self.assertFalse(empty_model.validate_inputs())
    
    def test_validate_inputs_missing_columns(self):
        """Test input validation with missing required columns."""
        incomplete_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'Close': [100.5, 101.5, 102.5]
            # Missing High, Low, Volume
        })
        incomplete_model = ConcretePredictionModel("TEST", incomplete_data)
        self.assertFalse(incomplete_model.validate_inputs())
    
    def test_get_current_price(self):
        """Test current price retrieval."""
        current_price = self.model.get_current_price()
        self.assertIsInstance(current_price, float)
        self.assertGreater(current_price, 0)
        # Should be the last close price
        expected = float(self.sample_data['Close'].iloc[-1])
        self.assertEqual(current_price, expected)
    
    def test_get_support_resistance_levels(self):
        """Test support and resistance calculation."""
        levels = self.model.get_support_resistance_levels()
        
        self.assertIn('support', levels)
        self.assertIn('resistance', levels)
        self.assertIsInstance(levels['support'], float)
        self.assertIsInstance(levels['resistance'], float)
        self.assertLess(levels['support'], levels['resistance'])
    
    def test_calculate_returns(self):
        """Test returns calculation."""
        returns = self.model.calculate_returns()
        
        self.assertIsInstance(returns, pd.Series)
        self.assertEqual(len(returns), len(self.sample_data) - 1)  # One less due to diff
        # Check that returns are percentage changes
        manual_returns = self.sample_data['Close'].pct_change().dropna()
        pd.testing.assert_series_equal(returns, manual_returns)
    
    def test_calculate_volatility(self):
        """Test volatility calculation."""
        volatility = self.model.calculate_volatility()
        
        self.assertIsInstance(volatility, float)
        self.assertGreater(volatility, 0)
        # Should be standard deviation of returns
        returns = self.sample_data['Close'].pct_change().dropna()
        expected_volatility = returns.std()
        self.assertAlmostEqual(volatility, expected_volatility, places=5)
    
    def test_predict_abstract_method(self):
        """Test that predict method is implemented."""
        result = self.model.predict()
        self.assertIn('predicted_price', result)
        self.assertIn('confidence', result)
        self.assertIn('method', result)
    
    def test_get_confidence_abstract_method(self):
        """Test that get_confidence method is implemented."""
        confidence = self.model.get_confidence()
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_prediction_days_parameter(self):
        """Test prediction_days parameter handling."""
        model_30 = ConcretePredictionModel("TEST", self.sample_data, prediction_days=30)
        self.assertEqual(model_30.prediction_days, 30)
        
        model_60 = ConcretePredictionModel("TEST", self.sample_data, prediction_days=60)
        self.assertEqual(model_60.prediction_days, 60)
    
    def test_support_resistance_edge_cases(self):
        """Test support/resistance with edge cases."""
        # Flat prices
        flat_data = self.sample_data.copy()
        flat_data['High'] = 100
        flat_data['Low'] = 100
        flat_data['Close'] = 100
        
        flat_model = ConcretePredictionModel("TEST", flat_data)
        levels = flat_model.get_support_resistance_levels()
        
        # Should still return valid levels
        self.assertIsInstance(levels['support'], float)
        self.assertIsInstance(levels['resistance'], float)


class TestBasePredictionModelEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def test_insufficient_data(self):
        """Test behavior with insufficient data."""
        # Only 2 days of data
        minimal_data = pd.DataFrame({
            'Open': [100, 101],
            'High': [105, 106],
            'Low': [95, 96],
            'Close': [102, 103],
            'Volume': [1000000, 1100000]
        })
        
        model = ConcretePredictionModel("TEST", minimal_data)
        self.assertFalse(model.validate_inputs())
    
    def test_nan_values_handling(self):
        """Test handling of NaN values in data."""
        data_with_nans = pd.DataFrame({
            'Open': [100, np.nan, 102],
            'High': [105, 106, np.nan],
            'Low': [95, 96, 97],
            'Close': [102, 103, 104],
            'Volume': [1000000, 1100000, 1200000]
        })
        
        model = ConcretePredictionModel("TEST", data_with_nans)
        # Should handle NaN values gracefully
        current_price = model.get_current_price()
        self.assertIsInstance(current_price, float)
        self.assertFalse(np.isnan(current_price))
    
    def test_invalid_symbol_handling(self):
        """Test handling of invalid symbols."""
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        valid_data = pd.DataFrame({
            'Open': [100] * 10,
            'High': [105] * 10,
            'Low': [95] * 10,
            'Close': [102] * 10,
            'Volume': [1000000] * 10
        }, index=dates)
        
        # Test with empty symbol
        model = ConcretePredictionModel("", valid_data)
        self.assertEqual(model.symbol, "")
        
        # Test with None symbol
        model = ConcretePredictionModel(None, valid_data)
        self.assertIsNone(model.symbol)


if __name__ == '__main__':
    unittest.main()
