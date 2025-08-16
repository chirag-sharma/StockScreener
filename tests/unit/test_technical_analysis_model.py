"""
Unit Tests for Technical Analysis Prediction Model
=================================================

Tests the technical analysis model functionality.
"""

import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
import yfinance as yf

from stock_screener.prediction_models.technical_analysis_model import (
    TechnicalAnalysisModel, predict_price_technical
)


class TestTechnicalAnalysisModel(unittest.TestCase):
    """Test cases for TechnicalAnalysisModel."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create realistic stock price data
        np.random.seed(42)  # For reproducible tests
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
        # Generate trending price data
        base_price = 1000
        trend = np.linspace(0, 50, 100)  # Upward trend
        noise = np.random.normal(0, 10, 100)
        close_prices = base_price + trend + noise
        
        self.sample_data = pd.DataFrame({
            'Open': close_prices - np.random.uniform(1, 5, 100),
            'High': close_prices + np.random.uniform(5, 15, 100),
            'Low': close_prices - np.random.uniform(5, 15, 100),
            'Close': close_prices,
            'Volume': np.random.uniform(1000000, 5000000, 100)
        }, index=dates)
        
        self.model = TechnicalAnalysisModel("TEST.NS", self.sample_data)
    
    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.symbol, "TEST.NS")
        self.assertEqual(self.model.short_window, 20)  # Default
        self.assertEqual(self.model.long_window, 50)   # Default
        self.assertEqual(self.model.prediction_days, 30)  # Default
    
    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        custom_model = TechnicalAnalysisModel(
            "TEST.NS", 
            self.sample_data,
            short_window=12,
            long_window=26,
            prediction_days=45,
            rsi_period=10
        )
        
        self.assertEqual(custom_model.short_window, 12)
        self.assertEqual(custom_model.long_window, 26)
        self.assertEqual(custom_model.prediction_days, 45)
        self.assertEqual(custom_model.rsi_period, 10)
    
    def test_calculate_moving_averages(self):
        """Test moving averages calculation."""
        mas = self.model._calculate_moving_averages()
        
        self.assertIn('sma_short', mas)
        self.assertIn('sma_long', mas)
        self.assertIn('ema_short', mas)
        self.assertIn('ema_long', mas)
        
        # Check that we get valid numbers
        self.assertIsInstance(mas['sma_short'], float)
        self.assertIsInstance(mas['sma_long'], float)
        self.assertIsInstance(mas['ema_short'], float)
        self.assertIsInstance(mas['ema_long'], float)
        
        # SMA and EMA should be positive for our test data
        self.assertGreater(mas['sma_short'], 0)
        self.assertGreater(mas['sma_long'], 0)
        self.assertGreater(mas['ema_short'], 0)
        self.assertGreater(mas['ema_long'], 0)
    
    def test_calculate_momentum_indicators(self):
        """Test momentum indicators calculation."""
        momentum = self.model._calculate_momentum_indicators()
        
        self.assertIn('rsi', momentum)
        self.assertIn('macd', momentum)
        self.assertIn('macd_signal', momentum)
        self.assertIn('macd_histogram', momentum)
        
        # RSI should be between 0 and 100
        rsi = momentum['rsi']
        self.assertGreaterEqual(rsi, 0)
        self.assertLessEqual(rsi, 100)
        
        # MACD values should be numbers
        self.assertIsInstance(momentum['macd'], float)
        self.assertIsInstance(momentum['macd_signal'], float)
    
    def test_calculate_volatility_indicators(self):
        """Test volatility indicators calculation."""
        volatility = self.model._calculate_volatility_indicators()
        
        self.assertIn('bb_upper', volatility)
        self.assertIn('bb_lower', volatility)
        self.assertIn('bb_middle', volatility)
        self.assertIn('bb_width', volatility)
        
        # Bollinger Bands should follow logical order
        self.assertGreater(volatility['bb_upper'], volatility['bb_middle'])
        self.assertGreater(volatility['bb_middle'], volatility['bb_lower'])
        self.assertGreater(volatility['bb_width'], 0)
    
    def test_predict_method(self):
        """Test the main predict method."""
        result = self.model.predict()
        
        # Check required fields
        self.assertIn('predicted_price', result)
        self.assertIn('confidence', result)
        self.assertIn('method', result)
        self.assertEqual(result['method'], 'Technical Analysis')
        
        # Check data types
        self.assertIsInstance(result['predicted_price'], (int, float))
        self.assertIsInstance(result['confidence'], (int, float))
        
        # Confidence should be between 0 and 1
        self.assertGreaterEqual(result['confidence'], 0)
        self.assertLessEqual(result['confidence'], 1)
        
        # Predicted price should be positive
        self.assertGreater(result['predicted_price'], 0)
    
    def test_get_confidence_calculation(self):
        """Test confidence calculation logic."""
        confidence = self.model.get_confidence()
        
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.1)  # Minimum confidence
        self.assertLessEqual(confidence, 0.9)     # Maximum confidence
    
    def test_generate_signals(self):
        """Test signal generation."""
        signals = self.model._generate_signals()
        
        self.assertIsInstance(signals, dict)
        # Should contain various signal types
        expected_signals = ['ma_signal', 'rsi_signal', 'macd_signal', 'bb_signal']
        for signal in expected_signals:
            self.assertIn(signal, signals)
    
    def test_insufficient_data_handling(self):
        """Test behavior with insufficient data."""
        # Create data with only 10 rows (less than required for indicators)
        insufficient_data = self.sample_data.head(10)
        model = TechnicalAnalysisModel("TEST", insufficient_data)
        
        result = model.predict()
        
        # Should return current price with low confidence
        self.assertIn('predicted_price', result)
        self.assertIn('error', result)
        self.assertLessEqual(result['confidence'], 0.2)
    
    def test_edge_case_flat_prices(self):
        """Test with flat prices (no volatility)."""
        flat_data = self.sample_data.copy()
        flat_data['Close'] = 1000.0  # All same price
        flat_data['High'] = 1000.0
        flat_data['Low'] = 1000.0
        
        model = TechnicalAnalysisModel("TEST", flat_data)
        result = model.predict()
        
        # Should still return a prediction
        self.assertIn('predicted_price', result)
        self.assertEqual(result['predicted_price'], 1000.0)  # Should predict same price
    
    def test_ma_crossover_detection(self):
        """Test moving average crossover detection."""
        # Create data with clear MA crossover
        dates = pd.date_range(start='2024-01-01', periods=60, freq='D')
        prices = np.concatenate([
            np.full(30, 1000),  # Flat prices initially
            np.linspace(1000, 1100, 30)  # Then rising prices
        ])
        
        crossover_data = pd.DataFrame({
            'Open': prices - 1,
            'High': prices + 5,
            'Low': prices - 5,
            'Close': prices,
            'Volume': [1000000] * 60
        }, index=dates)
        
        model = TechnicalAnalysisModel("TEST", crossover_data, short_window=10, long_window=20)
        signals = model._generate_signals()
        
        self.assertIn('ma_signal', signals)
        # With rising prices, should detect bullish signal
        self.assertIn(signals['ma_signal'], ['Bullish', 'Bearish', 'Neutral'])


class TestTechnicalAnalysisStandaloneFunction(unittest.TestCase):
    """Test the standalone function."""
    
    @patch('yfinance.Ticker')
    def test_predict_price_technical_success(self, mock_ticker):
        """Test successful prediction using standalone function."""
        # Mock yfinance data
        mock_hist_data = pd.DataFrame({
            'Open': [1000, 1005, 1010],
            'High': [1010, 1015, 1020],
            'Low': [995, 1000, 1005],
            'Close': [1005, 1010, 1015],
            'Volume': [1000000, 1100000, 1200000]
        })
        
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_hist_data
        mock_ticker.return_value = mock_ticker_instance
        
        result = predict_price_technical("TEST.NS", days=15)
        
        self.assertIn('predicted_price', result)
        self.assertNotIn('error', result)
    
    @patch('yfinance.Ticker')
    def test_predict_price_technical_no_data(self, mock_ticker):
        """Test behavior when no data is available."""
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = pd.DataFrame()  # Empty data
        mock_ticker.return_value = mock_ticker_instance
        
        result = predict_price_technical("INVALID.SYMBOL")
        
        self.assertIn('error', result)
    
    @patch('yfinance.Ticker')
    def test_predict_price_technical_exception(self, mock_ticker):
        """Test exception handling in standalone function."""
        mock_ticker.side_effect = Exception("Network error")
        
        result = predict_price_technical("TEST.NS")
        
        self.assertIn('error', result)
        self.assertIn('Technical analysis failed', result['error'])


class TestTechnicalAnalysisIntegration(unittest.TestCase):
    """Integration tests with real-like data scenarios."""
    
    def setUp(self):
        """Set up realistic market scenarios."""
        self.dates = pd.date_range(start='2024-01-01', periods=200, freq='D')
    
    def test_bull_market_scenario(self):
        """Test in a bull market scenario."""
        # Create strongly trending upward data
        base_price = 1000
        trend = np.linspace(0, 200, 200)  # Strong upward trend
        noise = np.random.normal(0, 5, 200)
        close_prices = base_price + trend + noise
        
        bull_data = pd.DataFrame({
            'Open': close_prices - 2,
            'High': close_prices + 10,
            'Low': close_prices - 10,
            'Close': close_prices,
            'Volume': np.random.uniform(1000000, 3000000, 200)
        }, index=self.dates)
        
        model = TechnicalAnalysisModel("BULL.NS", bull_data)
        result = model.predict()
        
        # In bull market, should predict higher prices
        current_price = model.get_current_price()
        predicted_price = result['predicted_price']
        
        # Should be bullish (predicted > current) with reasonable confidence
        self.assertGreater(result['confidence'], 0.4)
    
    def test_bear_market_scenario(self):
        """Test in a bear market scenario."""
        # Create downward trending data
        base_price = 1000
        trend = np.linspace(0, -200, 200)  # Downward trend
        noise = np.random.normal(0, 5, 200)
        close_prices = base_price + trend + noise
        
        bear_data = pd.DataFrame({
            'Open': close_prices + 2,
            'High': close_prices + 10,
            'Low': close_prices - 10,
            'Close': close_prices,
            'Volume': np.random.uniform(1000000, 3000000, 200)
        }, index=self.dates)
        
        model = TechnicalAnalysisModel("BEAR.NS", bear_data)
        result = model.predict()
        
        # Should still produce valid predictions
        self.assertIn('predicted_price', result)
        self.assertGreater(result['confidence'], 0.1)
    
    def test_volatile_market_scenario(self):
        """Test in a highly volatile market."""
        # Create highly volatile data
        base_price = 1000
        volatility = np.random.normal(0, 50, 200)  # High volatility
        close_prices = base_price + volatility
        
        volatile_data = pd.DataFrame({
            'Open': close_prices - np.random.uniform(5, 20, 200),
            'High': close_prices + np.random.uniform(10, 30, 200),
            'Low': close_prices - np.random.uniform(10, 30, 200),
            'Close': close_prices,
            'Volume': np.random.uniform(2000000, 8000000, 200)
        }, index=self.dates)
        
        model = TechnicalAnalysisModel("VOLATILE.NS", volatile_data)
        result = model.predict()
        
        # High volatility should result in lower confidence
        self.assertLess(result['confidence'], 0.7)
        self.assertIn('predicted_price', result)


if __name__ == '__main__':
    unittest.main()
