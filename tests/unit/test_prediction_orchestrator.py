"""
Unit Tests for Price Prediction Orchestrator
===========================================

Tests the main orchestrator that coordinates all prediction models.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

from stock_screener.prediction_models.prediction_orchestrator import (
    PricePredictionOrchestrator, predict_stock_price, compare_prediction_models
)


class TestPricePredictionOrchestrator(unittest.TestCase):
    """Test cases for PricePredictionOrchestrator."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample historical data
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
        self.sample_data = pd.DataFrame({
            'Open': np.random.uniform(100, 110, 100),
            'High': np.random.uniform(110, 120, 100),
            'Low': np.random.uniform(90, 100, 100),
            'Close': np.random.uniform(100, 110, 100),
            'Volume': np.random.uniform(1000000, 5000000, 100)
        }, index=dates)
        
        self.orchestrator = PricePredictionOrchestrator("TEST.NS", self.sample_data)
    
    def test_initialization_with_data(self):
        """Test orchestrator initialization with provided data."""
        self.assertEqual(self.orchestrator.symbol, "TEST.NS")
        self.assertIsNotNone(self.orchestrator.historical_data)
        self.assertEqual(len(self.orchestrator.historical_data), 100)
        self.assertIn('technical', self.orchestrator.models)
        self.assertIn('fundamental', self.orchestrator.models)
        self.assertIn('machine_learning', self.orchestrator.models)
        
    @patch('yfinance.Ticker')
    def test_initialization_without_data(self, mock_ticker):
        """Test orchestrator initialization with data fetching."""
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = self.sample_data
        mock_ticker.return_value = mock_ticker_instance
        
        orchestrator = PricePredictionOrchestrator("TEST.NS")
        
        self.assertEqual(orchestrator.symbol, "TEST.NS")
        self.assertIsNotNone(orchestrator.historical_data)
        mock_ticker.assert_called_once_with("TEST.NS")
    
    def test_custom_model_weights(self):
        """Test initialization with custom model weights."""
        custom_weights = {
            'technical': 0.30,
            'fundamental': 0.10,
            'machine_learning': 0.40,
            'time_series': 0.10,
            'pattern_recognition': 0.05,
            'volume_analysis': 0.05
        }
        
        orchestrator = PricePredictionOrchestrator(
            "TEST.NS", 
            self.sample_data,
            model_weights=custom_weights
        )
        
        self.assertEqual(orchestrator.model_weights, custom_weights)
    
    def test_get_individual_predictions(self):
        """Test getting predictions from individual models."""
        predictions = self.orchestrator._get_individual_predictions(30)
        
        # Should have predictions from all 6 models
        expected_models = ['technical', 'fundamental', 'machine_learning', 
                          'time_series', 'pattern_recognition', 'volume_analysis']
        
        for model_name in expected_models:
            self.assertIn(model_name, predictions)
            prediction = predictions[model_name]
            
            # Each prediction should have required fields
            if 'error' not in prediction:
                self.assertIn('predicted_price', prediction)
                self.assertIn('confidence', prediction)
                self.assertIn('method', prediction)
    
    def test_predict_comprehensive(self):
        """Test comprehensive prediction method."""
        result = self.orchestrator.predict_comprehensive(target_days=30)
        
        # Check required fields
        required_fields = [
            'predicted_price', 'current_price', 'price_change', 
            'price_change_pct', 'confidence', 'market_signal', 
            'method', 'models_used', 'individual_predictions'
        ]
        
        for field in required_fields:
            self.assertIn(field, result)
        
        # Check data types and ranges
        self.assertIsInstance(result['predicted_price'], (int, float))
        self.assertIsInstance(result['current_price'], (int, float))
        self.assertIsInstance(result['confidence'], (int, float))
        self.assertGreaterEqual(result['confidence'], 0.1)
        self.assertLessEqual(result['confidence'], 0.9)
        
        # Check market signal is valid
        valid_signals = ['STRONG BUY', 'BUY', 'HOLD', 'SELL', 'STRONG SELL', 'UNCERTAIN']
        self.assertIn(result['market_signal'], valid_signals)
    
    def test_calculate_ensemble_prediction(self):
        """Test ensemble prediction calculation."""
        # Mock individual predictions
        mock_predictions = {
            'technical': {
                'predicted_price': 1000.0,
                'confidence': 0.8
            },
            'fundamental': {
                'predicted_price': 1100.0,
                'confidence': 0.6
            },
            'machine_learning': {
                'predicted_price': 1050.0,
                'confidence': 0.7
            }
        }
        
        result = self.orchestrator._calculate_ensemble_prediction(mock_predictions)
        
        self.assertIn('predicted_price', result)
        self.assertIn('confidence', result)
        self.assertIn('market_signal', result)
        
        # Ensemble price should be somewhere between individual predictions
        predicted_price = result['predicted_price']
        self.assertGreaterEqual(predicted_price, 950)  # Below minimum with some tolerance
        self.assertLessEqual(predicted_price, 1150)    # Above maximum with some tolerance
    
    def test_calculate_ensemble_confidence(self):
        """Test ensemble confidence calculation."""
        mock_predictions = [
            {'price': 1000, 'weight': 0.5, 'confidence': 0.8},
            {'price': 1050, 'weight': 0.3, 'confidence': 0.6},
            {'price': 1025, 'weight': 0.2, 'confidence': 0.7}
        ]
        
        confidence_scores = [p['confidence'] for p in mock_predictions]
        
        confidence = self.orchestrator._calculate_ensemble_confidence(
            mock_predictions, confidence_scores
        )
        
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.1)
        self.assertLessEqual(confidence, 0.9)
    
    def test_determine_market_signal(self):
        """Test market signal determination logic."""
        # Test strong buy
        signal = self.orchestrator._determine_market_signal(0.08, 0.7)
        self.assertEqual(signal, 'STRONG BUY')
        
        # Test buy
        signal = self.orchestrator._determine_market_signal(0.03, 0.5)
        self.assertEqual(signal, 'BUY')
        
        # Test hold
        signal = self.orchestrator._determine_market_signal(0.01, 0.5)
        self.assertEqual(signal, 'HOLD')
        
        # Test sell
        signal = self.orchestrator._determine_market_signal(-0.03, 0.5)
        self.assertEqual(signal, 'SELL')
        
        # Test strong sell
        signal = self.orchestrator._determine_market_signal(-0.08, 0.7)
        self.assertEqual(signal, 'STRONG SELL')
        
        # Test uncertain (low confidence)
        signal = self.orchestrator._determine_market_signal(0.05, 0.2)
        self.assertEqual(signal, 'UNCERTAIN')
    
    def test_assess_prediction_risk(self):
        """Test prediction risk assessment."""
        mock_predictions = [
            {'price': 1000}, {'price': 1020}, {'price': 980}
        ]
        
        risk_assessment = self.orchestrator._assess_prediction_risk(mock_predictions, 0.6)
        
        self.assertIn('risk_level', risk_assessment)
        self.assertIn('prediction_volatility', risk_assessment)
        self.assertIn('recommendation', risk_assessment)
        
        valid_risk_levels = ['LOW', 'MEDIUM', 'HIGH']
        self.assertIn(risk_assessment['risk_level'], valid_risk_levels)
    
    def test_get_individual_model_prediction(self):
        """Test getting prediction from a specific model."""
        # Test valid model
        result = self.orchestrator.get_individual_model_prediction('technical', days=15)
        
        if 'error' not in result:
            self.assertIn('predicted_price', result)
            self.assertIn('confidence', result)
        
        # Test invalid model
        result = self.orchestrator.get_individual_model_prediction('nonexistent', days=15)
        self.assertIn('error', result)
    
    def test_get_model_comparison(self):
        """Test model comparison functionality."""
        comparison = self.orchestrator.get_model_comparison(days=30)
        
        self.assertIn('symbol', comparison)
        self.assertIn('current_price', comparison)
        self.assertIn('prediction_comparison', comparison)
        
        # Should have comparisons for each model
        prediction_comparison = comparison['prediction_comparison']
        expected_models = ['technical', 'fundamental', 'machine_learning', 
                          'time_series', 'pattern_recognition', 'volume_analysis']
        
        for model in expected_models:
            self.assertIn(model, prediction_comparison)
    
    def test_backward_compatibility_methods(self):
        """Test backward compatibility methods."""
        # Test predict_price method (alias for predict_comprehensive)
        result = self.orchestrator.predict_price(days=30)
        
        self.assertIn('predicted_price', result)
        self.assertIn('confidence', result)
        self.assertIn('method', result)
    
    def test_data_quality_assessment(self):
        """Test data quality assessment."""
        quality = self.orchestrator._assess_data_quality()
        
        self.assertIn('data_points', quality)
        self.assertIn('date_range_days', quality)
        self.assertIn('data_completeness', quality)
        
        self.assertEqual(quality['data_points'], len(self.sample_data))
        self.assertGreaterEqual(quality['data_completeness'], 0)
        self.assertLessEqual(quality['data_completeness'], 100)
    
    def test_fallback_prediction(self):
        """Test fallback prediction when ensemble fails."""
        fallback = self.orchestrator._get_fallback_prediction()
        
        self.assertIn('predicted_price', fallback)
        self.assertIn('confidence', fallback)
        self.assertIn('market_signal', fallback)
        self.assertEqual(fallback['method'], 'Fallback')
        self.assertEqual(fallback['confidence'], 0.1)


class TestOrchestratorStandaloneFunctions(unittest.TestCase):
    """Test standalone functions."""
    
    @patch('stock_screener.prediction_models.prediction_orchestrator.PricePredictionOrchestrator')
    def test_predict_stock_price_function(self, mock_orchestrator_class):
        """Test predict_stock_price standalone function."""
        # Mock orchestrator instance
        mock_orchestrator = Mock()
        mock_orchestrator.predict_comprehensive.return_value = {
            'predicted_price': 1000.0,
            'confidence': 0.7,
            'market_signal': 'BUY'
        }
        mock_orchestrator_class.return_value = mock_orchestrator
        
        result = predict_stock_price('TEST.NS', days=30)
        
        mock_orchestrator_class.assert_called_once_with('TEST.NS')
        mock_orchestrator.predict_comprehensive.assert_called_once_with(30)
        
        self.assertEqual(result['predicted_price'], 1000.0)
        self.assertEqual(result['confidence'], 0.7)
        self.assertEqual(result['market_signal'], 'BUY')
    
    @patch('stock_screener.prediction_models.prediction_orchestrator.PricePredictionOrchestrator')
    def test_predict_stock_price_exception(self, mock_orchestrator_class):
        """Test exception handling in predict_stock_price."""
        mock_orchestrator_class.side_effect = Exception("Test error")
        
        result = predict_stock_price('TEST.NS')
        
        self.assertIn('error', result)
        self.assertIn('Price prediction failed', result['error'])
    
    @patch('stock_screener.prediction_models.prediction_orchestrator.PricePredictionOrchestrator')
    def test_compare_prediction_models_function(self, mock_orchestrator_class):
        """Test compare_prediction_models standalone function."""
        # Mock orchestrator instance
        mock_orchestrator = Mock()
        mock_orchestrator.get_model_comparison.return_value = {
            'symbol': 'TEST.NS',
            'current_price': 1000.0,
            'prediction_comparison': {
                'technical': {'predicted_price': 1050.0}
            }
        }
        mock_orchestrator_class.return_value = mock_orchestrator
        
        result = compare_prediction_models('TEST.NS', days=30)
        
        mock_orchestrator_class.assert_called_once_with('TEST.NS')
        mock_orchestrator.get_model_comparison.assert_called_once_with(30)
        
        self.assertEqual(result['symbol'], 'TEST.NS')
        self.assertEqual(result['current_price'], 1000.0)


class TestOrchestratorErrorHandling(unittest.TestCase):
    """Test error handling scenarios."""
    
    def test_invalid_data_handling(self):
        """Test behavior with invalid data."""
        empty_data = pd.DataFrame()
        
        with self.assertRaises(Exception):
            PricePredictionOrchestrator("TEST.NS", empty_data)
    
    @patch('yfinance.Ticker')
    def test_data_fetch_failure(self, mock_ticker):
        """Test handling of data fetch failure."""
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = pd.DataFrame()  # Empty data
        mock_ticker.return_value = mock_ticker_instance
        
        with self.assertRaises(ValueError):
            PricePredictionOrchestrator("INVALID.NS")
    
    def test_model_initialization_failure(self):
        """Test handling when some models fail to initialize."""
        # This test would require mocking individual model imports
        # For now, we test that the orchestrator handles partial failures gracefully
        
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        minimal_data = pd.DataFrame({
            'Open': [100] * 10,
            'High': [105] * 10,
            'Low': [95] * 10,
            'Close': [102] * 10,
            'Volume': [1000000] * 10
        }, index=dates)
        
        # Even with minimal data, orchestrator should initialize
        orchestrator = PricePredictionOrchestrator("TEST.NS", minimal_data)
        result = orchestrator.predict_comprehensive(target_days=5)
        
        # Should return some result even if some models fail
        self.assertIn('predicted_price', result)
        self.assertIn('confidence', result)


class TestOrchestratorIntegration(unittest.TestCase):
    """Integration tests for orchestrator."""
    
    def setUp(self):
        """Set up integration test data."""
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=200, freq='D')
        
        # Create realistic price data with trend
        base_price = 1000
        trend = np.cumsum(np.random.normal(0, 0.01, 200))  # Random walk trend
        prices = base_price * (1 + trend)
        
        self.realistic_data = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.005, 200)),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.01, 200))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, 200))),
            'Close': prices,
            'Volume': np.random.uniform(1000000, 10000000, 200)
        }, index=dates)
    
    def test_full_prediction_pipeline(self):
        """Test the complete prediction pipeline."""
        orchestrator = PricePredictionOrchestrator("INTEGRATION.NS", self.realistic_data)
        
        # Test comprehensive prediction
        result = orchestrator.predict_comprehensive(target_days=30)
        
        # Validate comprehensive result structure
        self.assertIn('predicted_price', result)
        self.assertIn('confidence', result)
        self.assertIn('market_signal', result)
        self.assertIn('individual_predictions', result)
        self.assertIn('prediction_statistics', result)
        self.assertIn('risk_assessment', result)
        
        # Test that we get predictions from multiple models
        individual_predictions = result['individual_predictions']
        working_models = [name for name, pred in individual_predictions.items() 
                         if 'error' not in pred]
        
        # Should have at least 3 working models
        self.assertGreaterEqual(len(working_models), 3)
        
        # Test model comparison
        comparison = orchestrator.get_model_comparison(days=30)
        self.assertIn('prediction_comparison', comparison)
        
        # Test individual model access
        tech_result = orchestrator.get_individual_model_prediction('technical')
        if 'error' not in tech_result:
            self.assertIn('predicted_price', tech_result)


if __name__ == '__main__':
    unittest.main()
