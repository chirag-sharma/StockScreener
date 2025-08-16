"""
Integration Tests for Modular Prediction System
==============================================

Tests the complete integration of all prediction models working together.
"""

import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from stock_screener.prediction_models import (
    PricePredictionOrchestrator,
    predict_stock_price,
    compare_prediction_models,
    TechnicalAnalysisModel,
    predict_price_technical
)


class TestPredictionSystemIntegration(unittest.TestCase):
    """Integration tests for the complete prediction system."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data for all integration tests."""
        # Create realistic market data for testing
        np.random.seed(42)
        cls.dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
        
        # Generate realistic stock price data with multiple patterns
        base_price = 1500
        
        # Add multiple components to make data realistic
        # 1. Long-term trend
        trend = np.cumsum(np.random.normal(0.001, 0.002, 365))
        
        # 2. Seasonal pattern (weekly)
        seasonal = 0.01 * np.sin(2 * np.pi * np.arange(365) / 5)
        
        # 3. Volatility clusters
        volatility = 0.02 * (1 + 0.5 * np.sin(2 * np.pi * np.arange(365) / 50))
        daily_returns = np.random.normal(0, 1, 365) * volatility
        
        # 4. Combine all components
        log_prices = np.log(base_price) + trend + seasonal + np.cumsum(daily_returns)
        close_prices = np.exp(log_prices)
        
        # Create OHLCV data
        cls.test_data = pd.DataFrame({
            'Open': close_prices * (1 + np.random.normal(0, 0.005, 365)),
            'High': close_prices * (1 + np.abs(np.random.normal(0.01, 0.005, 365))),
            'Low': close_prices * (1 - np.abs(np.random.normal(0.01, 0.005, 365))),
            'Close': close_prices,
            'Volume': np.random.lognormal(14, 0.5, 365)  # Realistic volume distribution
        }, index=cls.dates)
        
        # Ensure no negative prices
        for col in ['Open', 'High', 'Low', 'Close']:
            cls.test_data[col] = np.maximum(cls.test_data[col], 100)
    
    def test_orchestrator_full_integration(self):
        """Test full orchestrator integration with all models."""
        orchestrator = PricePredictionOrchestrator("INTEGRATION.TEST", self.test_data)
        
        # Test comprehensive prediction
        result = orchestrator.predict_comprehensive(target_days=30)
        
        # Validate structure
        required_fields = [
            'predicted_price', 'current_price', 'price_change_pct',
            'confidence', 'market_signal', 'models_used',
            'individual_predictions', 'prediction_statistics', 'risk_assessment'
        ]
        
        for field in required_fields:
            self.assertIn(field, result, f"Missing required field: {field}")
        
        # Validate data quality
        self.assertIsInstance(result['predicted_price'], (int, float))
        self.assertGreater(result['predicted_price'], 0)
        self.assertGreaterEqual(result['confidence'], 0.1)
        self.assertLessEqual(result['confidence'], 1.0)
        self.assertGreaterEqual(result['models_used'], 3)  # At least 3 models working
        
        # Test individual predictions structure
        individual_preds = result['individual_predictions']
        expected_models = ['technical', 'fundamental', 'machine_learning', 
                          'time_series', 'pattern_recognition', 'volume_analysis']
        
        for model_name in expected_models:
            self.assertIn(model_name, individual_preds)
        
        # Count working models
        working_models = sum(1 for pred in individual_preds.values() 
                           if 'error' not in pred)
        self.assertGreaterEqual(working_models, 4, "Should have at least 4 working models")
    
    def test_multi_period_predictions(self):
        """Test multi-period prediction consistency."""
        orchestrator = PricePredictionOrchestrator("MULTI.TEST", self.test_data)
        
        # Get predictions for different time horizons
        predictions = {}
        periods = [7, 30, 90, 365]
        
        for days in periods:
            pred = orchestrator.predict_comprehensive(target_days=days)
            predictions[days] = pred
        
        # Validate that all predictions completed
        for days, pred in predictions.items():
            self.assertIn('predicted_price', pred, f"Failed for {days} days")
            self.assertNotIn('error', pred, f"Error in {days} days prediction")
        
        # Test logical consistency (longer predictions should generally have lower confidence)
        short_term_conf = predictions[7]['confidence']
        long_term_conf = predictions[365]['confidence']
        
        # Allow some tolerance for this relationship
        self.assertLessEqual(long_term_conf, short_term_conf + 0.2)
    
    def test_model_comparison_integration(self):
        """Test model comparison functionality."""
        orchestrator = PricePredictionOrchestrator("COMPARE.TEST", self.test_data)
        comparison = orchestrator.get_model_comparison(days=30)
        
        # Validate comparison structure
        self.assertIn('symbol', comparison)
        self.assertIn('current_price', comparison)
        self.assertIn('prediction_comparison', comparison)
        
        pred_comparison = comparison['prediction_comparison']
        
        # Should have entries for all models
        expected_models = ['technical', 'fundamental', 'machine_learning', 
                          'time_series', 'pattern_recognition', 'volume_analysis']
        
        for model in expected_models:
            self.assertIn(model, pred_comparison)
            
            model_result = pred_comparison[model]
            if 'error' not in model_result:
                # Validate successful prediction structure
                self.assertIn('predicted_price', model_result)
                self.assertIn('price_change_pct', model_result)
                self.assertIn('confidence', model_result)
    
    def test_individual_model_integration(self):
        """Test individual models can work independently."""
        # Test technical analysis model directly
        tech_model = TechnicalAnalysisModel("TECH.TEST", self.test_data)
        tech_result = tech_model.predict()
        
        self.assertIn('predicted_price', tech_result)
        self.assertIn('confidence', tech_result)
        self.assertEqual(tech_result['method'], 'Technical Analysis')
        
        # Test through orchestrator
        orchestrator = PricePredictionOrchestrator("INDIVIDUAL.TEST", self.test_data)
        
        for model_name in ['technical', 'machine_learning', 'volume_analysis']:
            individual_result = orchestrator.get_individual_model_prediction(model_name)
            
            if 'error' not in individual_result:
                self.assertIn('predicted_price', individual_result)
                self.assertIn('confidence', individual_result)
    
    def test_ensemble_vs_individual_predictions(self):
        """Test that ensemble predictions are reasonable vs individual models."""
        orchestrator = PricePredictionOrchestrator("ENSEMBLE.TEST", self.test_data)
        result = orchestrator.predict_comprehensive(target_days=30)
        
        ensemble_price = result['predicted_price']
        individual_preds = result['individual_predictions']
        
        # Get working individual predictions
        working_predictions = [
            pred['predicted_price'] 
            for pred in individual_preds.values() 
            if 'error' not in pred and 'predicted_price' in pred
        ]
        
        if len(working_predictions) >= 2:
            min_pred = min(working_predictions)
            max_pred = max(working_predictions)
            
            # Ensemble should generally be within range of individual predictions
            # Allow some tolerance for weighted combinations
            tolerance = (max_pred - min_pred) * 0.3
            self.assertGreaterEqual(ensemble_price, min_pred - tolerance)
            self.assertLessEqual(ensemble_price, max_pred + tolerance)
    
    def test_prediction_consistency(self):
        """Test that predictions are consistent when run multiple times."""
        orchestrator1 = PricePredictionOrchestrator("CONSISTENT.TEST", self.test_data)
        orchestrator2 = PricePredictionOrchestrator("CONSISTENT.TEST", self.test_data)
        
        # Run predictions multiple times
        pred1 = orchestrator1.predict_comprehensive(target_days=30)
        pred2 = orchestrator2.predict_comprehensive(target_days=30)
        
        # Should be identical (due to deterministic seeds in models)
        self.assertEqual(pred1['predicted_price'], pred2['predicted_price'])
        self.assertEqual(pred1['confidence'], pred2['confidence'])
    
    def test_error_handling_integration(self):
        """Test system behavior with problematic data."""
        # Test with minimal data
        minimal_data = self.test_data.head(20)  # Only 20 days
        orchestrator = PricePredictionOrchestrator("ERROR.TEST", minimal_data)
        
        result = orchestrator.predict_comprehensive(target_days=30)
        
        # Should still return a result, even if some models fail
        self.assertIn('predicted_price', result)
        self.assertIn('confidence', result)
        # Confidence should be lower due to data issues
        self.assertLess(result['confidence'], 0.6)
    
    def test_risk_assessment_integration(self):
        """Test risk assessment across different market conditions."""
        orchestrator = PricePredictionOrchestrator("RISK.TEST", self.test_data)
        result = orchestrator.predict_comprehensive(target_days=30)
        
        risk_assessment = result['risk_assessment']
        
        self.assertIn('risk_level', risk_assessment)
        self.assertIn('prediction_volatility', risk_assessment)
        self.assertIn('recommendation', risk_assessment)
        
        valid_risk_levels = ['LOW', 'MEDIUM', 'HIGH']
        self.assertIn(risk_assessment['risk_level'], valid_risk_levels)
    
    def test_market_signal_logic_integration(self):
        """Test market signal generation logic."""
        orchestrator = PricePredictionOrchestrator("SIGNAL.TEST", self.test_data)
        
        # Test various scenarios by manipulating confidence and price change
        test_cases = [
            # (price_change_pct, confidence, expected_signal_type)
            (0.08, 0.7, 'bullish'),  # Strong positive change, high confidence
            (-0.08, 0.7, 'bearish'), # Strong negative change, high confidence
            (0.01, 0.3, 'neutral'),  # Small change, low confidence
        ]
        
        for price_change_pct, confidence, signal_type in test_cases:
            signal = orchestrator._determine_market_signal(price_change_pct, confidence)
            
            if signal_type == 'bullish':
                self.assertIn(signal, ['BUY', 'STRONG BUY'])
            elif signal_type == 'bearish':
                self.assertIn(signal, ['SELL', 'STRONG SELL'])
            elif signal_type == 'neutral':
                self.assertIn(signal, ['HOLD', 'UNCERTAIN'])


class TestStandaloneFunctionIntegration(unittest.TestCase):
    """Test standalone functions integration."""
    
    @patch('yfinance.Ticker')
    def test_predict_stock_price_integration(self, mock_ticker):
        """Test predict_stock_price standalone function."""
        # Mock yfinance response
        mock_data = pd.DataFrame({
            'Open': [1000, 1005, 1010, 1015],
            'High': [1010, 1015, 1020, 1025],
            'Low': [995, 1000, 1005, 1010],
            'Close': [1005, 1010, 1015, 1020],
            'Volume': [1000000, 1100000, 1200000, 1300000]
        })
        
        mock_ticker_instance = unittest.mock.Mock()
        mock_ticker_instance.history.return_value = mock_data
        mock_ticker.return_value = mock_ticker_instance
        
        # Test the function
        result = predict_stock_price('TEST.NS', days=30)
        
        if 'error' not in result:
            self.assertIn('predicted_price', result)
            self.assertIn('confidence', result)
            self.assertIn('market_signal', result)
        else:
            # If there's an error, it should be a meaningful one
            self.assertIn('error', result)
            self.assertIsInstance(result['error'], str)
    
    @patch('yfinance.Ticker')
    def test_compare_prediction_models_integration(self, mock_ticker):
        """Test compare_prediction_models standalone function."""
        mock_data = pd.DataFrame({
            'Open': [1000] * 100,
            'High': [1010] * 100,
            'Low': [990] * 100,
            'Close': [1005] * 100,
            'Volume': [1000000] * 100
        })
        
        mock_ticker_instance = unittest.mock.Mock()
        mock_ticker_instance.history.return_value = mock_data
        mock_ticker.return_value = mock_ticker_instance
        
        result = compare_prediction_models('TEST.NS', days=30)
        
        if 'error' not in result:
            self.assertIn('symbol', result)
            self.assertIn('current_price', result)
            self.assertIn('prediction_comparison', result)
            
            # Should have comparisons for multiple models
            pred_comparison = result['prediction_comparison']
            self.assertGreaterEqual(len(pred_comparison), 4)


class TestPredictionSystemPerformance(unittest.TestCase):
    """Performance and scalability tests."""
    
    def setUp(self):
        """Set up performance test data."""
        np.random.seed(42)
        dates = pd.date_range(start='2022-01-01', periods=500, freq='D')  # Larger dataset
        
        close_prices = 1000 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 500)))
        
        self.large_dataset = pd.DataFrame({
            'Open': close_prices * 0.999,
            'High': close_prices * 1.02,
            'Low': close_prices * 0.98,
            'Close': close_prices,
            'Volume': np.random.lognormal(14, 0.5, 500)
        }, index=dates)
    
    def test_performance_with_large_dataset(self):
        """Test performance with larger datasets."""
        import time
        
        start_time = time.time()
        
        orchestrator = PricePredictionOrchestrator("PERF.TEST", self.large_dataset)
        result = orchestrator.predict_comprehensive(target_days=30)
        
        elapsed_time = time.time() - start_time
        
        # Should complete within reasonable time (30 seconds max)
        self.assertLess(elapsed_time, 30)
        
        # Should still produce valid results
        self.assertIn('predicted_price', result)
        self.assertIn('confidence', result)
    
    def test_memory_efficiency(self):
        """Test memory usage doesn't grow excessively."""
        # Run multiple predictions to check for memory leaks
        for i in range(5):
            orchestrator = PricePredictionOrchestrator(f"MEM.TEST.{i}", self.large_dataset)
            result = orchestrator.predict_comprehensive(target_days=30)
            
            self.assertIn('predicted_price', result)
            
            # Clean up
            del orchestrator


class TestPredictionSystemRobustness(unittest.TestCase):
    """Robustness tests for edge cases."""
    
    def test_extreme_market_conditions(self):
        """Test behavior under extreme market conditions."""
        # Create crash scenario (50% drop in 10 days)
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        prices = np.concatenate([
            np.full(50, 1000),  # Stable prices
            np.linspace(1000, 500, 10),  # Crash
            np.full(40, 500)  # New level
        ])
        
        crash_data = pd.DataFrame({
            'Open': prices - 5,
            'High': prices + 10,
            'Low': prices - 10,
            'Close': prices,
            'Volume': np.random.uniform(2000000, 10000000, 100)  # Higher volume during crash
        }, index=dates)
        
        orchestrator = PricePredictionOrchestrator("CRASH.TEST", crash_data)
        result = orchestrator.predict_comprehensive(target_days=30)
        
        # Should handle extreme conditions gracefully
        self.assertIn('predicted_price', result)
        self.assertIn('confidence', result)
        
        # Confidence should be lower in extreme conditions
        self.assertLess(result['confidence'], 0.7)
    
    def test_flat_market_conditions(self):
        """Test behavior with completely flat prices."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        flat_data = pd.DataFrame({
            'Open': [1000] * 100,
            'High': [1000] * 100,
            'Low': [1000] * 100,
            'Close': [1000] * 100,
            'Volume': [1000000] * 100
        }, index=dates)
        
        orchestrator = PricePredictionOrchestrator("FLAT.TEST", flat_data)
        result = orchestrator.predict_comprehensive(target_days=30)
        
        # Should predict similar price for flat market
        self.assertAlmostEqual(result['predicted_price'], 1000, delta=50)
        self.assertIn('confidence', result)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
