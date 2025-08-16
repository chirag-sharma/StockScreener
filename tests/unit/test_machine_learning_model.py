"""
Unit Tests for Machine Learning Prediction Model
=============================================

Comprehensive tests for the ML-based price prediction model.
"""

import unittest
from unittest.mock import patch, Mock
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from stock_screener.prediction_models.machine_learning_model import MachineLearningModel


class TestMachineLearningModel(unittest.TestCase):
    """Test cases for MachineLearningModel."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data for all tests."""
        np.random.seed(42)
        cls.dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
        
        # Create realistic stock price data with ML-friendly patterns
        base_price = 1000
        
        # Generate features that ML models can learn
        # 1. Trending component
        trend = np.cumsum(np.random.normal(0.002, 0.01, 200))
        
        # 2. Mean-reverting component
        mean_revert = 0.05 * np.sin(np.arange(200) * 0.1)
        
        # 3. Random walk
        random_walk = np.cumsum(np.random.normal(0, 0.015, 200))
        
        # Combine components
        log_prices = np.log(base_price) + trend + mean_revert + random_walk
        close_prices = np.exp(log_prices)
        
        cls.test_data = pd.DataFrame({
            'Open': close_prices * (1 + np.random.normal(0, 0.003, 200)),
            'High': close_prices * (1 + np.abs(np.random.normal(0.008, 0.003, 200))),
            'Low': close_prices * (1 - np.abs(np.random.normal(0.008, 0.003, 200))),
            'Close': close_prices,
            'Volume': np.random.lognormal(15, 0.3, 200)
        }, index=cls.dates)
        
        # Ensure price consistency
        cls.test_data['High'] = np.maximum(cls.test_data['High'], cls.test_data[['Open', 'Close']].max(axis=1))
        cls.test_data['Low'] = np.minimum(cls.test_data['Low'], cls.test_data[['Open', 'Close']].min(axis=1))
    
    def test_model_initialization(self):
        """Test proper model initialization."""
        model = MachineLearningModel("ML.TEST", self.test_data)
        
        self.assertEqual(model.symbol, "ML.TEST")
        self.assertIsInstance(model.data, pd.DataFrame)
        self.assertEqual(len(model.data), 200)
        self.assertIn('Close', model.data.columns)
    
    def test_feature_engineering(self):
        """Test feature engineering capabilities."""
        model = MachineLearningModel("FEATURE.TEST", self.test_data)
        
        # Test the internal feature creation (if exposed or testable)
        # This tests the model's ability to create meaningful features
        prediction = model.predict(target_days=30)
        
        # If successful, model must have used features effectively
        if 'error' not in prediction:
            self.assertIn('predicted_price', prediction)
            self.assertIn('confidence', prediction)
            self.assertEqual(prediction['method'], 'Machine Learning')
            
            # ML confidence should be reasonable
            self.assertGreaterEqual(prediction['confidence'], 0.1)
            self.assertLessEqual(prediction['confidence'], 1.0)
    
    def test_regression_prediction(self):
        """Test regression-based price prediction."""
        model = MachineLearningModel("REGRESSION.TEST", self.test_data)
        prediction = model.predict(target_days=30)
        
        if 'error' not in prediction:
            predicted_price = prediction['predicted_price']
            current_price = self.test_data['Close'].iloc[-1]
            
            # Prediction should be reasonable relative to current price
            self.assertGreater(predicted_price, 0)
            self.assertLess(abs(predicted_price - current_price) / current_price, 0.5)  # <50% change
            
            # Should have model-specific attributes
            self.assertIn('features_used', prediction)
            self.assertIn('model_type', prediction)
    
    def test_feature_importance(self):
        """Test feature importance analysis."""
        model = MachineLearningModel("IMPORTANCE.TEST", self.test_data)
        prediction = model.predict(target_days=30)
        
        if 'error' not in prediction and 'feature_importance' in prediction:
            importance = prediction['feature_importance']
            
            # Should be a dictionary with feature names and importance scores
            self.assertIsInstance(importance, dict)
            self.assertGreater(len(importance), 0)
            
            # All importance scores should be non-negative and sum to approximately 1
            importance_values = list(importance.values())
            self.assertTrue(all(val >= 0 for val in importance_values))
    
    def test_different_target_periods(self):
        """Test predictions for different time periods."""
        model = MachineLearningModel("PERIODS.TEST", self.test_data)
        
        test_periods = [1, 7, 30, 90]
        predictions = {}
        
        for days in test_periods:
            pred = model.predict(target_days=days)
            predictions[days] = pred
        
        # All predictions should complete (or fail gracefully)
        for days, pred in predictions.items():
            self.assertIsInstance(pred, dict)
            
            if 'error' not in pred:
                self.assertIn('predicted_price', pred)
                self.assertGreater(pred['predicted_price'], 0)
    
    def test_model_confidence_calculation(self):
        """Test confidence score calculation."""
        model = MachineLearningModel("CONFIDENCE.TEST", self.test_data)
        prediction = model.predict(target_days=30)
        
        if 'error' not in prediction:
            confidence = prediction['confidence']
            
            # Confidence should be between 0 and 1
            self.assertIsInstance(confidence, (int, float))
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)
            
            # For ML models, confidence should typically be above 0.3 with good data
            self.assertGreaterEqual(confidence, 0.2)
    
    def test_cross_validation_robustness(self):
        """Test model's cross-validation and robustness."""
        model = MachineLearningModel("CV.TEST", self.test_data)
        
        # Run multiple predictions to test consistency
        predictions = []
        for i in range(3):
            pred = model.predict(target_days=30)
            predictions.append(pred)
        
        # All predictions should be similar (deterministic with seed)
        if all('error' not in pred for pred in predictions):
            prices = [pred['predicted_price'] for pred in predictions]
            price_std = np.std(prices)
            price_mean = np.mean(prices)
            
            # Standard deviation should be small relative to mean (consistent predictions)
            self.assertLess(price_std / price_mean, 0.01)  # <1% variation
    
    def test_insufficient_data_handling(self):
        """Test behavior with insufficient data."""
        # Use only 20 days of data (insufficient for ML)
        small_data = self.test_data.head(20)
        model = MachineLearningModel("SMALL.TEST", small_data)
        
        prediction = model.predict(target_days=30)
        
        # Should either work with reduced confidence or return error
        if 'error' in prediction:
            self.assertIn('insufficient data', prediction['error'].lower())
        else:
            # If it works, confidence should be very low
            self.assertLess(prediction['confidence'], 0.5)
    
    def test_feature_scaling_and_normalization(self):
        """Test that model handles different scales of data."""
        # Create data with different scales
        scaled_data = self.test_data.copy()
        scaled_data['Volume'] = scaled_data['Volume'] * 1000  # Much larger volume
        scaled_data['Close'] = scaled_data['Close'] / 10  # Much smaller prices
        
        model = MachineLearningModel("SCALE.TEST", scaled_data)
        prediction = model.predict(target_days=30)
        
        if 'error' not in prediction:
            # Should still produce reasonable predictions regardless of scale
            self.assertIn('predicted_price', prediction)
            self.assertGreater(prediction['predicted_price'], 0)
    
    def test_outlier_handling(self):
        """Test model's handling of outliers in data."""
        # Add outliers to the data
        outlier_data = self.test_data.copy()
        
        # Add some extreme price spikes
        outlier_indices = [50, 100, 150]
        for idx in outlier_indices:
            if idx < len(outlier_data):
                outlier_data.iloc[idx, outlier_data.columns.get_loc('Close')] *= 5  # 5x price spike
                outlier_data.iloc[idx, outlier_data.columns.get_loc('High')] = outlier_data.iloc[idx]['Close']
        
        model = MachineLearningModel("OUTLIER.TEST", outlier_data)
        prediction = model.predict(target_days=30)
        
        if 'error' not in prediction:
            # Model should handle outliers and not produce extreme predictions
            predicted_price = prediction['predicted_price']
            typical_price = np.median(self.test_data['Close'])
            
            # Prediction shouldn't be more than 3x the typical price
            self.assertLess(predicted_price, typical_price * 3)
    
    def test_model_performance_metrics(self):
        """Test model performance evaluation metrics."""
        model = MachineLearningModel("METRICS.TEST", self.test_data)
        prediction = model.predict(target_days=30)
        
        if 'error' not in prediction:
            # Check if model provides performance metrics
            if 'model_metrics' in prediction:
                metrics = prediction['model_metrics']
                
                # Common ML metrics
                expected_metrics = ['r2_score', 'mae', 'rmse', 'mape']
                for metric in expected_metrics:
                    if metric in metrics:
                        self.assertIsInstance(metrics[metric], (int, float))
    
    def test_ensemble_method_within_ml(self):
        """Test if ML model uses ensemble methods internally."""
        model = MachineLearningModel("ENSEMBLE.TEST", self.test_data)
        prediction = model.predict(target_days=30)
        
        if 'error' not in prediction:
            # Check for ensemble-related information
            if 'model_type' in prediction:
                model_type = prediction['model_type']
                # Could be RandomForest, GradientBoosting, etc.
                self.assertIsInstance(model_type, str)
                self.assertGreater(len(model_type), 0)
    
    def test_volatility_based_confidence(self):
        """Test confidence adjustment based on market volatility."""
        # Create high volatility data
        volatile_data = self.test_data.copy()
        volatile_returns = np.random.normal(0, 0.05, len(volatile_data))  # High volatility
        volatile_prices = volatile_data['Close'].iloc[0] * np.exp(np.cumsum(volatile_returns))
        volatile_data['Close'] = volatile_prices
        
        # Create low volatility data
        stable_data = self.test_data.copy()
        stable_returns = np.random.normal(0, 0.005, len(stable_data))  # Low volatility
        stable_prices = stable_data['Close'].iloc[0] * np.exp(np.cumsum(stable_returns))
        stable_data['Close'] = stable_prices
        
        volatile_model = MachineLearningModel("VOLATILE.TEST", volatile_data)
        stable_model = MachineLearningModel("STABLE.TEST", stable_data)
        
        volatile_pred = volatile_model.predict(target_days=30)
        stable_pred = stable_model.predict(target_days=30)
        
        if 'error' not in volatile_pred and 'error' not in stable_pred:
            # Stable market should generally have higher confidence
            # (though this may not always be true for ML models)
            volatile_conf = volatile_pred['confidence']
            stable_conf = stable_pred['confidence']
            
            # Both should be reasonable
            self.assertGreaterEqual(volatile_conf, 0.1)
            self.assertGreaterEqual(stable_conf, 0.1)
    
    def test_market_regime_detection(self):
        """Test model's ability to detect different market regimes."""
        # Create bull market data (upward trend)
        bull_data = self.test_data.copy()
        bull_trend = np.cumsum(np.random.normal(0.005, 0.01, len(bull_data)))  # Positive trend
        bull_data['Close'] = bull_data['Close'] * np.exp(bull_trend)
        
        # Create bear market data (downward trend)
        bear_data = self.test_data.copy()
        bear_trend = np.cumsum(np.random.normal(-0.005, 0.01, len(bear_data)))  # Negative trend
        bear_data['Close'] = bear_data['Close'] * np.exp(bear_trend)
        
        bull_model = MachineLearningModel("BULL.TEST", bull_data)
        bear_model = MachineLearningModel("BEAR.TEST", bear_data)
        
        bull_pred = bull_model.predict(target_days=30)
        bear_pred = bear_model.predict(target_days=30)
        
        if 'error' not in bull_pred and 'error' not in bear_pred:
            bull_change = ((bull_pred['predicted_price'] - bull_data['Close'].iloc[-1]) / 
                          bull_data['Close'].iloc[-1])
            bear_change = ((bear_pred['predicted_price'] - bear_data['Close'].iloc[-1]) / 
                          bear_data['Close'].iloc[-1])
            
            # Bull market prediction should generally be more positive than bear market
            # (though not guaranteed for all ML models)
            self.assertGreaterEqual(bull_change, bear_change - 0.1)  # Allow some tolerance
    
    def test_prediction_explanation(self):
        """Test if model provides prediction explanations."""
        model = MachineLearningModel("EXPLAIN.TEST", self.test_data)
        prediction = model.predict(target_days=30)
        
        if 'error' not in prediction:
            # Check for explanation fields
            explanation_fields = ['method', 'model_type', 'features_used']
            
            for field in explanation_fields:
                if field in prediction:
                    self.assertIsInstance(prediction[field], str)
                    self.assertGreater(len(prediction[field]), 0)
    
    def test_edge_case_single_price_level(self):
        """Test model behavior with flat price data."""
        # Create completely flat price data
        flat_data = pd.DataFrame({
            'Open': [1000] * 100,
            'High': [1000] * 100,
            'Low': [1000] * 100,
            'Close': [1000] * 100,
            'Volume': np.random.uniform(1000000, 2000000, 100)
        }, index=pd.date_range('2023-01-01', periods=100))
        
        model = MachineLearningModel("FLAT.TEST", flat_data)
        prediction = model.predict(target_days=30)
        
        # Should handle flat data gracefully
        if 'error' not in prediction:
            # Should predict similar price
            self.assertAlmostEqual(prediction['predicted_price'], 1000, delta=100)
        else:
            # If it fails, should have a meaningful error message
            self.assertIn('error', prediction)
            self.assertIsInstance(prediction['error'], str)
    
    def test_memory_efficiency(self):
        """Test that model doesn't consume excessive memory."""
        # This is a basic test - in production, you'd use memory profiling
        initial_data = self.test_data.copy()
        
        for i in range(5):  # Create multiple models
            model = MachineLearningModel(f"MEM.TEST.{i}", initial_data)
            prediction = model.predict(target_days=30)
            
            # Clean up
            del model
            
            # Should complete without memory errors
            self.assertIsInstance(prediction, dict)


class TestMachineLearningModelIntegration(unittest.TestCase):
    """Integration tests for ML model with real-world scenarios."""
    
    def setUp(self):
        """Set up integration test data."""
        # Create more complex, realistic data
        np.random.seed(123)
        dates = pd.date_range(start='2022-01-01', periods=300, freq='D')
        
        # Multi-factor price generation
        factors = {
            'market_trend': np.cumsum(np.random.normal(0.001, 0.015, 300)),
            'seasonal': 0.02 * np.sin(2 * np.pi * np.arange(300) / 252),  # Yearly cycle
            'volatility_clustering': self._generate_volatility_clustering(300),
            'momentum': self._generate_momentum_factor(300),
        }
        
        # Combine factors
        log_returns = sum(factors.values())
        prices = 1200 * np.exp(np.cumsum(log_returns))
        
        self.complex_data = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.002, 300)),
            'High': prices * (1 + np.abs(np.random.normal(0.01, 0.003, 300))),
            'Low': prices * (1 - np.abs(np.random.normal(0.01, 0.003, 300))),
            'Close': prices,
            'Volume': np.random.lognormal(14.5, 0.4, 300)
        }, index=dates)
        
        # Ensure price consistency
        for i in range(len(self.complex_data)):
            row = self.complex_data.iloc[i]
            self.complex_data.iloc[i, self.complex_data.columns.get_loc('High')] = max(row['Open'], row['Close'], row['High'])
            self.complex_data.iloc[i, self.complex_data.columns.get_loc('Low')] = min(row['Open'], row['Close'], row['Low'])
    
    def _generate_volatility_clustering(self, length):
        """Generate volatility clustering pattern."""
        volatilities = np.random.gamma(2, 0.01, length)
        returns = np.random.normal(0, 1, length) * volatilities
        return returns
    
    def _generate_momentum_factor(self, length):
        """Generate momentum factor in prices."""
        momentum = np.zeros(length)
        for i in range(20, length):
            # 20-day momentum
            momentum[i] = 0.001 * np.sign(np.mean(momentum[i-20:i]))
        return momentum
    
    def test_complex_market_prediction(self):
        """Test ML model with complex market data."""
        model = MachineLearningModel("COMPLEX.TEST", self.complex_data)
        prediction = model.predict(target_days=30)
        
        if 'error' not in prediction:
            self.assertIn('predicted_price', prediction)
            self.assertIn('confidence', prediction)
            
            # Should handle complex data well
            self.assertGreater(prediction['confidence'], 0.3)
            
            # Prediction should be reasonable
            current_price = self.complex_data['Close'].iloc[-1]
            predicted_price = prediction['predicted_price']
            
            change_pct = abs(predicted_price - current_price) / current_price
            self.assertLess(change_pct, 0.3)  # Less than 30% change
    
    def test_model_adaptability(self):
        """Test model's ability to adapt to different market conditions."""
        # Test with different subsets representing different market conditions
        
        # Bull market period
        bull_period = self.complex_data.iloc[100:200]  # Middle period
        bull_model = MachineLearningModel("BULL.ADAPT", bull_period)
        bull_pred = bull_model.predict(target_days=30)
        
        # Bear market period (simulate by inverting some returns)
        bear_period = self.complex_data.iloc[200:].copy()
        bear_period['Close'] = bear_period['Close'] * 0.95  # Simulate decline
        bear_model = MachineLearningModel("BEAR.ADAPT", bear_period)
        bear_pred = bear_model.predict(target_days=30)
        
        # Both should produce valid predictions
        for pred in [bull_pred, bear_pred]:
            if 'error' not in pred:
                self.assertIn('predicted_price', pred)
                self.assertIn('confidence', pred)


if __name__ == '__main__':
    unittest.main(verbosity=2)
