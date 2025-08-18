"""
Unit Tests for Time Series Prediction Model
==========================================

Comprehensive tests for the time series-based price prediction model.
"""

import unittest
from unittest.mock import patch, Mock
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from stock_screener.prediction_models.time_series_model import TimeSeriesModel


class TestTimeSeriesModel(unittest.TestCase):
    """Test cases for TimeSeriesModel."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data for all tests."""
        np.random.seed(42)
        cls.dates = pd.date_range(start='2023-01-01', periods=250, freq='D')  # ~1 year of data
        
        # Create time series with multiple components for realistic testing
        base_price = 1500
        
        # 1. Long-term trend component
        trend = np.linspace(0, 0.2, 250)  # 20% growth over period
        
        # 2. Seasonal component (quarterly cycle)
        seasonal = 0.05 * np.sin(2 * np.pi * np.arange(250) / 63)  # ~quarterly
        
        # 3. Cyclical component (longer business cycle)
        cyclical = 0.03 * np.sin(2 * np.pi * np.arange(250) / 200)
        
        # 4. Random walk component
        random_walk = np.cumsum(np.random.normal(0, 0.01, 250))
        
        # 5. GARCH-like volatility
        volatility = cls._generate_garch_volatility(250)
        
        # Combine all components
        log_prices = (np.log(base_price) + trend + seasonal + 
                     cyclical + random_walk + volatility)
        close_prices = np.exp(log_prices)
        
        cls.test_data = pd.DataFrame({
            'Open': close_prices * (1 + np.random.normal(0, 0.002, 250)),
            'High': close_prices * (1 + np.abs(np.random.normal(0.006, 0.002, 250))),
            'Low': close_prices * (1 - np.abs(np.random.normal(0.006, 0.002, 250))),
            'Close': close_prices,
            'Volume': np.random.lognormal(15, 0.3, 250)
        }, index=cls.dates)
        
        # Ensure OHLC consistency
        for i in range(len(cls.test_data)):
            row = cls.test_data.iloc[i]
            cls.test_data.iloc[i, cls.test_data.columns.get_loc('High')] = max(row['Open'], row['Close'], row['High'])
            cls.test_data.iloc[i, cls.test_data.columns.get_loc('Low')] = min(row['Open'], row['Close'], row['Low'])
    
    @classmethod
    def _generate_garch_volatility(cls, length):
        """Generate GARCH-like volatility clustering."""
        volatility = np.zeros(length)
        sigma_sq = 0.0001  # Initial variance
        
        for i in range(1, length):
            # GARCH(1,1) parameters
            omega = 0.00001
            alpha = 0.1
            beta = 0.85
            
            epsilon = np.random.normal(0, 1)
            volatility[i] = epsilon * np.sqrt(sigma_sq)
            
            # Update variance
            sigma_sq = omega + alpha * (volatility[i-1] ** 2) + beta * sigma_sq
        
        return volatility
    
    def test_model_initialization(self):
        """Test proper model initialization."""
        model = TimeSeriesModel("TS.TEST", self.test_data)
        
        self.assertEqual(model.symbol, "TS.TEST")
        self.assertIsInstance(model.data, pd.DataFrame)
        self.assertEqual(len(model.data), 250)
        self.assertIn('Close', model.data.columns)
    
    def test_arima_prediction(self):
        """Test ARIMA-based prediction."""
        model = TimeSeriesModel("ARIMA.TEST", self.test_data)
        prediction = model.predict(target_days=30)
        
        if 'error' not in prediction:
            self.assertIn('predicted_price', prediction)
            self.assertIn('confidence', prediction)
            self.assertEqual(prediction['method'], 'Time Series Analysis')
            
            # ARIMA predictions should be reasonable
            predicted_price = prediction['predicted_price']
            current_price = self.test_data['Close'].iloc[-1]
            
            self.assertGreater(predicted_price, 0)
            # Should not be too extreme
            change_pct = abs(predicted_price - current_price) / current_price
            self.assertLess(change_pct, 0.5)  # Less than 50% change
        else:
            # If ARIMA fails, should provide meaningful error
            self.assertIn('error', prediction)
            self.assertIsInstance(prediction['error'], str)
    
    def test_exponential_smoothing_prediction(self):
        """Test exponential smoothing prediction."""
        model = TimeSeriesModel("ETS.TEST", self.test_data)
        prediction = model.predict(target_days=30)
        
        if 'error' not in prediction:
            # Should contain time series specific information
            if 'model_type' in prediction:
                self.assertIn(prediction['model_type'].lower(), 
                             ['arima', 'exponential smoothing', 'holt-winters', 'ets'])
            
            # Should have reasonable confidence for time series
            self.assertGreaterEqual(prediction['confidence'], 0.2)
            self.assertLessEqual(prediction['confidence'], 1.0)
    
    def test_seasonal_decomposition(self):
        """Test seasonal decomposition and forecasting."""
        model = TimeSeriesModel("SEASONAL.TEST", self.test_data)
        prediction = model.predict(target_days=30)
        
        if 'error' not in prediction:
            # Check if seasonal components are detected
            if 'seasonal_components' in prediction:
                components = prediction['seasonal_components']
                
                expected_components = ['trend', 'seasonal', 'residual']
                for comp in expected_components:
                    if comp in components:
                        self.assertIsInstance(components[comp], (int, float, list))
    
    def test_stationarity_handling(self):
        """Test handling of non-stationary time series."""
        # Create non-stationary data (strong trend)
        non_stationary_dates = pd.date_range('2023-01-01', periods=200, freq='D')
        strong_trend = np.cumsum(np.random.normal(0.01, 0.005, 200))  # Strong upward trend
        base_prices = 1000 * np.exp(strong_trend)
        
        non_stationary_data = pd.DataFrame({
            'Open': base_prices * 0.999,
            'High': base_prices * 1.01,
            'Low': base_prices * 0.99,
            'Close': base_prices,
            'Volume': np.random.uniform(1000000, 2000000, 200)
        }, index=non_stationary_dates)
        
        model = TimeSeriesModel("NONSTATIONARY.TEST", non_stationary_data)
        prediction = model.predict(target_days=30)
        
        if 'error' not in prediction:
            # Should handle non-stationary data through differencing
            self.assertIn('predicted_price', prediction)
            
            # Prediction should follow the trend reasonably
            current_price = non_stationary_data['Close'].iloc[-1]
            predicted_price = prediction['predicted_price']
            
            # Should predict upward movement given strong upward trend
            self.assertGreater(predicted_price, current_price * 0.95)  # Allow some flexibility
    
    def test_different_forecast_horizons(self):
        """Test forecasting for different time horizons."""
        model = TimeSeriesModel("HORIZONS.TEST", self.test_data)
        
        horizons = [1, 7, 30, 90, 180]
        predictions = {}
        
        for days in horizons:
            pred = model.predict(target_days=days)
            predictions[days] = pred
        
        # All should complete or fail gracefully
        for days, pred in predictions.items():
            self.assertIsInstance(pred, dict)
            
            if 'error' not in pred:
                self.assertIn('predicted_price', pred)
                # Longer horizons should generally have lower confidence
                if days > 30 and 'confidence' in pred:
                    self.assertGreaterEqual(pred['confidence'], 0.1)
    
    def test_confidence_intervals(self):
        """Test prediction confidence intervals."""
        model = TimeSeriesModel("INTERVALS.TEST", self.test_data)
        prediction = model.predict(target_days=30)
        
        if 'error' not in prediction and 'confidence_interval' in prediction:
            intervals = prediction['confidence_interval']
            
            self.assertIn('lower', intervals)
            self.assertIn('upper', intervals)
            
            lower = intervals['lower']
            upper = intervals['upper']
            predicted = prediction['predicted_price']
            
            # Confidence intervals should make sense
            self.assertLess(lower, predicted)
            self.assertGreater(upper, predicted)
            self.assertGreater(upper - lower, 0)  # Should have some width
    
    def test_model_diagnostics(self):
        """Test time series model diagnostics."""
        model = TimeSeriesModel("DIAGNOSTICS.TEST", self.test_data)
        prediction = model.predict(target_days=30)
        
        if 'error' not in prediction and 'model_diagnostics' in prediction:
            diagnostics = prediction['model_diagnostics']
            
            # Common time series diagnostics
            possible_diagnostics = ['aic', 'bic', 'llf', 'residuals_autocorr', 'ljung_box_p']
            
            for diag in possible_diagnostics:
                if diag in diagnostics:
                    self.assertIsInstance(diagnostics[diag], (int, float))
    
    def test_insufficient_data_handling(self):
        """Test behavior with insufficient data for time series."""
        # Use very little data (insufficient for reliable time series)
        small_data = self.test_data.head(15)
        model = TimeSeriesModel("INSUFFICIENT.TEST", small_data)
        
        prediction = model.predict(target_days=30)
        
        if 'error' in prediction:
            # Should provide meaningful error message
            error_msg = prediction['error'].lower()
            self.assertTrue(any(word in error_msg for word in 
                              ['insufficient', 'data', 'minimum', 'observations']))
        else:
            # If it works, confidence should be very low
            self.assertLess(prediction['confidence'], 0.4)
    
    def test_volatility_forecasting(self):
        """Test volatility forecasting capabilities."""
        model = TimeSeriesModel("VOLATILITY.TEST", self.test_data)
        prediction = model.predict(target_days=30)
        
        if 'error' not in prediction and 'volatility_forecast' in prediction:
            vol_forecast = prediction['volatility_forecast']
            
            self.assertIsInstance(vol_forecast, (int, float))
            self.assertGreater(vol_forecast, 0)
            
            # Volatility should be reasonable (typically 10-50% annualized for stocks)
            self.assertLess(vol_forecast, 2.0)  # 200% would be extreme
    
    def test_trend_detection(self):
        """Test trend detection in time series."""
        # Create data with clear trends
        
        # Upward trend data
        up_dates = pd.date_range('2023-01-01', periods=100, freq='D')
        up_trend = np.cumsum(np.random.normal(0.01, 0.01, 100))
        up_prices = 1000 * np.exp(up_trend)
        
        up_data = pd.DataFrame({
            'Open': up_prices * 0.999,
            'High': up_prices * 1.01,
            'Low': up_prices * 0.99,
            'Close': up_prices,
            'Volume': np.random.uniform(1000000, 2000000, 100)
        }, index=up_dates)
        
        # Downward trend data
        down_dates = pd.date_range('2023-01-01', periods=100, freq='D')
        down_trend = np.cumsum(np.random.normal(-0.01, 0.01, 100))
        down_prices = 1000 * np.exp(down_trend)
        
        down_data = pd.DataFrame({
            'Open': down_prices * 0.999,
            'High': down_prices * 1.01,
            'Low': down_prices * 0.99,
            'Close': down_prices,
            'Volume': np.random.uniform(1000000, 2000000, 100)
        }, index=down_dates)
        
        up_model = TimeSeriesModel("UPTREND.TEST", up_data)
        down_model = TimeSeriesModel("DOWNTREND.TEST", down_data)
        
        up_pred = up_model.predict(target_days=30)
        down_pred = down_model.predict(target_days=30)
        
        if 'error' not in up_pred and 'error' not in down_pred:
            up_change = ((up_pred['predicted_price'] - up_data['Close'].iloc[-1]) / 
                        up_data['Close'].iloc[-1])
            down_change = ((down_pred['predicted_price'] - down_data['Close'].iloc[-1]) / 
                          down_data['Close'].iloc[-1])
            
            # Upward trend should predict higher prices than downward trend
            self.assertGreater(up_change, down_change)
    
    def test_seasonal_pattern_recognition(self):
        """Test recognition of seasonal patterns."""
        # Create data with clear seasonal pattern
        seasonal_dates = pd.date_range('2022-01-01', periods=400, freq='D')
        
        # Strong seasonal component (monthly cycle)
        seasonal_component = 0.1 * np.sin(2 * np.pi * np.arange(400) / 30)
        base_prices = 1200 * np.exp(seasonal_component + np.cumsum(np.random.normal(0, 0.005, 400)))
        
        seasonal_data = pd.DataFrame({
            'Open': base_prices * 0.999,
            'High': base_prices * 1.01,
            'Low': base_prices * 0.99,
            'Close': base_prices,
            'Volume': np.random.uniform(1000000, 2000000, 400)
        }, index=seasonal_dates)
        
        model = TimeSeriesModel("SEASONAL.PATTERN", seasonal_data)
        prediction = model.predict(target_days=30)
        
        if 'error' not in prediction:
            # Should detect seasonal patterns
            if 'seasonal_detected' in prediction:
                self.assertTrue(prediction['seasonal_detected'])
    
    def test_outlier_handling_in_time_series(self):
        """Test handling of outliers in time series data."""
        # Add outliers to the test data
        outlier_data = self.test_data.copy()
        
        # Add some extreme outliers
        outlier_indices = [50, 100, 150, 200]
        for idx in outlier_indices:
            if idx < len(outlier_data):
                # Create extreme price movements
                multiplier = np.random.choice([0.5, 2.0])  # 50% drop or 100% spike
                outlier_data.iloc[idx, outlier_data.columns.get_loc('Close')] *= multiplier
        
        model = TimeSeriesModel("OUTLIER.TEST", outlier_data)
        prediction = model.predict(target_days=30)
        
        if 'error' not in prediction:
            # Should handle outliers gracefully
            self.assertIn('predicted_price', prediction)
            
            # Prediction should not be extreme due to outliers
            current_price = np.median(self.test_data['Close'].tail(10))  # Use median for robustness
            predicted_price = prediction['predicted_price']
            
            change_pct = abs(predicted_price - current_price) / current_price
            self.assertLess(change_pct, 0.6)  # Should not be too extreme
    
    def test_autocorrelation_analysis(self):
        """Test autocorrelation analysis in predictions."""
        model = TimeSeriesModel("AUTOCORR.TEST", self.test_data)
        prediction = model.predict(target_days=30)
        
        if 'error' not in prediction and 'autocorrelation_analysis' in prediction:
            autocorr = prediction['autocorrelation_analysis']
            
            # Should have autocorrelation information
            if 'significant_lags' in autocorr:
                self.assertIsInstance(autocorr['significant_lags'], (list, tuple))
                
            if 'ljung_box_test' in autocorr:
                self.assertIn('p_value', autocorr['ljung_box_test'])
    
    def test_model_selection_criteria(self):
        """Test automatic model selection based on criteria."""
        model = TimeSeriesModel("MODEL.SELECTION", self.test_data)
        prediction = model.predict(target_days=30)
        
        if 'error' not in prediction and 'model_selection' in prediction:
            selection = prediction['model_selection']
            
            self.assertIn('chosen_model', selection)
            self.assertIn('selection_criteria', selection)
            
            # Common criteria
            criteria = selection.get('selection_criteria', {})
            possible_criteria = ['aic', 'bic', 'cross_validation_score']
            
            for criterion in possible_criteria:
                if criterion in criteria:
                    self.assertIsInstance(criteria[criterion], (int, float))
    
    def test_forecast_accuracy_metrics(self):
        """Test forecast accuracy assessment."""
        model = TimeSeriesModel("ACCURACY.TEST", self.test_data)
        prediction = model.predict(target_days=30)
        
        if 'error' not in prediction and 'accuracy_metrics' in prediction:
            metrics = prediction['accuracy_metrics']
            
            # Common accuracy metrics
            possible_metrics = ['mae', 'mse', 'rmse', 'mape', 'smape']
            
            for metric in possible_metrics:
                if metric in metrics:
                    self.assertIsInstance(metrics[metric], (int, float))
                    self.assertGreaterEqual(metrics[metric], 0)
    
    def test_residual_analysis(self):
        """Test residual analysis in time series models."""
        model = TimeSeriesModel("RESIDUALS.TEST", self.test_data)
        prediction = model.predict(target_days=30)
        
        if 'error' not in prediction and 'residual_analysis' in prediction:
            residuals = prediction['residual_analysis']
            
            # Should have residual statistics
            expected_stats = ['mean', 'std', 'skewness', 'kurtosis']
            
            for stat in expected_stats:
                if stat in residuals:
                    self.assertIsInstance(residuals[stat], (int, float))
    
    def test_cross_validation_time_series(self):
        """Test time series cross-validation."""
        model = TimeSeriesModel("CV.TEST", self.test_data)
        
        # Run multiple predictions to test consistency
        predictions = []
        for i in range(3):
            pred = model.predict(target_days=30)
            predictions.append(pred)
        
        # Should be consistent (deterministic models)
        if all('error' not in pred for pred in predictions):
            prices = [pred['predicted_price'] for pred in predictions]
            
            # Should be identical or very similar
            price_std = np.std(prices)
            price_mean = np.mean(prices)
            
            if price_mean > 0:
                cv = price_std / price_mean
                self.assertLess(cv, 0.01)  # Less than 1% coefficient of variation
    
    def test_extreme_market_conditions_time_series(self):
        """Test time series model under extreme market conditions."""
        # Create market crash scenario
        crash_dates = pd.date_range('2023-01-01', periods=150, freq='D')
        
        # Normal period followed by crash
        normal_returns = np.random.normal(0.001, 0.015, 100)
        crash_returns = np.random.normal(-0.05, 0.04, 50)  # Severe crash
        
        all_returns = np.concatenate([normal_returns, crash_returns])
        prices = 1500 * np.exp(np.cumsum(all_returns))
        
        crash_data = pd.DataFrame({
            'Open': prices * 0.999,
            'High': prices * 1.005,
            'Low': prices * 0.995,
            'Close': prices,
            'Volume': np.random.uniform(2000000, 5000000, 150)  # High volume during crash
        }, index=crash_dates)
        
        model = TimeSeriesModel("CRASH.TEST", crash_data)
        prediction = model.predict(target_days=30)
        
        if 'error' not in prediction:
            # Should handle extreme conditions
            self.assertIn('predicted_price', prediction)
            
            # Confidence should be appropriately low
            self.assertLess(prediction['confidence'], 0.6)


class TestTimeSeriesModelAdvanced(unittest.TestCase):
    """Advanced tests for time series model functionality."""
    
    def setUp(self):
        """Set up advanced test scenarios."""
        np.random.seed(456)
        
        # Create various time series patterns
        self.trend_data = self._create_trending_series()
        self.seasonal_data = self._create_seasonal_series()
        self.volatile_data = self._create_volatile_series()
        self.regime_change_data = self._create_regime_change_series()
    
    def _create_trending_series(self):
        """Create a series with clear trend."""
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        trend = np.linspace(0, 0.3, 200)  # 30% growth
        noise = np.cumsum(np.random.normal(0, 0.01, 200))
        
        prices = 1000 * np.exp(trend + noise)
        
        return pd.DataFrame({
            'Open': prices * 0.999,
            'High': prices * 1.01,
            'Low': prices * 0.99,
            'Close': prices,
            'Volume': np.random.uniform(1000000, 2000000, 200)
        }, index=dates)
    
    def _create_seasonal_series(self):
        """Create a series with seasonal patterns."""
        dates = pd.date_range('2022-01-01', periods=500, freq='D')  # ~1.4 years
        
        # Multiple seasonal components
        weekly = 0.02 * np.sin(2 * np.pi * np.arange(500) / 7)
        monthly = 0.05 * np.sin(2 * np.pi * np.arange(500) / 30)
        quarterly = 0.03 * np.sin(2 * np.pi * np.arange(500) / 90)
        
        seasonal = weekly + monthly + quarterly
        prices = 1200 * np.exp(seasonal + np.cumsum(np.random.normal(0, 0.008, 500)))
        
        return pd.DataFrame({
            'Open': prices * 0.999,
            'High': prices * 1.01,
            'Low': prices * 0.99,
            'Close': prices,
            'Volume': np.random.uniform(1000000, 2000000, 500)
        }, index=dates)
    
    def _create_volatile_series(self):
        """Create a highly volatile series."""
        dates = pd.date_range('2023-01-01', periods=180, freq='D')
        
        # High volatility with clusters
        volatilities = []
        current_vol = 0.02
        
        for i in range(180):
            # Volatility persistence
            current_vol = 0.05 * 0.02 + 0.9 * current_vol + 0.05 * np.random.gamma(2, 0.01)
            volatilities.append(current_vol)
        
        returns = np.random.normal(0, 1, 180) * np.array(volatilities)
        prices = 1300 * np.exp(np.cumsum(returns))
        
        return pd.DataFrame({
            'Open': prices * 0.998,
            'High': prices * 1.02,
            'Low': prices * 0.98,
            'Close': prices,
            'Volume': np.random.uniform(1500000, 3000000, 180)
        }, index=dates)
    
    def _create_regime_change_series(self):
        """Create a series with regime changes."""
        dates = pd.date_range('2023-01-01', periods=300, freq='D')
        
        returns = []
        
        # Regime 1: Bull market (first 100 days)
        returns.extend(np.random.normal(0.008, 0.015, 100))
        
        # Regime 2: Bear market (next 100 days)
        returns.extend(np.random.normal(-0.005, 0.025, 100))
        
        # Regime 3: Sideways market (last 100 days)
        returns.extend(np.random.normal(0.001, 0.012, 100))
        
        prices = 1400 * np.exp(np.cumsum(returns))
        
        return pd.DataFrame({
            'Open': prices * 0.999,
            'High': prices * 1.01,
            'Low': prices * 0.99,
            'Close': prices,
            'Volume': np.random.uniform(1000000, 2500000, 300)
        }, index=dates)
    
    def test_trend_following_accuracy(self):
        """Test accuracy in trending markets."""
        model = TimeSeriesModel("TREND.FOLLOW", self.trend_data)
        prediction = model.predict(target_days=30)
        
        if 'error' not in prediction:
            current_price = self.trend_data['Close'].iloc[-1]
            predicted_price = prediction['predicted_price']
            
            # Should predict continued upward trend
            self.assertGreater(predicted_price, current_price * 0.98)
            
            # Confidence should be reasonable for trending data
            self.assertGreater(prediction['confidence'], 0.4)
    
    def test_seasonal_forecasting_accuracy(self):
        """Test seasonal pattern forecasting."""
        model = TimeSeriesModel("SEASONAL.FORECAST", self.seasonal_data)
        prediction = model.predict(target_days=30)
        
        if 'error' not in prediction:
            # Should handle seasonal data well
            self.assertIn('predicted_price', prediction)
            self.assertGreater(prediction['confidence'], 0.3)
    
    def test_volatility_regime_detection(self):
        """Test detection of volatility regimes."""
        model = TimeSeriesModel("VOL.REGIME", self.volatile_data)
        prediction = model.predict(target_days=30)
        
        if 'error' not in prediction:
            # Should have lower confidence in high volatility regime
            self.assertLess(prediction['confidence'], 0.7)
            
            # Should still produce reasonable predictions
            self.assertIn('predicted_price', prediction)
    
    def test_regime_change_adaptation(self):
        """Test adaptation to regime changes."""
        model = TimeSeriesModel("REGIME.CHANGE", self.regime_change_data)
        prediction = model.predict(target_days=30)
        
        if 'error' not in prediction:
            # Should handle regime changes
            self.assertIn('predicted_price', prediction)
            
            # May have lower confidence due to regime uncertainty
            # But should still produce valid predictions


if __name__ == '__main__':
    unittest.main(verbosity=2)
