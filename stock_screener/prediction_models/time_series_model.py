"""
Time Series Price Prediction Model
=================================

Uses statistical time series models to predict stock prices.
"""

from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from .base_model import BasePredictionModel

# Optional imports for time series analysis
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.api import ExponentialSmoothing
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    import scipy.stats as stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class TimeSeriesModel(BasePredictionModel):
    """
    Time series-based price prediction using:
    - ARIMA (AutoRegressive Integrated Moving Average)
    - Exponential Smoothing
    - Seasonal decomposition
    - Trend analysis
    """
    
    def __init__(self, symbol: str, historical_data: pd.DataFrame, **kwargs):
        """
        Initialize Time Series Model.
        
        Args:
            symbol: Stock symbol
            historical_data: Historical OHLCV data
            **kwargs: Additional parameters
                - prediction_days: Number of days to predict (default: 30)
                - arima_order: ARIMA order (p,d,q) (default: (1,1,1))
                - deterministic: Use fixed random seed (default: True)
        """
        super().__init__(symbol, historical_data, **kwargs)
        
        if not HAS_STATSMODELS:
            raise ImportError("statsmodels is required for time series models")
        
        self.prediction_days = kwargs.get('prediction_days', 30)
        self.arima_order = kwargs.get('arima_order', (1, 1, 1))
        self.deterministic = kwargs.get('deterministic', True)
        
        # Set random seed for reproducibility
        if self.deterministic:
            np.random.seed(42)
        
        # Prepare time series data
        self.price_series = self._prepare_time_series()
    
    def predict(self) -> Dict[str, Any]:
        """
        Generate time series-based price prediction.
        
        Returns:
            Dict with predicted_price, confidence, model_info, etc.
        """
        try:
            if self.price_series is None or len(self.price_series) < 100:
                return {
                    "predicted_price": self.get_current_price(),
                    "confidence": 0.1,
                    "method": "Time Series",
                    "error": "Insufficient data for time series analysis"
                }
            
            predictions = {}
            
            # 1. ARIMA prediction
            arima_result = self._arima_prediction()
            if arima_result and 'predicted_price' in arima_result:
                predictions['ARIMA'] = arima_result
            
            # 2. Exponential Smoothing prediction
            exp_smooth_result = self._exponential_smoothing_prediction()
            if exp_smooth_result and 'predicted_price' in exp_smooth_result:
                predictions['Exponential_Smoothing'] = exp_smooth_result
            
            # 3. Trend-based prediction
            trend_result = self._trend_based_prediction()
            if trend_result and 'predicted_price' in trend_result:
                predictions['Trend_Based'] = trend_result
            
            if not predictions:
                return {
                    "predicted_price": self.get_current_price(),
                    "confidence": 0.1,
                    "method": "Time Series",
                    "error": "All time series models failed"
                }
            
            # Ensemble prediction (weighted average)
            ensemble_price = self._calculate_ensemble_prediction(predictions)
            confidence = self.get_confidence()
            
            # Get best individual model
            best_model = self._get_best_model(predictions)
            
            return {
                "predicted_price": round(ensemble_price, 2),
                "confidence": round(confidence, 2),
                "best_individual_model": best_model,
                "individual_predictions": {
                    name: round(result['predicted_price'], 2) 
                    for name, result in predictions.items()
                },
                "method": "Time Series",
                "prediction_date": pd.Timestamp.now() + pd.Timedelta(days=self.prediction_days),
                "additional_metrics": {
                    "data_points": len(self.price_series),
                    "stationarity": self._check_stationarity(),
                    "trend_direction": self._determine_trend_direction(),
                    "seasonality_detected": self._detect_seasonality()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Time series prediction failed: {e}")
            return {
                "predicted_price": self.get_current_price(),
                "confidence": 0.1,
                "method": "Time Series",
                "error": f"Prediction failed: {e}"
            }
    
    def get_confidence(self) -> float:
        """
        Calculate confidence based on model fit and prediction horizon.
        
        Returns:
            Confidence score between 0.1 and 0.75
        """
        try:
            confidence_factors = []
            
            # Data quantity factor
            data_length = len(self.price_series)
            data_factor = min(1.0, data_length / 250)  # 1 year of trading days
            confidence_factors.append(data_factor * 0.3)
            
            # Stationarity factor (stationary data is better for time series)
            stationarity = self._check_stationarity()
            stationarity_factor = 0.8 if stationarity else 0.4
            confidence_factors.append(stationarity_factor * 0.2)
            
            # Volatility factor (lower volatility = higher confidence)
            returns = self.price_series.pct_change().dropna()
            volatility = returns.std()
            volatility_factor = max(0.2, min(1.0, 1 - volatility * 20))
            confidence_factors.append(volatility_factor * 0.2)
            
            # Prediction horizon factor (shorter predictions = higher confidence)
            horizon_factor = max(0.3, 1.0 - (self.prediction_days - 1) / 100)
            confidence_factors.append(horizon_factor * 0.3)
            
            overall_confidence = sum(confidence_factors)
            return max(0.1, min(0.75, overall_confidence))
            
        except Exception as e:
            self.logger.error(f"Error calculating TS confidence: {e}")
            return 0.1
    
    def _prepare_time_series(self) -> Optional[pd.Series]:
        """Prepare price series for time series analysis."""
        try:
            # Use closing prices
            prices = self.historical_data['Close'].copy()
            
            # Remove any NaN values
            prices = prices.dropna()
            
            # Ensure we have enough data
            if len(prices) < 50:
                self.logger.warning(f"Limited time series data: {len(prices)} points")
                return None
            
            return prices
            
        except Exception as e:
            self.logger.error(f"Error preparing time series: {e}")
            return None
    
    def _arima_prediction(self) -> Optional[Dict[str, Any]]:
        """ARIMA model prediction."""
        try:
            # Convert to simple numeric array
            price_values = self.price_series.values
            
            # Fit ARIMA model
            model = ARIMA(price_values, order=self.arima_order)
            fitted_model = model.fit()
            
            # Generate forecast
            forecast = fitted_model.forecast(steps=self.prediction_days)
            predicted_price = forecast[-1] if hasattr(forecast, '__getitem__') else float(forecast)
            
            # Calculate model confidence based on AIC
            aic = fitted_model.aic
            model_confidence = max(0.3, min(0.8, 1000 / aic)) if aic > 0 else 0.3
            
            # Get confidence intervals if available
            forecast_ci = None
            try:
                forecast_result = fitted_model.get_forecast(steps=self.prediction_days)
                forecast_ci = forecast_result.conf_int()
                
                # Handle both DataFrame and array cases
                if hasattr(forecast_ci, 'iloc'):
                    # pandas DataFrame - use iloc
                    ci_lower = forecast_ci.iloc[-1, 0]
                    ci_upper = forecast_ci.iloc[-1, 1]
                    ci_interval = forecast_ci.iloc[-1].tolist()
                elif hasattr(forecast_ci, '__getitem__') and len(forecast_ci.shape) > 1:
                    # numpy array with 2D shape
                    ci_lower = forecast_ci[-1, 0]
                    ci_upper = forecast_ci[-1, 1]
                    ci_interval = [ci_lower, ci_upper]
                else:
                    # 1D array or other format - skip confidence interval processing
                    ci_lower = ci_upper = None
                    ci_interval = None
                
                if ci_lower is not None and ci_upper is not None:
                    ci_width = (ci_upper - ci_lower) / predicted_price
                    # Lower confidence for wider intervals
                    model_confidence *= max(0.5, 1 - ci_width)
                else:
                    ci_interval = None
                    
            except Exception as e:
                self.logger.warning(f"Could not compute confidence intervals: {e}")
                ci_interval = None
            
            return {
                'predicted_price': predicted_price,
                'model_confidence': model_confidence,
                'aic': aic,
                'order': self.arima_order,
                'confidence_interval': ci_interval
            }
            
        except Exception as e:
            self.logger.error(f"ARIMA prediction error: {e}")
            return None
    
    def _exponential_smoothing_prediction(self) -> Optional[Dict[str, Any]]:
        """Exponential smoothing prediction."""
        try:
            # Simple exponential smoothing for now
            model = ExponentialSmoothing(self.price_series.values, trend=None, seasonal=None)
            fitted_model = model.fit()
            
            # Generate forecast
            forecast = fitted_model.forecast(steps=self.prediction_days)
            predicted_price = forecast[-1] if hasattr(forecast, '__getitem__') else float(forecast)
            
            # Simple confidence based on how well the model fits recent data
            fitted_values = fitted_model.fittedvalues
            recent_error = np.mean(np.abs(self.price_series.values[-10:] - fitted_values[-10:]))
            current_price = self.get_current_price()
            model_confidence = max(0.2, min(0.7, 1 - recent_error / current_price))
            
            return {
                'predicted_price': predicted_price,
                'model_confidence': model_confidence,
                'recent_fit_error': recent_error
            }
            
        except Exception as e:
            self.logger.error(f"Exponential smoothing error: {e}")
            return None
    
    def _trend_based_prediction(self) -> Optional[Dict[str, Any]]:
        """Simple trend-based prediction using linear extrapolation."""
        try:
            # Use last 30 days to determine trend
            recent_prices = self.price_series.tail(30).values
            x = np.arange(len(recent_prices))
            
            # Linear regression to find trend
            if HAS_SCIPY:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, recent_prices)
            else:
                # Simple slope calculation if scipy not available
                slope = (recent_prices[-1] - recent_prices[0]) / (len(recent_prices) - 1)
                intercept = recent_prices[0]
                r_value = 0.5  # Default correlation
            
            # Extrapolate trend
            future_x = len(recent_prices) + self.prediction_days - 1
            predicted_price = slope * future_x + intercept
            
            # Confidence based on trend strength (R-squared)
            model_confidence = max(0.2, min(0.6, abs(r_value) ** 2))
            
            return {
                'predicted_price': predicted_price,
                'model_confidence': model_confidence,
                'trend_slope': slope,
                'trend_strength': abs(r_value) if 'r_value' in locals() else 0.5
            }
            
        except Exception as e:
            self.logger.error(f"Trend-based prediction error: {e}")
            return None
    
    def _calculate_ensemble_prediction(self, predictions: Dict[str, Dict]) -> float:
        """Calculate weighted ensemble prediction."""
        try:
            weighted_sum = 0
            total_weight = 0
            
            # Define model weights
            weights = {
                'ARIMA': 0.5,
                'Exponential_Smoothing': 0.3,
                'Trend_Based': 0.2
            }
            
            for model_name, result in predictions.items():
                if 'predicted_price' in result:
                    weight = weights.get(model_name, 0.1)
                    # Adjust weight by model confidence
                    if 'model_confidence' in result:
                        weight *= result['model_confidence']
                    
                    weighted_sum += result['predicted_price'] * weight
                    total_weight += weight
            
            if total_weight > 0:
                return weighted_sum / total_weight
            else:
                # Fallback: simple average
                prices = [result['predicted_price'] for result in predictions.values() 
                         if 'predicted_price' in result]
                return np.mean(prices) if prices else self.get_current_price()
                
        except Exception as e:
            self.logger.error(f"Error calculating ensemble: {e}")
            return self.get_current_price()
    
    def _get_best_model(self, predictions: Dict[str, Dict]) -> str:
        """Get the name of the best performing individual model."""
        try:
            best_model = "Unknown"
            best_confidence = 0
            
            for model_name, result in predictions.items():
                if 'model_confidence' in result:
                    if result['model_confidence'] > best_confidence:
                        best_confidence = result['model_confidence']
                        best_model = model_name
            
            return best_model
            
        except Exception:
            return "ARIMA"  # Default
    
    def _check_stationarity(self) -> bool:
        """Check if the time series is stationary using Augmented Dickey-Fuller test."""
        try:
            if HAS_STATSMODELS and len(self.price_series) > 50:
                result = adfuller(self.price_series.dropna())
                # If p-value < 0.05, reject null hypothesis (series is stationary)
                return result[1] < 0.05
            return False
        except Exception:
            return False
    
    def _determine_trend_direction(self) -> str:
        """Determine overall trend direction."""
        try:
            recent_prices = self.price_series.tail(20)
            if len(recent_prices) < 2:
                return "Unknown"
            
            start_price = recent_prices.iloc[0]
            end_price = recent_prices.iloc[-1]
            change_percent = ((end_price - start_price) / start_price) * 100
            
            if change_percent > 2:
                return "Upward"
            elif change_percent < -2:
                return "Downward"
            else:
                return "Sideways"
                
        except Exception:
            return "Unknown"
    
    def _detect_seasonality(self) -> bool:
        """Simple seasonality detection."""
        try:
            if len(self.price_series) < 100:
                return False
            
            # Simple check: compare weekly patterns
            # This is a simplified approach
            returns = self.price_series.pct_change().dropna()
            
            # Check if there's any weekly pattern (every 5 trading days)
            if len(returns) >= 50:
                weekly_corr = []
                for i in range(1, 6):
                    if len(returns) > i * 5:
                        corr = np.corrcoef(returns[:-i*5], returns[i*5:])[0, 1]
                        if not np.isnan(corr):
                            weekly_corr.append(abs(corr))
                
                # If any weekly correlation is > 0.3, consider seasonal
                return any(corr > 0.3 for corr in weekly_corr) if weekly_corr else False
            
            return False
            
        except Exception:
            return False


# Standalone function for direct usage
def predict_price_timeseries(symbol: str, days: int = 30, arima_order: Tuple[int, int, int] = (1, 1, 1)) -> Dict[str, Any]:
    """
    Standalone function to predict price using time series analysis.
    
    Args:
        symbol: Stock symbol
        days: Prediction period in days
        arima_order: ARIMA model order (p, d, q)
        
    Returns:
        Prediction results dictionary
    """
    try:
        import yfinance as yf
        
        # Fetch data
        ticker = yf.Ticker(symbol)
        historical_data = ticker.history(period="1y")
        
        if historical_data.empty:
            return {"error": f"No data found for {symbol}"}
        
        # Create and run model
        model = TimeSeriesModel(symbol, historical_data, prediction_days=days, arima_order=arima_order)
        return model.predict()
        
    except Exception as e:
        return {"error": f"Time series prediction failed: {e}"}
