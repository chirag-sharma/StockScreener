"""
Advanced Price Prediction Service
=================================

This module provides comprehensive price prediction capabilities using:
1. Technical Analysis (Moving averages, support/resistance)
2. Fundamental Analysis (DCF, P/E based, Graham formula)
3. Machine Learning Models (Linear Regression, ARIMA)
4. AI-Enhanced Predictions (Multi-factor analysis)
5. Ensemble Methods (Combining multiple approaches)

Usage:
    from stock_screener.services.pricePrediction import PricePredictionService
    
    predictor = PricePredictionService('RELIANCE.NS')
    predictions = predictor.get_comprehensive_predictions()
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Optional imports for advanced features
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

logger = logging.getLogger(__name__)


class PricePredictionService:
    """
    Advanced price prediction service using multiple methodologies
    """
    
    def __init__(self, symbol: str, prediction_days: int = 30):
        """
        Initialize the price prediction service
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE.NS')
            prediction_days: Number of days to predict ahead
        """
        self.symbol = symbol
        self.prediction_days = prediction_days
        self.data = None
        self.current_price = None
        self.fundamental_data = None
        
        # Load data
        self._load_historical_data()
        self._load_fundamental_data()
    
    def _load_historical_data(self, period: str = "2y"):
        """Load historical price and volume data"""
        try:
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(period=period)
            if not self.data.empty:
                self.current_price = self.data['Close'].iloc[-1]
                logger.info(f"Loaded {len(self.data)} days of data for {self.symbol}")
            else:
                logger.error(f"No data available for {self.symbol}")
        except Exception as e:
            logger.error(f"Error loading data for {self.symbol}: {e}")
    
    def _load_fundamental_data(self):
        """Load fundamental data for valuation-based predictions"""
        try:
            ticker = yf.Ticker(self.symbol)
            self.fundamental_data = ticker.info
            logger.info(f"Loaded fundamental data for {self.symbol}")
        except Exception as e:
            logger.error(f"Error loading fundamental data for {self.symbol}: {e}")
            self.fundamental_data = {}
    
    def get_comprehensive_predictions(self) -> Dict:
        """
        Get comprehensive price predictions using all available methods
        
        Returns:
            Dict containing predictions from different methods and ensemble result
        """
        if self.data is None or self.data.empty:
            return {"error": "No data available for prediction"}
        
        results = {
            "symbol": self.symbol,
            "current_price": self.current_price,
            "prediction_date": (datetime.now() + timedelta(days=self.prediction_days)).strftime("%Y-%m-%d"),
            "prediction_days": self.prediction_days,
            "methods": {}
        }
        
        # 1. Technical Analysis Predictions
        results["methods"]["technical"] = self._technical_analysis_prediction()
        
        # 2. Fundamental Analysis Predictions
        results["methods"]["fundamental"] = self._fundamental_analysis_prediction()
        
        # 3. Machine Learning Predictions
        if HAS_SKLEARN:
            results["methods"]["machine_learning"] = self._machine_learning_prediction()
        
        # 4. Time Series Analysis
        if HAS_STATSMODELS:
            results["methods"]["time_series"] = self._time_series_prediction()
        
        # 5. Pattern Recognition
        results["methods"]["pattern"] = self._pattern_recognition_prediction()
        
        # 6. Volume Analysis
        results["methods"]["volume"] = self._volume_analysis_prediction()
        
        # 7. Ensemble Prediction (combines all methods)
        results["ensemble"] = self._ensemble_prediction(results["methods"])
        
        # 8. Risk Assessment
        results["risk_assessment"] = self._calculate_prediction_risk()
        
        return results
    
    def _technical_analysis_prediction(self) -> Dict:
        """Technical analysis-based price prediction"""
        try:
            data = self.data.copy()
            
            # Calculate technical indicators
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            data['SMA_200'] = data['Close'].rolling(window=200).mean()
            data['EMA_12'] = data['Close'].ewm(span=12).mean()
            data['EMA_26'] = data['Close'].ewm(span=26).mean()
            
            # Calculate MACD if talib available
            if HAS_TALIB:
                data['MACD'], data['MACD_signal'], data['MACD_hist'] = talib.MACD(data['Close'])
                data['RSI'] = talib.RSI(data['Close'])
                data['BB_upper'], data['BB_middle'], data['BB_lower'] = talib.BBANDS(data['Close'])
            
            # Support and Resistance levels
            recent_data = data.tail(50)
            resistance = recent_data['High'].max()
            support = recent_data['Low'].min()
            
            # Trend analysis
            sma_trend = "Bullish" if data['Close'].iloc[-1] > data['SMA_50'].iloc[-1] else "Bearish"
            momentum = (data['Close'].iloc[-1] - data['Close'].iloc[-5]) / data['Close'].iloc[-5] * 100
            
            # Simple technical prediction based on trend
            if sma_trend == "Bullish" and momentum > 2:
                price_change = 0.05  # 5% increase
            elif sma_trend == "Bearish" and momentum < -2:
                price_change = -0.05  # 5% decrease
            else:
                price_change = momentum / 100 * 0.5  # Conservative momentum-based
            
            predicted_price = self.current_price * (1 + price_change)
            
            return {
                "predicted_price": round(predicted_price, 2),
                "confidence": 0.6,
                "trend": sma_trend,
                "momentum": round(momentum, 2),
                "support_level": round(support, 2),
                "resistance_level": round(resistance, 2),
                "method": "Technical Analysis"
            }
            
        except Exception as e:
            logger.error(f"Technical analysis prediction failed: {e}")
            return {"error": "Technical analysis failed", "predicted_price": self.current_price}
    
    def _fundamental_analysis_prediction(self) -> Dict:
        """Fundamental analysis-based valuation"""
        try:
            if not self.fundamental_data:
                return {"error": "No fundamental data available"}
            
            predictions = []
            methods_used = []
            
            # 1. P/E based valuation
            pe_ratio = self.fundamental_data.get('trailingPE')
            eps = self.fundamental_data.get('trailingEps')
            if pe_ratio and eps:
                # Use sector average P/E (simplified to 18 for Indian market)
                fair_pe = 18
                pe_based_price = eps * fair_pe
                predictions.append(pe_based_price)
                methods_used.append("P/E Valuation")
            
            # 2. P/B based valuation
            pb_ratio = self.fundamental_data.get('priceToBook')
            book_value = self.fundamental_data.get('bookValue')
            if pb_ratio and book_value:
                # Use conservative P/B of 2 for fair value
                fair_pb = 2.0
                pb_based_price = book_value * fair_pb
                predictions.append(pb_based_price)
                methods_used.append("P/B Valuation")
            
            # 3. DCF approximation (simplified)
            free_cash_flow = self.fundamental_data.get('freeCashflow')
            shares_outstanding = self.fundamental_data.get('sharesOutstanding')
            if free_cash_flow and shares_outstanding:
                # Simplified DCF: FCF * 15 (conservative multiplier) / shares
                dcf_price = (free_cash_flow * 15) / shares_outstanding
                predictions.append(dcf_price)
                methods_used.append("DCF Approximation")
            
            # 4. Graham Formula
            growth_rate = self.fundamental_data.get('earningsQuarterlyGrowth', 0) * 100
            if eps and growth_rate:
                # Graham Formula: EPS * (8.5 + 2 * growth_rate)
                graham_price = eps * (8.5 + 2 * abs(growth_rate))
                predictions.append(graham_price)
                methods_used.append("Graham Formula")
            
            if predictions:
                avg_prediction = sum(predictions) / len(predictions)
                confidence = min(0.8, 0.4 + len(predictions) * 0.1)  # Higher confidence with more methods
                
                return {
                    "predicted_price": round(avg_prediction, 2),
                    "confidence": confidence,
                    "methods_used": methods_used,
                    "individual_predictions": [round(p, 2) for p in predictions],
                    "method": "Fundamental Analysis"
                }
            else:
                return {"error": "Insufficient fundamental data for valuation"}
                
        except Exception as e:
            logger.error(f"Fundamental analysis prediction failed: {e}")
            return {"error": "Fundamental analysis failed", "predicted_price": self.current_price}
    
    def _machine_learning_prediction(self) -> Dict:
        """Machine learning-based price prediction"""
        try:
            if not HAS_SKLEARN:
                return {"error": "Scikit-learn not available"}
            
            data = self.data.copy()
            
            # Feature engineering
            data['SMA_5'] = data['Close'].rolling(window=5).mean()
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
            data['Price_Change'] = data['Close'].pct_change()
            data['Volume_Change'] = data['Volume'].pct_change()
            data['High_Low_Ratio'] = data['High'] / data['Low']
            
            # Create features matrix
            features = ['Open', 'High', 'Low', 'Volume', 'SMA_5', 'SMA_20', 
                       'Volume_SMA', 'Price_Change', 'Volume_Change', 'High_Low_Ratio']
            
            # Remove NaN values
            data = data.dropna()
            
            if len(data) < 50:  # Need minimum data for ML
                return {"error": "Insufficient data for machine learning"}
            
            X = data[features].values
            y = data['Close'].values
            
            # Split data (use last 20% as validation)
            split_point = int(len(data) * 0.8)
            X_train, X_val = X[:split_point], X[split_point:]
            y_train, y_val = y[:split_point], y[split_point:]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Train models
            models = {
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42)
            }
            
            predictions = {}
            best_model = None
            best_score = float('inf')
            
            for name, model in models.items():
                model.fit(X_train_scaled, y_train)
                val_pred = model.predict(X_val_scaled)
                mse = mean_squared_error(y_val, val_pred)
                
                if mse < best_score:
                    best_score = mse
                    best_model = (name, model)
                
                predictions[name] = {
                    'mse': mse,
                    'mae': mean_absolute_error(y_val, val_pred)
                }
            
            # Make future prediction with best model
            last_features = scaler.transform([X[-1]])
            future_price = best_model[1].predict(last_features)[0]
            
            # Calculate confidence based on model performance
            confidence = max(0.3, min(0.8, 1 - (best_score ** 0.5) / self.current_price))
            
            return {
                "predicted_price": round(future_price, 2),
                "confidence": round(confidence, 2),
                "best_model": best_model[0],
                "model_performance": predictions,
                "method": "Machine Learning"
            }
            
        except Exception as e:
            logger.error(f"Machine learning prediction failed: {e}")
            return {"error": "Machine learning prediction failed", "predicted_price": self.current_price}
    
    def _time_series_prediction(self) -> Dict:
        """Time series analysis using ARIMA"""
        try:
            if not HAS_STATSMODELS:
                return {"error": "Statsmodels not available"}
            
            prices = self.data['Close'].dropna()
            
            if len(prices) < 100:
                return {"error": "Insufficient data for time series analysis"}
            
            # Simple ARIMA model (1,1,1)
            model = ARIMA(prices, order=(1, 1, 1))
            fitted_model = model.fit()
            
            # Forecast
            forecast = fitted_model.forecast(steps=self.prediction_days)
            predicted_price = forecast.iloc[-1] if hasattr(forecast, 'iloc') else forecast[-1]
            
            # Calculate confidence based on model AIC
            aic = fitted_model.aic
            confidence = max(0.3, min(0.7, 1000 / aic))
            
            return {
                "predicted_price": round(predicted_price, 2),
                "confidence": round(confidence, 2),
                "model_aic": round(aic, 2),
                "method": "ARIMA Time Series"
            }
            
        except Exception as e:
            logger.error(f"Time series prediction failed: {e}")
            return {"error": "Time series prediction failed", "predicted_price": self.current_price}
    
    def _pattern_recognition_prediction(self) -> Dict:
        """Pattern recognition-based prediction"""
        try:
            data = self.data.copy()
            
            # Recent price pattern analysis
            recent_prices = data['Close'].tail(20).values
            
            # Calculate trend strength
            returns = np.diff(recent_prices) / recent_prices[:-1]
            trend_strength = np.mean(returns)
            volatility = np.std(returns)
            
            # Simple pattern recognition
            if trend_strength > 0.01:  # Strong uptrend
                price_multiplier = 1 + (trend_strength * 2)
            elif trend_strength < -0.01:  # Strong downtrend
                price_multiplier = 1 + (trend_strength * 2)
            else:  # Sideways
                price_multiplier = 1 + (trend_strength * 0.5)
            
            predicted_price = self.current_price * price_multiplier
            
            # Confidence based on trend consistency
            confidence = max(0.3, min(0.7, 0.5 + abs(trend_strength) * 10))
            
            return {
                "predicted_price": round(predicted_price, 2),
                "confidence": round(confidence, 2),
                "trend_strength": round(trend_strength * 100, 2),
                "volatility": round(volatility * 100, 2),
                "method": "Pattern Recognition"
            }
            
        except Exception as e:
            logger.error(f"Pattern recognition prediction failed: {e}")
            return {"error": "Pattern recognition failed", "predicted_price": self.current_price}
    
    def _volume_analysis_prediction(self) -> Dict:
        """Volume-based price prediction"""
        try:
            data = self.data.copy()
            
            # Volume trend analysis
            data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
            data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
            data['Price_Volume_Trend'] = data['Close'].pct_change() * data['Volume_Ratio']
            
            recent_pvt = data['Price_Volume_Trend'].tail(10).mean()
            volume_trend = data['Volume_Ratio'].tail(5).mean()
            
            # Volume-based prediction
            if volume_trend > 1.5 and recent_pvt > 0:  # High volume with positive price change
                price_change = 0.03  # 3% increase
            elif volume_trend > 1.5 and recent_pvt < 0:  # High volume with negative price change
                price_change = -0.03  # 3% decrease
            else:
                price_change = recent_pvt * 0.1  # Conservative volume-based change
            
            predicted_price = self.current_price * (1 + price_change)
            confidence = max(0.3, min(0.6, volume_trend / 3))
            
            return {
                "predicted_price": round(predicted_price, 2),
                "confidence": round(confidence, 2),
                "volume_trend": round(volume_trend, 2),
                "price_volume_trend": round(recent_pvt * 100, 2),
                "method": "Volume Analysis"
            }
            
        except Exception as e:
            logger.error(f"Volume analysis prediction failed: {e}")
            return {"error": "Volume analysis failed", "predicted_price": self.current_price}
    
    def _ensemble_prediction(self, methods: Dict) -> Dict:
        """Combine predictions from all methods using weighted average"""
        try:
            valid_predictions = []
            total_confidence = 0
            method_weights = {
                "fundamental": 0.3,
                "machine_learning": 0.25,
                "technical": 0.2,
                "time_series": 0.15,
                "pattern": 0.05,
                "volume": 0.05
            }
            
            for method_name, result in methods.items():
                if isinstance(result, dict) and "predicted_price" in result and "error" not in result:
                    weight = method_weights.get(method_name, 0.1)
                    confidence = result.get("confidence", 0.5)
                    weighted_price = result["predicted_price"] * weight * confidence
                    valid_predictions.append(weighted_price)
                    total_confidence += weight * confidence
            
            if valid_predictions:
                # Weighted ensemble prediction
                ensemble_price = sum(valid_predictions) / total_confidence if total_confidence > 0 else sum(valid_predictions) / len(valid_predictions)
                
                # Calculate price range (±10% based on volatility)
                price_volatility = abs(ensemble_price - self.current_price) / self.current_price
                price_range_low = ensemble_price * (1 - price_volatility * 0.5)
                price_range_high = ensemble_price * (1 + price_volatility * 0.5)
                
                return {
                    "predicted_price": round(ensemble_price, 2),
                    "confidence": round(min(0.85, total_confidence), 2),
                    "price_range": {
                        "low": round(price_range_low, 2),
                        "high": round(price_range_high, 2)
                    },
                    "methods_used": len(valid_predictions),
                    "recommendation": self._get_recommendation(ensemble_price),
                    "method": "Ensemble (Multiple Methods)"
                }
            else:
                return {"error": "No valid predictions available for ensemble"}
                
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            return {"error": "Ensemble prediction failed", "predicted_price": self.current_price}
    
    def _calculate_prediction_risk(self) -> Dict:
        """Calculate risk metrics for predictions"""
        try:
            data = self.data.copy()
            
            # Calculate volatility (30-day)
            returns = data['Close'].pct_change().dropna()
            volatility_30d = returns.tail(30).std() * np.sqrt(252) * 100  # Annualized
            
            # Calculate maximum drawdown
            rolling_max = data['Close'].rolling(window=252).max()
            drawdown = (data['Close'] - rolling_max) / rolling_max
            max_drawdown = abs(drawdown.min()) * 100
            
            # Risk level assessment
            if volatility_30d > 50:
                risk_level = "Very High"
            elif volatility_30d > 30:
                risk_level = "High"
            elif volatility_30d > 20:
                risk_level = "Medium"
            else:
                risk_level = "Low"
            
            return {
                "volatility_30d": round(volatility_30d, 2),
                "max_drawdown": round(max_drawdown, 2),
                "risk_level": risk_level,
                "confidence_adjustment": max(0.1, 1 - volatility_30d / 100)
            }
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            return {"error": "Risk assessment failed"}
    
    def _get_recommendation(self, predicted_price: float) -> str:
        """Get buy/sell recommendation based on predicted price"""
        price_change = (predicted_price - self.current_price) / self.current_price * 100
        
        if price_change > 15:
            return "Strong Buy"
        elif price_change > 5:
            return "Buy"
        elif price_change > -5:
            return "Hold"
        elif price_change > -15:
            return "Sell"
        else:
            return "Strong Sell"
    
    def get_quick_prediction(self) -> Dict:
        """Get a quick prediction using fundamental and technical analysis only"""
        technical = self._technical_analysis_prediction()
        fundamental = self._fundamental_analysis_prediction()
        
        predictions = []
        confidences = []
        
        if "predicted_price" in technical:
            predictions.append(technical["predicted_price"])
            confidences.append(technical.get("confidence", 0.5))
        
        if "predicted_price" in fundamental:
            predictions.append(fundamental["predicted_price"])
            confidences.append(fundamental.get("confidence", 0.5))
        
        if predictions:
            # Weighted average
            weighted_price = sum(p * c for p, c in zip(predictions, confidences)) / sum(confidences)
            avg_confidence = sum(confidences) / len(confidences)
            
            return {
                "symbol": self.symbol,
                "current_price": self.current_price,
                "predicted_price": round(weighted_price, 2),
                "price_change_percent": round((weighted_price - self.current_price) / self.current_price * 100, 2),
                "confidence": round(avg_confidence, 2),
                "recommendation": self._get_recommendation(weighted_price),
                "method": "Quick Prediction (Technical + Fundamental)"
            }
        else:
            return {"error": "Unable to generate prediction"}


def get_batch_predictions(symbols: List[str], prediction_days: int = 30) -> Dict:
    """
    Get price predictions for multiple stocks
    
    Args:
        symbols: List of stock symbols
        prediction_days: Number of days to predict ahead
        
    Returns:
        Dictionary with predictions for each symbol
    """
    results = {}
    
    for symbol in symbols:
        try:
            logger.info(f"Getting predictions for {symbol}")
            predictor = PricePredictionService(symbol, prediction_days)
            results[symbol] = predictor.get_quick_prediction()
        except Exception as e:
            logger.error(f"Failed to get predictions for {symbol}: {e}")
            results[symbol] = {"error": f"Prediction failed: {str(e)}"}
    
    return results


# Example usage
if __name__ == "__main__":
    # Test the service
    predictor = PricePredictionService("RELIANCE.NS")
    predictions = predictor.get_comprehensive_predictions()
    
    print(f"Comprehensive Predictions for {predictions['symbol']}:")
    print(f"Current Price: ₹{predictions['current_price']}")
    print(f"Ensemble Prediction: ₹{predictions['ensemble']['predicted_price']}")
    print(f"Confidence: {predictions['ensemble']['confidence']*100:.1f}%")
    print(f"Recommendation: {predictions['ensemble']['recommendation']}")
