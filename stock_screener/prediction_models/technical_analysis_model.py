"""
Technical Analysis Price Prediction Model
========================================

Uses technical indicators, trend analysis, and momentum to predict stock prices.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np
from .base_model import BasePredictionModel

# Optional imports
try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False


class TechnicalAnalysisModel(BasePredictionModel):
    """
    Technical analysis-based price prediction using:
    - Moving averages (SMA, EMA)
    - MACD, RSI, Bollinger Bands
    - Support/Resistance levels
    - Momentum and trend analysis
    """
    
    def __init__(self, symbol: str, historical_data: pd.DataFrame, **kwargs):
        """
        Initialize Technical Analysis Model.
        
        Args:
            symbol: Stock symbol
            historical_data: Historical OHLCV data
            **kwargs: Additional parameters
                - prediction_days: Number of days to predict (default: 30)
                - lookback_period: Days to analyze (default: 50)
        """
        super().__init__(symbol, historical_data, **kwargs)
        self.prediction_days = kwargs.get('prediction_days', 30)
        self.lookback_period = kwargs.get('lookback_period', 50)
        
        # Add technical indicators to data
        self.data_with_indicators = self.add_technical_indicators()
    
    def predict(self) -> Dict[str, Any]:
        """
        Generate technical analysis-based price prediction.
        
        Returns:
            Dict with predicted_price, confidence, trend, momentum, etc.
        """
        try:
            data = self.data_with_indicators.copy()
            current_price = self.get_current_price()
            
            # Calculate additional technical indicators
            data = self._add_advanced_indicators(data)
            
            # Support and Resistance levels
            sr_levels = self.get_support_resistance_levels()
            resistance = sr_levels['resistance']
            support = sr_levels['support']
            
            # Trend analysis
            sma_trend = self._determine_trend(data)
            
            # Momentum calculation
            momentum = self._calculate_momentum(data)
            
            # Volatility analysis
            recent_volatility = self._calculate_volatility(data)
            
            # Technical prediction based on momentum and volatility
            predicted_price = self._calculate_technical_prediction(
                current_price, sma_trend, momentum, recent_volatility
            )
            
            # Calculate confidence
            confidence = self.get_confidence()
            
            return {
                "predicted_price": round(predicted_price, 2),
                "confidence": round(confidence, 2),
                "trend": sma_trend,
                "momentum": round(momentum, 2),
                "volatility": round(recent_volatility, 2),
                "support_level": round(support, 2),
                "resistance_level": round(resistance, 2),
                "method": "Technical Analysis",
                "prediction_date": pd.Timestamp.now() + pd.Timedelta(days=self.prediction_days),
                "additional_metrics": {
                    "rsi": self._get_latest_rsi(data),
                    "macd_signal": self._get_macd_signal(data),
                    "bollinger_position": self._get_bollinger_position(data)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Technical analysis prediction failed: {e}")
            return {
                "predicted_price": self.get_current_price(),
                "confidence": 0.1,
                "method": "Technical Analysis",
                "error": f"Prediction failed: {e}"
            }
    
    def get_confidence(self) -> float:
        """
        Calculate confidence based on data quality and trend consistency.
        
        Returns:
            Confidence score between 0.1 and 0.9
        """
        try:
            data = self.data_with_indicators
            
            # Data quality factors
            data_length = len(data)
            data_confidence = min(1.0, data_length / 100)  # More data = higher confidence
            
            # Trend consistency (lower volatility = higher confidence)
            recent_volatility = self._calculate_volatility(data)
            trend_consistency = 1 - min(1.0, recent_volatility / 100)
            
            # Technical indicator alignment
            indicator_alignment = self._calculate_indicator_alignment(data)
            
            # Overall confidence weighted combination
            overall_confidence = (
                trend_consistency * 0.4 + 
                data_confidence * 0.3 + 
                indicator_alignment * 0.3
            )
            
            return max(0.1, min(0.9, overall_confidence))
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return 0.1
    
    def _add_advanced_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add advanced technical indicators using TA-Lib if available."""
        try:
            if HAS_TALIB:
                data['MACD'], data['MACD_signal'], data['MACD_hist'] = talib.MACD(data['Close'])
                data['RSI'] = talib.RSI(data['Close'])
                data['BB_upper'], data['BB_middle'], data['BB_lower'] = talib.BBANDS(data['Close'])
                data['ADX'] = talib.ADX(data['High'], data['Low'], data['Close'])
                data['CCI'] = talib.CCI(data['High'], data['Low'], data['Close'])
            else:
                # Fallback manual calculations if TA-Lib not available
                self.logger.warning("TA-Lib not available, using basic indicators")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error adding advanced indicators: {e}")
            return data
    
    def _determine_trend(self, data: pd.DataFrame) -> str:
        """Determine overall trend based on multiple indicators."""
        try:
            current_price = data['Close'].iloc[-1]
            sma_20 = data['SMA_20'].iloc[-1]
            sma_50 = data['SMA_50'].iloc[-1]
            ema_12 = data['EMA_12'].iloc[-1]
            ema_26 = data['EMA_26'].iloc[-1]
            
            bullish_signals = 0
            total_signals = 0
            
            # SMA signals
            if pd.notna(sma_20):
                total_signals += 1
                if current_price > sma_20:
                    bullish_signals += 1
            
            if pd.notna(sma_50):
                total_signals += 1
                if current_price > sma_50:
                    bullish_signals += 1
            
            # EMA signals
            if pd.notna(ema_12) and pd.notna(ema_26):
                total_signals += 1
                if ema_12 > ema_26:
                    bullish_signals += 1
            
            # MACD signal
            if 'MACD' in data.columns and 'MACD_signal' in data.columns:
                macd = data['MACD'].iloc[-1]
                macd_signal = data['MACD_signal'].iloc[-1]
                if pd.notna(macd) and pd.notna(macd_signal):
                    total_signals += 1
                    if macd > macd_signal:
                        bullish_signals += 1
            
            if total_signals == 0:
                return "Neutral"
            
            bullish_ratio = bullish_signals / total_signals
            
            if bullish_ratio >= 0.6:
                return "Bullish"
            elif bullish_ratio <= 0.4:
                return "Bearish"
            else:
                return "Neutral"
                
        except Exception as e:
            self.logger.error(f"Error determining trend: {e}")
            return "Neutral"
    
    def _calculate_momentum(self, data: pd.DataFrame) -> float:
        """Calculate price momentum over recent period."""
        try:
            if len(data) < 5:
                return 0.0
            
            close_5_days_ago = data['Close'].iloc[-5]
            current_close = data['Close'].iloc[-1]
            
            if close_5_days_ago > 0:
                return ((current_close - close_5_days_ago) / close_5_days_ago) * 100
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error calculating momentum: {e}")
            return 0.0
    
    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calculate recent volatility as percentage."""
        try:
            return data['Close'].pct_change().tail(20).std() * 100
        except Exception as e:
            self.logger.error(f"Error calculating volatility: {e}")
            return 5.0  # Default volatility
    
    def _calculate_technical_prediction(self, current_price: float, trend: str, 
                                      momentum: float, volatility: float) -> float:
        """Calculate predicted price based on technical factors."""
        try:
            # Base prediction on momentum with volatility constraints
            if trend == "Bullish" and momentum > 2:
                # Use momentum but cap it based on historical volatility
                price_change = min(momentum / 100 * 0.3, volatility / 100 * 0.5)
            elif trend == "Bearish" and momentum < -2:
                # Use momentum but cap it based on historical volatility  
                price_change = max(momentum / 100 * 0.3, -volatility / 100 * 0.5)
            else:
                # Conservative momentum-based prediction scaled by actual volatility
                price_change = (momentum / 100) * min(0.3, volatility / 100)
            
            return current_price * (1 + price_change)
            
        except Exception as e:
            self.logger.error(f"Error calculating technical prediction: {e}")
            return current_price
    
    def _calculate_indicator_alignment(self, data: pd.DataFrame) -> float:
        """Calculate how well technical indicators align."""
        try:
            signals = []
            
            # RSI signal
            if 'RSI' in data.columns:
                rsi = data['RSI'].iloc[-1]
                if pd.notna(rsi):
                    if 30 <= rsi <= 70:  # Neutral zone
                        signals.append(0.5)
                    elif rsi > 70:  # Overbought
                        signals.append(0.2)
                    else:  # Oversold
                        signals.append(0.8)
            
            # MACD alignment
            if 'MACD' in data.columns and 'MACD_signal' in data.columns:
                macd = data['MACD'].iloc[-1]
                macd_signal = data['MACD_signal'].iloc[-1]
                if pd.notna(macd) and pd.notna(macd_signal):
                    signals.append(0.8 if macd > macd_signal else 0.2)
            
            # Bollinger Bands position
            bb_position = self._get_bollinger_position(data)
            if bb_position is not None:
                # Middle band is neutral, extreme positions get lower scores
                signals.append(1 - abs(bb_position - 0.5) * 2)
            
            return np.mean(signals) if signals else 0.5
            
        except Exception as e:
            self.logger.error(f"Error calculating indicator alignment: {e}")
            return 0.5
    
    def _get_latest_rsi(self, data: pd.DataFrame) -> float:
        """Get latest RSI value."""
        try:
            if 'RSI' in data.columns:
                rsi = data['RSI'].iloc[-1]
                return round(rsi, 2) if pd.notna(rsi) else None
            return None
        except Exception:
            return None
    
    def _get_macd_signal(self, data: pd.DataFrame) -> str:
        """Get MACD signal."""
        try:
            if 'MACD' in data.columns and 'MACD_signal' in data.columns:
                macd = data['MACD'].iloc[-1]
                macd_signal = data['MACD_signal'].iloc[-1]
                if pd.notna(macd) and pd.notna(macd_signal):
                    return "Bullish" if macd > macd_signal else "Bearish"
            return "Neutral"
        except Exception:
            return "Neutral"
    
    def _get_bollinger_position(self, data: pd.DataFrame) -> float:
        """Get position within Bollinger Bands (0=lower, 0.5=middle, 1=upper)."""
        try:
            if all(col in data.columns for col in ['BB_upper', 'BB_lower']):
                current_price = data['Close'].iloc[-1]
                bb_upper = data['BB_upper'].iloc[-1]
                bb_lower = data['BB_lower'].iloc[-1]
                
                if pd.notna(bb_upper) and pd.notna(bb_lower) and bb_upper != bb_lower:
                    return (current_price - bb_lower) / (bb_upper - bb_lower)
            return None
        except Exception:
            return None


# Standalone function for direct usage
def predict_price_technical(symbol: str, days: int = 30) -> Dict[str, Any]:
    """
    Standalone function to predict price using technical analysis.
    
    Args:
        symbol: Stock symbol
        days: Prediction period in days
        
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
        model = TechnicalAnalysisModel(symbol, historical_data, prediction_days=days)
        return model.predict()
        
    except Exception as e:
        return {"error": f"Technical analysis failed: {e}"}
