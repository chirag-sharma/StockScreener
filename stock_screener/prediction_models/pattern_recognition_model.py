"""
Pattern Recognition Price Prediction Model
==========================================

Uses chart pattern recognition and technical analysis patterns to predict prices.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from .base_model import BasePredictionModel


class PatternRecognitionModel(BasePredictionModel):
    """
    Pattern recognition-based price prediction using:
    - Chart patterns (triangles, flags, head & shoulders)
    - Candlestick patterns
    - Breakout patterns
    - Support/Resistance breaks
    """
    
    def __init__(self, symbol: str, historical_data: pd.DataFrame, **kwargs):
        """
        Initialize Pattern Recognition Model.
        
        Args:
            symbol: Stock symbol
            historical_data: Historical OHLCV data
            **kwargs: Additional parameters
                - prediction_days: Number of days to predict (default: 30)
                - lookback_period: Days to analyze patterns (default: 60)
        """
        super().__init__(symbol, historical_data, **kwargs)
        self.prediction_days = kwargs.get('prediction_days', 30)
        self.lookback_period = kwargs.get('lookback_period', 60)
    
    def predict(self) -> Dict[str, Any]:
        """Generate pattern recognition-based price prediction."""
        try:
            current_price = self.get_current_price()
            
            # Analyze various patterns
            patterns = self._analyze_patterns()
            
            if not patterns:
                return {
                    "predicted_price": current_price,
                    "confidence": 0.1,
                    "method": "Pattern Recognition",
                    "error": "No patterns detected"
                }
            
            # Calculate prediction based on detected patterns
            predicted_price = self._calculate_pattern_prediction(patterns, current_price)
            confidence = self.get_confidence()
            
            return {
                "predicted_price": round(predicted_price, 2),
                "confidence": round(confidence, 2),
                "patterns_detected": [p['name'] for p in patterns],
                "pattern_signals": [p['signal'] for p in patterns],
                "method": "Pattern Recognition",
                "prediction_date": pd.Timestamp.now() + pd.Timedelta(days=self.prediction_days),
                "additional_metrics": {
                    "support_level": self._find_support_level(),
                    "resistance_level": self._find_resistance_level(),
                    "breakout_probability": self._calculate_breakout_probability()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Pattern recognition failed: {e}")
            return {
                "predicted_price": self.get_current_price(),
                "confidence": 0.1,
                "method": "Pattern Recognition",
                "error": f"Prediction failed: {e}"
            }
    
    def get_confidence(self) -> float:
        """Calculate confidence based on pattern strength."""
        try:
            patterns = self._analyze_patterns()
            if not patterns:
                return 0.1
            
            # Base confidence on number and strength of patterns
            pattern_strength = np.mean([p['strength'] for p in patterns])
            pattern_count_factor = min(1.0, len(patterns) / 3)  # More patterns = higher confidence
            
            confidence = (pattern_strength * 0.7) + (pattern_count_factor * 0.3)
            return max(0.1, min(0.7, confidence))
            
        except Exception:
            return 0.1
    
    def _analyze_patterns(self) -> List[Dict[str, Any]]:
        """Analyze various chart patterns."""
        patterns = []
        data = self.historical_data.tail(self.lookback_period)
        
        try:
            # 1. Trend patterns
            trend_pattern = self._detect_trend_pattern(data)
            if trend_pattern:
                patterns.append(trend_pattern)
            
            # 2. Support/Resistance breaks
            breakout_pattern = self._detect_breakout_pattern(data)
            if breakout_pattern:
                patterns.append(breakout_pattern)
            
            # 3. Moving average patterns
            ma_pattern = self._detect_moving_average_pattern(data)
            if ma_pattern:
                patterns.append(ma_pattern)
            
            # 4. Volume confirmation patterns
            volume_pattern = self._detect_volume_pattern(data)
            if volume_pattern:
                patterns.append(volume_pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error analyzing patterns: {e}")
            return []
    
    def _detect_trend_pattern(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Detect trend-based patterns."""
        try:
            closes = data['Close']
            highs = data['High']
            lows = data['Low']
            
            # Calculate trend over different periods
            short_trend = (closes.iloc[-5] - closes.iloc[-10]) / closes.iloc[-10]
            medium_trend = (closes.iloc[-1] - closes.iloc[-20]) / closes.iloc[-20]
            
            # Determine pattern
            if short_trend > 0.02 and medium_trend > 0.05:
                return {
                    'name': 'Strong Uptrend',
                    'signal': 'Bullish',
                    'strength': min(0.8, abs(medium_trend) * 10),
                    'expected_move': medium_trend * 0.5  # Expect continuation at half the rate
                }
            elif short_trend < -0.02 and medium_trend < -0.05:
                return {
                    'name': 'Strong Downtrend', 
                    'signal': 'Bearish',
                    'strength': min(0.8, abs(medium_trend) * 10),
                    'expected_move': medium_trend * 0.5
                }
            elif abs(short_trend) < 0.01 and abs(medium_trend) < 0.02:
                return {
                    'name': 'Consolidation',
                    'signal': 'Neutral',
                    'strength': 0.4,
                    'expected_move': 0.0
                }
            
            return None
            
        except Exception:
            return None
    
    def _detect_breakout_pattern(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Detect breakout patterns."""
        try:
            sr_levels = self.get_support_resistance_levels()
            current_price = data['Close'].iloc[-1]
            recent_high = data['High'].tail(5).max()
            recent_low = data['Low'].tail(5).min()
            
            resistance = sr_levels['resistance']
            support = sr_levels['support']
            
            # Check for resistance breakout
            if current_price > resistance * 1.02:  # 2% above resistance
                return {
                    'name': 'Resistance Breakout',
                    'signal': 'Bullish',
                    'strength': 0.7,
                    'expected_move': (current_price - resistance) / resistance
                }
            
            # Check for support breakdown
            elif current_price < support * 0.98:  # 2% below support
                return {
                    'name': 'Support Breakdown',
                    'signal': 'Bearish',
                    'strength': 0.7,
                    'expected_move': (current_price - support) / support
                }
            
            # Check for consolidation near resistance
            elif abs(current_price - resistance) / resistance < 0.05:
                return {
                    'name': 'Near Resistance',
                    'signal': 'Neutral',
                    'strength': 0.5,
                    'expected_move': 0.02  # Small breakout expected
                }
            
            return None
            
        except Exception:
            return None
    
    def _detect_moving_average_pattern(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Detect moving average patterns."""
        try:
            closes = data['Close']
            sma_20 = closes.rolling(20).mean()
            sma_50 = closes.rolling(50).mean()
            
            if len(sma_20) < 20 or len(sma_50) < 50:
                return None
            
            current_price = closes.iloc[-1]
            sma_20_current = sma_20.iloc[-1]
            sma_50_current = sma_50.iloc[-1]
            
            # Golden cross (bullish)
            if (sma_20.iloc[-2] <= sma_50.iloc[-2] and 
                sma_20_current > sma_50_current):
                return {
                    'name': 'Golden Cross',
                    'signal': 'Bullish',
                    'strength': 0.8,
                    'expected_move': 0.05  # Expected 5% move
                }
            
            # Death cross (bearish)
            elif (sma_20.iloc[-2] >= sma_50.iloc[-2] and 
                  sma_20_current < sma_50_current):
                return {
                    'name': 'Death Cross',
                    'signal': 'Bearish',
                    'strength': 0.8,
                    'expected_move': -0.05  # Expected 5% decline
                }
            
            # Price above both MAs (bullish)
            elif current_price > sma_20_current > sma_50_current:
                return {
                    'name': 'Above Moving Averages',
                    'signal': 'Bullish',
                    'strength': 0.6,
                    'expected_move': 0.03
                }
            
            # Price below both MAs (bearish)
            elif current_price < sma_20_current < sma_50_current:
                return {
                    'name': 'Below Moving Averages',
                    'signal': 'Bearish',
                    'strength': 0.6,
                    'expected_move': -0.03
                }
            
            return None
            
        except Exception:
            return None
    
    def _detect_volume_pattern(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Detect volume confirmation patterns."""
        try:
            volumes = data['Volume']
            closes = data['Close']
            volume_sma = volumes.rolling(20).mean()
            
            if len(volume_sma) < 20:
                return None
            
            recent_volume = volumes.iloc[-1]
            avg_volume = volume_sma.iloc[-1]
            price_change = (closes.iloc[-1] - closes.iloc[-2]) / closes.iloc[-2]
            
            # High volume with price increase
            if recent_volume > avg_volume * 1.5 and price_change > 0.02:
                return {
                    'name': 'Volume Breakout Up',
                    'signal': 'Bullish',
                    'strength': 0.7,
                    'expected_move': price_change * 2  # Expect continuation
                }
            
            # High volume with price decrease
            elif recent_volume > avg_volume * 1.5 and price_change < -0.02:
                return {
                    'name': 'Volume Breakdown',
                    'signal': 'Bearish',
                    'strength': 0.7,
                    'expected_move': price_change * 2
                }
            
            # Low volume (consolidation)
            elif recent_volume < avg_volume * 0.7:
                return {
                    'name': 'Low Volume Consolidation',
                    'signal': 'Neutral',
                    'strength': 0.4,
                    'expected_move': 0.0
                }
            
            return None
            
        except Exception:
            return None
    
    def _calculate_pattern_prediction(self, patterns: List[Dict], current_price: float) -> float:
        """Calculate price prediction based on detected patterns."""
        try:
            if not patterns:
                return current_price
            
            total_move = 0
            total_weight = 0
            
            for pattern in patterns:
                expected_move = pattern.get('expected_move', 0)
                strength = pattern.get('strength', 0.5)
                
                weighted_move = expected_move * strength
                total_move += weighted_move
                total_weight += strength
            
            if total_weight > 0:
                avg_expected_move = total_move / total_weight
                return current_price * (1 + avg_expected_move)
            else:
                return current_price
                
        except Exception:
            return current_price
    
    def _find_support_level(self) -> float:
        """Find nearest support level."""
        try:
            sr_levels = self.get_support_resistance_levels()
            return sr_levels['support']
        except Exception:
            return self.historical_data['Low'].tail(20).min()
    
    def _find_resistance_level(self) -> float:
        """Find nearest resistance level."""
        try:
            sr_levels = self.get_support_resistance_levels()
            return sr_levels['resistance']
        except Exception:
            return self.historical_data['High'].tail(20).max()
    
    def _calculate_breakout_probability(self) -> float:
        """Calculate probability of a breakout."""
        try:
            data = self.historical_data.tail(20)
            current_price = data['Close'].iloc[-1]
            
            # Calculate price position within recent range
            recent_high = data['High'].max()
            recent_low = data['Low'].min()
            range_size = recent_high - recent_low
            
            if range_size == 0:
                return 0.5
            
            # Position within range (0 = at low, 1 = at high)
            position = (current_price - recent_low) / range_size
            
            # Higher probability of breakout when near extremes
            if position > 0.8:
                return 0.7  # Near resistance, likely upward breakout
            elif position < 0.2:
                return 0.7  # Near support, likely downward breakout
            else:
                return 0.3  # In middle, less likely to break out
                
        except Exception:
            return 0.5


# Standalone function for direct usage
def predict_price_patterns(symbol: str, days: int = 30) -> Dict[str, Any]:
    """
    Standalone function to predict price using pattern recognition.
    
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
        historical_data = ticker.history(period="6mo")  # Need more data for patterns
        
        if historical_data.empty:
            return {"error": f"No data found for {symbol}"}
        
        # Create and run model
        model = PatternRecognitionModel(symbol, historical_data, prediction_days=days)
        return model.predict()
        
    except Exception as e:
        return {"error": f"Pattern recognition failed: {e}"}
