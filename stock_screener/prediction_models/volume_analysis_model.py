"""
Volume Analysis Price Prediction Model
=====================================

Uses volume-price relationships and volume patterns to predict prices.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from .base_model import BasePredictionModel


class VolumeAnalysisModel(BasePredictionModel):
    """
    Volume analysis-based price prediction using:
    - Volume-price divergence
    - On-Balance Volume (OBV)
    - Volume breakouts
    - Accumulation/Distribution patterns
    """
    
    def __init__(self, symbol: str, historical_data: pd.DataFrame, **kwargs):
        """
        Initialize Volume Analysis Model.
        
        Args:
            symbol: Stock symbol
            historical_data: Historical OHLCV data
            **kwargs: Additional parameters
                - prediction_days: Number of days to predict (default: 30)
                - volume_window: Days for volume analysis (default: 20)
        """
        super().__init__(symbol, historical_data, **kwargs)
        self.prediction_days = kwargs.get('prediction_days', 30)
        self.volume_window = kwargs.get('volume_window', 20)
    
    def predict(self) -> Dict[str, Any]:
        """Generate volume analysis-based price prediction."""
        try:
            current_price = self.get_current_price()
            
            # Calculate volume indicators
            volume_indicators = self._calculate_volume_indicators()
            
            if not volume_indicators:
                return {
                    "predicted_price": current_price,
                    "confidence": 0.1,
                    "method": "Volume Analysis",
                    "error": "Insufficient volume data"
                }
            
            # Generate prediction based on volume analysis
            prediction_result = self._generate_volume_prediction(volume_indicators, current_price)
            confidence = self.get_confidence()
            
            return {
                "predicted_price": round(prediction_result['price'], 2),
                "confidence": round(confidence, 2),
                "volume_signal": prediction_result['signal'],
                "volume_strength": prediction_result['strength'],
                "method": "Volume Analysis",
                "prediction_date": pd.Timestamp.now() + pd.Timedelta(days=self.prediction_days),
                "volume_indicators": volume_indicators,
                "additional_metrics": {
                    "volume_trend": self._get_volume_trend(),
                    "accumulation_distribution": self._calculate_accumulation_distribution(),
                    "volume_price_correlation": self._calculate_volume_price_correlation()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Volume analysis failed: {e}")
            return {
                "predicted_price": self.get_current_price(),
                "confidence": 0.1,
                "method": "Volume Analysis",
                "error": f"Prediction failed: {e}"
            }
    
    def get_confidence(self) -> float:
        """Calculate confidence based on volume pattern strength."""
        try:
            volume_indicators = self._calculate_volume_indicators()
            
            if not volume_indicators:
                return 0.1
            
            # Base confidence on indicator alignment
            obv_signal_strength = abs(volume_indicators.get('obv_trend', 0))
            volume_breakout_strength = volume_indicators.get('volume_breakout_strength', 0)
            price_volume_correlation = abs(volume_indicators.get('price_volume_correlation', 0))
            
            # Weight different factors
            confidence = (
                obv_signal_strength * 0.4 +
                volume_breakout_strength * 0.3 +
                price_volume_correlation * 0.3
            )
            
            return max(0.1, min(0.8, confidence))
            
        except Exception:
            return 0.1
    
    def _calculate_volume_indicators(self) -> Dict[str, Any]:
        """Calculate various volume-based indicators."""
        try:
            data = self.historical_data.tail(max(50, self.volume_window))
            
            if len(data) < self.volume_window:
                return {}
            
            indicators = {}
            
            # 1. On-Balance Volume (OBV)
            indicators.update(self._calculate_obv(data))
            
            # 2. Volume Rate of Change
            indicators.update(self._calculate_volume_roc(data))
            
            # 3. Volume breakout analysis
            indicators.update(self._analyze_volume_breakout(data))
            
            # 4. Price-Volume correlation
            indicators['price_volume_correlation'] = self._calculate_volume_price_correlation()
            
            # 5. Volume moving averages
            indicators.update(self._calculate_volume_moving_averages(data))
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating volume indicators: {e}")
            return {}
    
    def _calculate_obv(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate On-Balance Volume and its trend."""
        try:
            closes = data['Close']
            volumes = data['Volume']
            
            # Calculate OBV
            obv = [0]
            for i in range(1, len(data)):
                if closes.iloc[i] > closes.iloc[i-1]:
                    obv.append(obv[-1] + volumes.iloc[i])
                elif closes.iloc[i] < closes.iloc[i-1]:
                    obv.append(obv[-1] - volumes.iloc[i])
                else:
                    obv.append(obv[-1])
            
            obv_series = pd.Series(obv, index=data.index)
            
            # Calculate OBV trend (slope of last 10 days)
            if len(obv_series) >= 10:
                x = np.arange(10)
                y = obv_series.tail(10).values
                slope = np.polyfit(x, y, 1)[0]
                obv_trend = slope / abs(obv_series.iloc[-1]) if obv_series.iloc[-1] != 0 else 0
            else:
                obv_trend = 0
            
            return {
                'obv_current': obv_series.iloc[-1],
                'obv_trend': obv_trend,
                'obv_signal': 'Bullish' if obv_trend > 0.01 else 'Bearish' if obv_trend < -0.01 else 'Neutral'
            }
            
        except Exception:
            return {'obv_current': 0, 'obv_trend': 0, 'obv_signal': 'Neutral'}
    
    def _calculate_volume_roc(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Volume Rate of Change."""
        try:
            volumes = data['Volume']
            
            # 5-day Volume ROC
            volume_roc_5 = (volumes.iloc[-1] - volumes.iloc[-6]) / volumes.iloc[-6] if len(volumes) >= 6 else 0
            
            # 10-day Volume ROC
            volume_roc_10 = (volumes.iloc[-1] - volumes.iloc[-11]) / volumes.iloc[-11] if len(volumes) >= 11 else 0
            
            return {
                'volume_roc_5': volume_roc_5,
                'volume_roc_10': volume_roc_10,
                'volume_momentum': 'Strong' if volume_roc_5 > 0.5 else 'Weak' if volume_roc_5 < -0.3 else 'Normal'
            }
            
        except Exception:
            return {'volume_roc_5': 0, 'volume_roc_10': 0, 'volume_momentum': 'Normal'}
    
    def _analyze_volume_breakout(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume breakouts and their strength."""
        try:
            volumes = data['Volume']
            volume_sma = volumes.rolling(self.volume_window).mean()
            
            current_volume = volumes.iloc[-1]
            avg_volume = volume_sma.iloc[-1]
            
            if avg_volume == 0:
                return {'volume_breakout_strength': 0, 'volume_breakout_signal': 'Neutral'}
            
            volume_ratio = current_volume / avg_volume
            
            # Determine breakout strength
            if volume_ratio > 2.0:
                strength = min(1.0, (volume_ratio - 1) / 4)  # Cap at 1.0
                signal = 'Very Strong'
            elif volume_ratio > 1.5:
                strength = 0.7
                signal = 'Strong'
            elif volume_ratio > 1.2:
                strength = 0.5
                signal = 'Moderate'
            elif volume_ratio < 0.5:
                strength = 0.3
                signal = 'Low Volume'
            else:
                strength = 0.4
                signal = 'Normal'
            
            return {
                'volume_breakout_strength': strength,
                'volume_breakout_signal': signal,
                'volume_ratio': volume_ratio
            }
            
        except Exception:
            return {'volume_breakout_strength': 0, 'volume_breakout_signal': 'Normal', 'volume_ratio': 1.0}
    
    def _calculate_volume_moving_averages(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volume moving averages and signals."""
        try:
            volumes = data['Volume']
            
            # Short and long volume MAs
            volume_sma_5 = volumes.rolling(5).mean()
            volume_sma_20 = volumes.rolling(20).mean()
            
            if len(volume_sma_5) < 5 or len(volume_sma_20) < 20:
                return {'volume_ma_signal': 'Neutral', 'volume_ma_strength': 0.4}
            
            current_sma5 = volume_sma_5.iloc[-1]
            current_sma20 = volume_sma_20.iloc[-1]
            
            # Volume MA crossover
            if current_sma5 > current_sma20 * 1.1:
                return {'volume_ma_signal': 'Increasing Volume', 'volume_ma_strength': 0.7}
            elif current_sma5 < current_sma20 * 0.9:
                return {'volume_ma_signal': 'Decreasing Volume', 'volume_ma_strength': 0.6}
            else:
                return {'volume_ma_signal': 'Stable Volume', 'volume_ma_strength': 0.4}
                
        except Exception:
            return {'volume_ma_signal': 'Neutral', 'volume_ma_strength': 0.4}
    
    def _calculate_volume_price_correlation(self) -> float:
        """Calculate correlation between volume and price changes."""
        try:
            data = self.historical_data.tail(30)  # Last 30 days
            
            if len(data) < 10:
                return 0
            
            price_changes = data['Close'].pct_change().dropna()
            volume_changes = data['Volume'].pct_change().dropna()
            
            # Align the series
            min_len = min(len(price_changes), len(volume_changes))
            if min_len < 5:
                return 0
            
            correlation = np.corrcoef(
                price_changes.tail(min_len).values,
                volume_changes.tail(min_len).values
            )[0, 1]
            
            return correlation if not np.isnan(correlation) else 0
            
        except Exception:
            return 0
    
    def _calculate_accumulation_distribution(self) -> float:
        """Calculate Accumulation/Distribution Line."""
        try:
            data = self.historical_data.tail(20)
            
            ad_values = []
            for _, row in data.iterrows():
                if row['High'] != row['Low']:
                    money_flow_multiplier = ((row['Close'] - row['Low']) - (row['High'] - row['Close'])) / (row['High'] - row['Low'])
                else:
                    money_flow_multiplier = 0
                
                money_flow_volume = money_flow_multiplier * row['Volume']
                ad_values.append(money_flow_volume)
            
            # Sum to get A/D line
            ad_line = np.cumsum(ad_values)
            
            # Return trend (positive = accumulation, negative = distribution)
            if len(ad_line) >= 5:
                return (ad_line[-1] - ad_line[-5]) / abs(ad_line[-5]) if ad_line[-5] != 0 else 0
            else:
                return 0
                
        except Exception:
            return 0
    
    def _get_volume_trend(self) -> str:
        """Determine overall volume trend."""
        try:
            volumes = self.historical_data['Volume'].tail(20)
            
            # Linear regression on volume
            x = np.arange(len(volumes))
            slope = np.polyfit(x, volumes.values, 1)[0]
            
            avg_volume = volumes.mean()
            trend_strength = abs(slope) / avg_volume if avg_volume != 0 else 0
            
            if slope > 0 and trend_strength > 0.02:
                return "Increasing"
            elif slope < 0 and trend_strength > 0.02:
                return "Decreasing"
            else:
                return "Stable"
                
        except Exception:
            return "Unknown"
    
    def _generate_volume_prediction(self, indicators: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """Generate price prediction based on volume indicators."""
        try:
            # Weight different volume signals
            signals = []
            
            # OBV signal
            obv_trend = indicators.get('obv_trend', 0)
            if obv_trend > 0.01:
                signals.append({'direction': 1, 'strength': min(0.8, obv_trend * 20), 'weight': 0.3})
            elif obv_trend < -0.01:
                signals.append({'direction': -1, 'strength': min(0.8, abs(obv_trend) * 20), 'weight': 0.3})
            
            # Volume breakout signal
            volume_strength = indicators.get('volume_breakout_strength', 0)
            volume_ratio = indicators.get('volume_ratio', 1.0)
            if volume_ratio > 1.5:  # High volume usually bullish
                signals.append({'direction': 1, 'strength': volume_strength, 'weight': 0.25})
            elif volume_ratio < 0.7:  # Low volume can be bearish
                signals.append({'direction': -1, 'strength': 0.4, 'weight': 0.15})
            
            # Price-volume correlation
            correlation = indicators.get('price_volume_correlation', 0)
            if abs(correlation) > 0.3:
                direction = 1 if correlation > 0 else -1
                signals.append({'direction': direction, 'strength': abs(correlation), 'weight': 0.2})
            
            # Volume ROC signal
            volume_roc = indicators.get('volume_roc_5', 0)
            if volume_roc > 0.3:
                signals.append({'direction': 1, 'strength': min(0.7, volume_roc), 'weight': 0.15})
            elif volume_roc < -0.3:
                signals.append({'direction': -1, 'strength': min(0.7, abs(volume_roc)), 'weight': 0.15})
            
            # Calculate weighted prediction
            if not signals:
                return {'price': current_price, 'signal': 'Neutral', 'strength': 0.3}
            
            total_weighted_signal = 0
            total_weight = 0
            
            for signal in signals:
                weighted_signal = signal['direction'] * signal['strength'] * signal['weight']
                total_weighted_signal += weighted_signal
                total_weight += signal['weight']
            
            if total_weight > 0:
                avg_signal = total_weighted_signal / total_weight
                expected_move = avg_signal * 0.05  # Max 5% move based on volume
                predicted_price = current_price * (1 + expected_move)
                
                signal_direction = 'Bullish' if avg_signal > 0.1 else 'Bearish' if avg_signal < -0.1 else 'Neutral'
                signal_strength = abs(avg_signal)
                
                return {
                    'price': predicted_price,
                    'signal': signal_direction,
                    'strength': signal_strength
                }
            else:
                return {'price': current_price, 'signal': 'Neutral', 'strength': 0.3}
                
        except Exception:
            return {'price': current_price, 'signal': 'Neutral', 'strength': 0.3}


# Standalone function for direct usage
def predict_price_volume(symbol: str, days: int = 30) -> Dict[str, Any]:
    """
    Standalone function to predict price using volume analysis.
    
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
        historical_data = ticker.history(period="3mo")  # Need volume data
        
        if historical_data.empty:
            return {"error": f"No data found for {symbol}"}
        
        # Create and run model
        model = VolumeAnalysisModel(symbol, historical_data, prediction_days=days)
        return model.predict()
        
    except Exception as e:
        return {"error": f"Volume analysis failed: {e}"}
