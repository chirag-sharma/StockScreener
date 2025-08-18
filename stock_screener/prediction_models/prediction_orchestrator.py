"""
Price Prediction Orchestrator
============================

Coordinates all 6 prediction models and provides a unified interface.
Maintains backward compatibility with the original PricePrediction service.
"""

from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# Import all prediction models
from .technical_analysis_model import TechnicalAnalysisModel
from .fundamental_analysis_model import FundamentalAnalysisModel
from .machine_learning_model import MachineLearningModel
from .time_series_model import TimeSeriesModel
from .pattern_recognition_model import PatternRecognitionModel
from .volume_analysis_model import VolumeAnalysisModel


class PricePredictionOrchestrator:
    """
    Orchestrates all 6 prediction models to provide comprehensive price predictions.
    
    Models included:
    1. Technical Analysis
    2. Fundamental Analysis  
    3. Machine Learning
    4. Time Series
    5. Pattern Recognition
    6. Volume Analysis
    """
    
    def __init__(self, symbol: str, historical_data: pd.DataFrame = None, **kwargs):
        """
        Initialize the Price Prediction Orchestrator.
        
        Args:
            symbol: Stock symbol
            historical_data: Historical OHLCV data (optional, will fetch if not provided)
            **kwargs: Additional parameters for models
        """
        self.symbol = symbol
        self.historical_data = historical_data
        self.kwargs = kwargs
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Model weights (can be adjusted based on performance)
        self.model_weights = kwargs.get('model_weights', {
            'technical': 0.20,
            'fundamental': 0.15,
            'machine_learning': 0.25,
            'time_series': 0.20,
            'pattern_recognition': 0.10,
            'volume_analysis': 0.10
        })
        
        # Fetch data if not provided
        if self.historical_data is None:
            self._fetch_historical_data()
        
        # Initialize models
        self._initialize_models()
    
    def _fetch_historical_data(self):
        """Fetch historical data if not provided."""
        try:
            import yfinance as yf
            
            ticker = yf.Ticker(self.symbol)
            self.historical_data = ticker.history(period="1y")  # Get 1 year of data
            
            if self.historical_data.empty:
                raise ValueError(f"No historical data found for {self.symbol}")
                
        except Exception as e:
            self.logger.error(f"Failed to fetch data for {self.symbol}: {e}")
            raise
    
    def _initialize_models(self):
        """Initialize all prediction models."""
        try:
            self.models = {
                'technical': TechnicalAnalysisModel(self.symbol, self.historical_data, **self.kwargs),
                'fundamental': FundamentalAnalysisModel(self.symbol, self.historical_data, **self.kwargs),
                'machine_learning': MachineLearningModel(self.symbol, self.historical_data, **self.kwargs),
                'time_series': TimeSeriesModel(self.symbol, self.historical_data, **self.kwargs),
                'pattern_recognition': PatternRecognitionModel(self.symbol, self.historical_data, **self.kwargs),
                'volume_analysis': VolumeAnalysisModel(self.symbol, self.historical_data, **self.kwargs)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to initialize models: {e}")
            raise
    
    def predict_comprehensive(self, target_days: int = 30) -> Dict[str, Any]:
        """
        Generate comprehensive prediction using all models.
        
        Args:
            target_days: Number of days to predict ahead
            
        Returns:
            Comprehensive prediction results
        """
        try:
            # Get predictions from all models
            individual_predictions = self._get_individual_predictions(target_days)
            
            # Calculate ensemble prediction
            ensemble_result = self._calculate_ensemble_prediction(individual_predictions)
            
            # Add metadata
            ensemble_result.update({
                'symbol': self.symbol,
                'prediction_date': datetime.now(),
                'target_date': datetime.now() + timedelta(days=target_days),
                'target_days': target_days,
                'individual_predictions': individual_predictions,
                'model_weights': self.model_weights,
                'data_quality': self._assess_data_quality()
            })
            
            return ensemble_result
            
        except Exception as e:
            self.logger.error(f"Comprehensive prediction failed: {e}")
            return self._get_fallback_prediction()
    
    def _get_individual_predictions(self, target_days: int) -> Dict[str, Any]:
        """Get predictions from all individual models."""
        predictions = {}
        
        for model_name, model in self.models.items():
            try:
                # Update prediction days for each model
                model.prediction_days = target_days
                prediction = model.predict()
                predictions[model_name] = prediction
                
            except Exception as e:
                self.logger.warning(f"Model {model_name} failed: {e}")
                predictions[model_name] = {
                    'predicted_price': self._get_current_price(),
                    'confidence': 0.1,
                    'method': model_name.title(),
                    'error': str(e)
                }
        
        return predictions
    
    def _calculate_ensemble_prediction(self, individual_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate ensemble prediction from individual model results."""
        try:
            valid_predictions = []
            confidence_scores = []
            model_contributions = {}
            
            current_price = self._get_current_price()
            
            # Process individual predictions
            for model_name, prediction in individual_predictions.items():
                if 'error' not in prediction:
                    predicted_price = prediction.get('predicted_price', current_price)
                    confidence = prediction.get('confidence', 0.1)
                    
                    # Only include reasonable predictions (within ±50% of current price)
                    if 0.5 * current_price <= predicted_price <= 1.5 * current_price:
                        weight = self.model_weights.get(model_name, 0.1) * confidence
                        valid_predictions.append({
                            'price': predicted_price,
                            'weight': weight,
                            'model': model_name,
                            'confidence': confidence
                        })
                        confidence_scores.append(confidence)
            
            if not valid_predictions:
                return self._get_fallback_prediction()
            
            # Calculate weighted average
            total_weighted_price = sum(p['price'] * p['weight'] for p in valid_predictions)
            total_weight = sum(p['weight'] for p in valid_predictions)
            
            if total_weight > 0:
                ensemble_price = total_weighted_price / total_weight
            else:
                ensemble_price = np.mean([p['price'] for p in valid_predictions])
            
            # Calculate ensemble confidence
            ensemble_confidence = self._calculate_ensemble_confidence(
                valid_predictions, confidence_scores
            )
            
            # Calculate prediction statistics
            prediction_stats = self._calculate_prediction_statistics(valid_predictions, current_price)
            
            # Determine market signal
            price_change_pct = (ensemble_price - current_price) / current_price
            market_signal = self._determine_market_signal(price_change_pct, ensemble_confidence)
            
            return {
                'predicted_price': round(ensemble_price, 2),
                'current_price': round(current_price, 2),
                'price_change': round(ensemble_price - current_price, 2),
                'price_change_pct': round(price_change_pct * 100, 2),
                'confidence': round(ensemble_confidence, 2),
                'market_signal': market_signal,
                'method': 'Ensemble (6 Models)',
                'models_used': len(valid_predictions),
                'prediction_statistics': prediction_stats,
                'risk_assessment': self._assess_prediction_risk(valid_predictions, ensemble_confidence)
            }
            
        except Exception as e:
            self.logger.error(f"Ensemble calculation failed: {e}")
            return self._get_fallback_prediction()
    
    def _calculate_ensemble_confidence(self, predictions: List[Dict], confidence_scores: List[float]) -> float:
        """Calculate ensemble confidence score."""
        try:
            if not predictions:
                return 0.1
            
            # Factor 1: Average confidence of individual models
            avg_confidence = np.mean(confidence_scores)
            
            # Factor 2: Agreement between models (lower std deviation = higher confidence)
            prices = [p['price'] for p in predictions]
            price_std = np.std(prices)
            current_price = self._get_current_price()
            price_cv = price_std / current_price if current_price > 0 else 1
            agreement_factor = max(0, 1 - price_cv)  # Lower CV = higher agreement
            
            # Factor 3: Number of contributing models
            model_factor = min(1.0, len(predictions) / 6)  # More models = higher confidence
            
            # Combined confidence
            ensemble_confidence = (
                avg_confidence * 0.5 +
                agreement_factor * 0.3 +
                model_factor * 0.2
            )
            
            return max(0.1, min(0.9, ensemble_confidence))
            
        except Exception:
            return 0.5
    
    def _calculate_prediction_statistics(self, predictions: List[Dict], current_price: float) -> Dict[str, Any]:
        """Calculate statistics about the predictions."""
        try:
            prices = [p['price'] for p in predictions]
            
            return {
                'mean_prediction': round(np.mean(prices), 2),
                'median_prediction': round(np.median(prices), 2),
                'std_deviation': round(np.std(prices), 2),
                'min_prediction': round(min(prices), 2),
                'max_prediction': round(max(prices), 2),
                'prediction_range': round(max(prices) - min(prices), 2),
                'coefficient_of_variation': round(np.std(prices) / np.mean(prices), 3) if np.mean(prices) > 0 else 0,
                'predictions_above_current': sum(1 for p in prices if p > current_price),
                'predictions_below_current': sum(1 for p in prices if p < current_price)
            }
            
        except Exception:
            return {'error': 'Could not calculate statistics'}
    
    def _determine_market_signal(self, price_change_pct: float, confidence: float) -> str:
        """Determine market signal based on prediction and confidence."""
        try:
            if confidence < 0.3:
                return 'UNCERTAIN'
            elif price_change_pct > 0.05 and confidence > 0.6:
                return 'STRONG BUY'
            elif price_change_pct > 0.02 and confidence > 0.4:
                return 'BUY'
            elif price_change_pct < -0.05 and confidence > 0.6:
                return 'STRONG SELL'
            elif price_change_pct < -0.02 and confidence > 0.4:
                return 'SELL'
            else:
                return 'HOLD'
                
        except Exception:
            return 'HOLD'
    
    def _assess_prediction_risk(self, predictions: List[Dict], confidence: float) -> Dict[str, Any]:
        """Assess the risk associated with the prediction."""
        try:
            prices = [p['price'] for p in predictions]
            current_price = self._get_current_price()
            
            # Calculate volatility
            price_changes = [(p - current_price) / current_price for p in prices]
            volatility = np.std(price_changes)
            
            # Risk categories
            if volatility > 0.2:
                risk_level = 'HIGH'
            elif volatility > 0.1:
                risk_level = 'MEDIUM'
            else:
                risk_level = 'LOW'
            
            return {
                'risk_level': risk_level,
                'prediction_volatility': round(volatility, 3),
                'confidence_adjusted_risk': 'HIGH' if confidence < 0.4 else risk_level,
                'recommendation': self._get_risk_recommendation(risk_level, confidence)
            }
            
        except Exception:
            return {'risk_level': 'UNKNOWN', 'recommendation': 'Exercise caution'}
    
    def _get_risk_recommendation(self, risk_level: str, confidence: float) -> str:
        """Get risk-based recommendation."""
        if risk_level == 'HIGH' or confidence < 0.3:
            return 'High uncertainty - consider smaller position sizes'
        elif risk_level == 'MEDIUM':
            return 'Moderate risk - maintain normal position sizing'
        else:
            return 'Lower risk prediction - standard position sizing appropriate'
    
    def _assess_data_quality(self) -> Dict[str, Any]:
        """Assess the quality of input data."""
        try:
            data = self.historical_data
            
            return {
                'data_points': len(data),
                'date_range_days': (data.index[-1] - data.index[0]).days,
                'missing_values': data.isnull().sum().sum(),
                'has_volume_data': 'Volume' in data.columns and data['Volume'].sum() > 0,
                'data_completeness': round((1 - data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100, 1)
            }
            
        except Exception:
            return {'quality': 'Unknown'}
    
    def _get_current_price(self) -> float:
        """Get current price from historical data."""
        try:
            return float(self.historical_data['Close'].iloc[-1])
        except Exception:
            return 0.0
    
    def _get_fallback_prediction(self) -> Dict[str, Any]:
        """Get fallback prediction when ensemble fails."""
        current_price = self._get_current_price()
        
        return {
            'predicted_price': current_price,
            'current_price': current_price,
            'price_change': 0.0,
            'price_change_pct': 0.0,
            'confidence': 0.1,
            'market_signal': 'HOLD',
            'method': 'Fallback',
            'models_used': 0,
            'error': 'Ensemble prediction failed'
        }
    
    # Backward compatibility methods (matching original PricePrediction interface)
    
    def predict_price(self, days: int = 30) -> Dict[str, Any]:
        """
        Main prediction method (backward compatibility).
        
        Args:
            days: Number of days to predict ahead
            
        Returns:
            Prediction results
        """
        return self.predict_comprehensive(days)
    
    def get_individual_model_prediction(self, model_name: str, days: int = 30) -> Dict[str, Any]:
        """
        Get prediction from a specific model.
        
        Args:
            model_name: Name of the model ('technical', 'fundamental', etc.)
            days: Number of days to predict ahead
            
        Returns:
            Individual model prediction
        """
        try:
            if model_name not in self.models:
                return {'error': f'Model {model_name} not found'}
            
            model = self.models[model_name]
            model.prediction_days = days
            return model.predict()
            
        except Exception as e:
            return {'error': f'Model {model_name} failed: {e}'}
    
    def get_model_comparison(self, days: int = 30) -> Dict[str, Any]:
        """
        Compare predictions from all models.
        
        Args:
            days: Number of days to predict ahead
            
        Returns:
            Model comparison results
        """
        try:
            individual_predictions = self._get_individual_predictions(days)
            current_price = self._get_current_price()
            
            comparison = {
                'symbol': self.symbol,
                'current_price': current_price,
                'prediction_comparison': {}
            }
            
            for model_name, prediction in individual_predictions.items():
                if 'error' not in prediction:
                    pred_price = prediction.get('predicted_price', current_price)
                    comparison['prediction_comparison'][model_name] = {
                        'predicted_price': pred_price,
                        'price_change': pred_price - current_price,
                        'price_change_pct': round(((pred_price - current_price) / current_price) * 100, 2),
                        'confidence': prediction.get('confidence', 0.1),
                        'method': prediction.get('method', model_name.title())
                    }
                else:
                    comparison['prediction_comparison'][model_name] = {
                        'error': prediction.get('error', 'Unknown error')
                    }
            
            return comparison
            
        except Exception as e:
            return {'error': f'Model comparison failed: {e}'}


# Standalone functions for direct usage (backward compatibility)

def predict_stock_price(symbol: str, days: int = 30, **kwargs) -> Dict[str, Any]:
    """
    Standalone function to predict stock price using all models.
    
    Args:
        symbol: Stock symbol
        days: Prediction period in days
        **kwargs: Additional parameters
        
    Returns:
        Comprehensive prediction results
    """
    try:
        orchestrator = PricePredictionOrchestrator(symbol, **kwargs)
        return orchestrator.predict_comprehensive(days)
    except Exception as e:
        return {"error": f"Price prediction failed: {e}"}


def compare_prediction_models(symbol: str, days: int = 30, **kwargs) -> Dict[str, Any]:
    """
    Compare predictions from all models.
    
    Args:
        symbol: Stock symbol
        days: Prediction period in days
        **kwargs: Additional parameters
        
    Returns:
        Model comparison results
    """
    try:
        orchestrator = PricePredictionOrchestrator(symbol, **kwargs)
        return orchestrator.get_model_comparison(days)
    except Exception as e:
        return {"error": f"Model comparison failed: {e}"}
