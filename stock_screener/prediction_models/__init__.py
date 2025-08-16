"""
Modular Price Prediction System
===============================

This package contains individual prediction models that can be used
independently or combined through the PricePredictionOrchestrator.

Each model follows the same interface:
- __init__(symbol, historical_data, **kwargs)
- predict() -> Dict[str, Any]
- get_confidence() -> float
- validate_inputs() -> bool

Models included:
1. TechnicalAnalysisModel - Moving averages, support/resistance
2. FundamentalAnalysisModel - DCF, P/E ratios, Graham formula  
3. MachineLearningModel - Regression, Random Forest
4. TimeSeriesModel - ARIMA, seasonal decomposition
5. PatternRecognitionModel - Chart patterns, trend analysis
6. VolumeAnalysisModel - Volume-price relationship analysis
"""

from .base_model import BasePredictionModel
from .technical_analysis_model import TechnicalAnalysisModel
from .fundamental_analysis_model import FundamentalAnalysisModel
from .machine_learning_model import MachineLearningModel
from .time_series_model import TimeSeriesModel
from .pattern_recognition_model import PatternRecognitionModel
from .volume_analysis_model import VolumeAnalysisModel
from .prediction_orchestrator import PricePredictionOrchestrator

# Import standalone functions
from .technical_analysis_model import predict_price_technical
from .fundamental_analysis_model import predict_price_fundamental
from .machine_learning_model import predict_price_ml
from .time_series_model import predict_price_timeseries
from .pattern_recognition_model import predict_price_patterns
from .volume_analysis_model import predict_price_volume
from .prediction_orchestrator import predict_stock_price, compare_prediction_models

__all__ = [
    'BasePredictionModel',
    'TechnicalAnalysisModel', 
    'FundamentalAnalysisModel',
    'MachineLearningModel',
    'TimeSeriesModel',
    'PatternRecognitionModel',
    'VolumeAnalysisModel',
    'PricePredictionOrchestrator',
    # Standalone functions
    'predict_price_technical',
    'predict_price_fundamental',
    'predict_price_ml',
    'predict_price_timeseries',
    'predict_price_patterns',
    'predict_price_volume',
    'predict_stock_price',
    'compare_prediction_models'
]
