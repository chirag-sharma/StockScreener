"""
Base Prediction Model
====================

Abstract base class for all prediction models to ensure consistent interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import logging


class BasePredictionModel(ABC):
    """
    Abstract base class for all price prediction models.
    
    This ensures all prediction models follow the same interface
    and can be used interchangeably by the orchestrator.
    """
    
    def __init__(self, symbol: str, historical_data: pd.DataFrame, **kwargs):
        """
        Initialize the prediction model.
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE.NS')
            historical_data: Historical price data with OHLCV columns
            **kwargs: Model-specific parameters
        """
        self.symbol = symbol
        self.historical_data = historical_data.copy()
        self.kwargs = kwargs
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Validate inputs
        if not self.validate_inputs():
            raise ValueError(f"Invalid inputs for {self.__class__.__name__}")
    
    @abstractmethod
    def predict(self) -> Dict[str, Any]:
        """
        Generate price prediction.
        
        Returns:
            Dict containing:
            - predicted_price: float
            - prediction_date: datetime
            - method: str (model name)
            - confidence: float (0-1)
            - additional_metrics: dict
        """
        pass
    
    @abstractmethod
    def get_confidence(self) -> float:
        """
        Calculate prediction confidence score.
        
        Returns:
            Confidence score between 0 and 1
        """
        pass
    
    def validate_inputs(self) -> bool:
        """
        Validate input data quality.
        
        Returns:
            True if inputs are valid, False otherwise
        """
        try:
            # Check if we have data
            if self.historical_data.empty:
                self.logger.error(f"No historical data for {self.symbol}")
                return False
            
            # Check required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns 
                             if col not in self.historical_data.columns]
            
            if missing_columns:
                self.logger.error(f"Missing columns for {self.symbol}: {missing_columns}")
                return False
            
            # Check for sufficient data points (at least 30 days)
            if len(self.historical_data) < 30:
                self.logger.warning(f"Limited data for {self.symbol}: {len(self.historical_data)} days")
                return False
            
            # Check for data quality
            if self.historical_data['Close'].isna().sum() > len(self.historical_data) * 0.1:
                self.logger.error(f"Too many NaN values in {self.symbol} data")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating inputs for {self.symbol}: {e}")
            return False
    
    def get_current_price(self) -> float:
        """Get the most recent closing price."""
        return float(self.historical_data['Close'].iloc[-1])
    
    def get_price_change_percent(self, predicted_price: float) -> float:
        """Calculate percentage change from current price."""
        current_price = self.get_current_price()
        return ((predicted_price - current_price) / current_price) * 100
    
    def add_technical_indicators(self) -> pd.DataFrame:
        """Add common technical indicators to the data."""
        df = self.historical_data.copy()
        
        try:
            # Simple Moving Averages
            df['SMA_10'] = df['Close'].rolling(window=10).mean()
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            
            # Exponential Moving Averages
            df['EMA_12'] = df['Close'].ewm(span=12).mean()
            df['EMA_26'] = df['Close'].ewm(span=26).mean()
            
            # MACD
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
            
            # Volume indicators
            df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding technical indicators: {e}")
            return df
    
    def get_support_resistance_levels(self) -> Dict[str, float]:
        """Calculate support and resistance levels."""
        try:
            df = self.historical_data.tail(100)  # Last 100 days
            
            # Find local maxima and minima
            high_points = df['High'].rolling(window=5, center=True).max() == df['High']
            low_points = df['Low'].rolling(window=5, center=True).min() == df['Low']
            
            resistance_levels = df[high_points]['High'].tolist()
            support_levels = df[low_points]['Low'].tolist()
            
            return {
                'resistance': np.mean(sorted(resistance_levels, reverse=True)[:3]) if resistance_levels else df['High'].max(),
                'support': np.mean(sorted(support_levels)[:3]) if support_levels else df['Low'].min()
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating support/resistance: {e}")
            return {
                'resistance': self.historical_data['High'].max(),
                'support': self.historical_data['Low'].min()
            }
