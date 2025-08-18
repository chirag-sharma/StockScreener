"""
Machine Learning Price Prediction Model
======================================

Uses scikit-learn models to predict stock prices based on technical features.
"""

from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from .base_model import BasePredictionModel

# Optional imports for machine learning
try:
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.svm import SVR
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import cross_val_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class MachineLearningModel(BasePredictionModel):
    """
    Machine learning-based price prediction using:
    - Linear Regression
    - Random Forest
    - Gradient Boosting
    - Support Vector Regression
    - Ridge/Lasso Regression
    """
    
    def __init__(self, symbol: str, historical_data: pd.DataFrame, **kwargs):
        """
        Initialize Machine Learning Model.
        
        Args:
            symbol: Stock symbol
            historical_data: Historical OHLCV data
            **kwargs: Additional parameters
                - prediction_days: Number of days to predict (default: 30)
                - models_to_use: List of model names to include
                - validation_split: Fraction for validation (default: 0.2)
        """
        super().__init__(symbol, historical_data, **kwargs)
        
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn is required for machine learning models")
        
        self.prediction_days = kwargs.get('prediction_days', 30)
        self.models_to_use = kwargs.get('models_to_use', ['all'])
        self.validation_split = kwargs.get('validation_split', 0.2)
        
        # Prepare features
        self.prepared_data = self._prepare_features()
    
    def predict(self) -> Dict[str, Any]:
        """
        Generate machine learning-based price prediction.
        
        Returns:
            Dict with predicted_price, confidence, best_model, etc.
        """
        try:
            if self.prepared_data is None or len(self.prepared_data) < 50:
                return {
                    "predicted_price": self.get_current_price(),
                    "confidence": 0.1,
                    "method": "Machine Learning",
                    "error": "Insufficient data for machine learning"
                }
            
            # Prepare training data
            X, y = self._create_feature_target_arrays()
            if X is None or len(X) < 30:
                return {
                    "predicted_price": self.get_current_price(),
                    "confidence": 0.1,
                    "method": "Machine Learning",
                    "error": "Failed to create feature arrays"
                }
            
            # Train and evaluate models
            model_results = self._train_and_evaluate_models(X, y)
            
            if not model_results:
                return {
                    "predicted_price": self.get_current_price(),
                    "confidence": 0.1,
                    "method": "Machine Learning",
                    "error": "All models failed to train"
                }
            
            # Get best model and make prediction
            best_result = min(model_results, key=lambda x: x['mse'])
            future_price = self._make_future_prediction(best_result, X)
            
            # Calculate confidence
            confidence = self.get_confidence()
            
            return {
                "predicted_price": round(future_price, 2),
                "confidence": round(confidence, 2),
                "best_model": best_result['name'],
                "model_performance": {
                    result['name']: {
                        'mse': round(result['mse'], 4),
                        'mae': round(result['mae'], 4),
                        'r2_score': round(result['r2'], 4)
                    }
                    for result in model_results
                },
                "method": "Machine Learning",
                "prediction_date": pd.Timestamp.now() + pd.Timedelta(days=self.prediction_days),
                "additional_metrics": {
                    "features_used": len(X[0]) if len(X) > 0 else 0,
                    "training_samples": len(X),
                    "ensemble_prediction": round(np.mean([r['prediction'] for r in model_results]), 2)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Machine learning prediction failed: {e}")
            return {
                "predicted_price": self.get_current_price(),
                "confidence": 0.1,
                "method": "Machine Learning",
                "error": f"Prediction failed: {e}"
            }
    
    def get_confidence(self) -> float:
        """
        Calculate confidence based on model performance and data quality.
        
        Returns:
            Confidence score between 0.1 and 0.8
        """
        try:
            if self.prepared_data is None:
                return 0.1
            
            confidence_factors = []
            
            # Data quantity factor
            data_count = len(self.prepared_data)
            data_factor = min(1.0, data_count / 200)  # 200+ samples = full confidence
            confidence_factors.append(data_factor * 0.3)
            
            # Feature quality factor (check for NaN ratio)
            feature_cols = [col for col in self.prepared_data.columns 
                           if col not in ['Close', 'Date']]
            if feature_cols:
                nan_ratio = self.prepared_data[feature_cols].isna().sum().sum() / (len(self.prepared_data) * len(feature_cols))
                feature_quality = 1 - nan_ratio
                confidence_factors.append(feature_quality * 0.3)
            
            # Price volatility factor (lower volatility = higher confidence)
            volatility = self.prepared_data['Close'].pct_change().std()
            volatility_factor = max(0.1, min(1.0, 1 - volatility * 10))  # Cap volatility impact
            confidence_factors.append(volatility_factor * 0.2)
            
            # Trend consistency factor
            price_changes = self.prepared_data['Close'].pct_change().dropna()
            if len(price_changes) > 0:
                trend_consistency = 1 - abs(price_changes.mean()) * 50  # Penalize extreme trends
                trend_consistency = max(0.1, min(1.0, trend_consistency))
                confidence_factors.append(trend_consistency * 0.2)
            
            overall_confidence = sum(confidence_factors) if confidence_factors else 0.1
            return max(0.1, min(0.8, overall_confidence))
            
        except Exception as e:
            self.logger.error(f"Error calculating ML confidence: {e}")
            return 0.1
    
    def _prepare_features(self) -> Optional[pd.DataFrame]:
        """Prepare feature set for machine learning."""
        try:
            data = self.historical_data.copy()
            
            # Basic price features
            data['SMA_5'] = data['Close'].rolling(window=5).mean()
            data['SMA_10'] = data['Close'].rolling(window=10).mean()
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50, min_periods=25).mean()
            
            # Exponential moving averages
            data['EMA_12'] = data['Close'].ewm(span=12).mean()
            data['EMA_26'] = data['Close'].ewm(span=26).mean()
            
            # Price changes and returns
            data['Price_Change'] = data['Close'].pct_change()
            data['Price_Change_2d'] = data['Close'].pct_change(periods=2)
            data['Price_Change_5d'] = data['Close'].pct_change(periods=5)
            
            # Volume features
            data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
            data['Volume_Change'] = data['Volume'].pct_change()
            data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
            
            # Price ratios and spreads
            data['High_Low_Ratio'] = data['High'] / data['Low'].replace(0, np.nan)
            data['Open_Close_Ratio'] = data['Open'] / data['Close'].replace(0, np.nan)
            data['High_Close_Spread'] = (data['High'] - data['Close']) / data['Close']
            data['Low_Close_Spread'] = (data['Close'] - data['Low']) / data['Close']
            
            # Volatility features
            data['Volatility_5d'] = data['Close'].rolling(window=5).std()
            data['Volatility_20d'] = data['Close'].rolling(window=20).std()
            
            # MACD
            data['MACD'] = data['EMA_12'] - data['EMA_26']
            data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
            data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
            
            # RSI (simple implementation)
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, np.nan)
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            bb_window = 20
            data['BB_Middle'] = data['Close'].rolling(window=bb_window).mean()
            bb_std = data['Close'].rolling(window=bb_window).std()
            data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
            data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
            data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
            
            # Lagged features
            for lag in [1, 2, 3, 5]:
                data[f'Close_lag_{lag}'] = data['Close'].shift(lag)
                data[f'Volume_lag_{lag}'] = data['Volume'].shift(lag)
            
            # Remove infinite values
            data = data.replace([np.inf, -np.inf], np.nan)
            
            # Drop early rows with NaN values
            data = data.dropna()
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            return None
    
    def _create_feature_target_arrays(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Create feature matrix X and target array y."""
        try:
            data = self.prepared_data
            
            # Select feature columns (exclude target and date columns)
            exclude_cols = ['Close', 'Date'] if 'Date' in data.columns else ['Close']
            feature_cols = [col for col in data.columns if col not in exclude_cols]
            
            X = data[feature_cols].values
            y = data['Close'].values
            
            # Final check for invalid values
            if not np.all(np.isfinite(X)):
                self.logger.warning("Found infinite values in features, using imputation")
                imputer = SimpleImputer(strategy='median')
                X = imputer.fit_transform(X)
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error creating feature arrays: {e}")
            return None, None
    
    def _get_ml_models(self) -> Dict[str, Any]:
        """Get available machine learning models."""
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(
                n_estimators=50, 
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=50,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'SVR': SVR(kernel='rbf', C=100, gamma=0.001)
        }
        
        # Filter models if specified
        if 'all' not in self.models_to_use:
            models = {name: model for name, model in models.items() 
                     if name in self.models_to_use}
        
        return models
    
    def _train_and_evaluate_models(self, X: np.ndarray, y: np.ndarray) -> List[Dict[str, Any]]:
        """Train and evaluate multiple ML models."""
        try:
            # Split data
            split_point = int(len(X) * (1 - self.validation_split))
            X_train, X_val = X[:split_point], X[split_point:]
            y_train, y_val = y[:split_point], y[split_point:]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            models = self._get_ml_models()
            results = []
            
            for name, model in models.items():
                try:
                    # Train model
                    model.fit(X_train_scaled, y_train)
                    
                    # Make predictions
                    val_pred = model.predict(X_val_scaled)
                    
                    # Calculate metrics
                    mse = mean_squared_error(y_val, val_pred)
                    mae = mean_absolute_error(y_val, val_pred)
                    r2 = r2_score(y_val, val_pred)
                    
                    # Make future prediction
                    last_features = scaler.transform([X[-1]])
                    future_pred = model.predict(last_features)[0]
                    
                    results.append({
                        'name': name,
                        'model': model,
                        'scaler': scaler,
                        'mse': mse,
                        'mae': mae,
                        'r2': r2,
                        'prediction': future_pred
                    })
                    
                    self.logger.debug(f"{name} - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
                    
                except Exception as e:
                    self.logger.error(f"Error training {name}: {e}")
                    continue
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error training models: {e}")
            return []
    
    def _make_future_prediction(self, best_result: Dict[str, Any], X: np.ndarray) -> float:
        """Make future prediction using the best model."""
        try:
            return best_result['prediction']
        except Exception as e:
            self.logger.error(f"Error making future prediction: {e}")
            return self.get_current_price()


# Standalone function for direct usage
def predict_price_ml(symbol: str, days: int = 30, models: List[str] = None) -> Dict[str, Any]:
    """
    Standalone function to predict price using machine learning.
    
    Args:
        symbol: Stock symbol
        days: Prediction period in days
        models: List of model names to use (None = all)
        
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
        kwargs = {'prediction_days': days}
        if models:
            kwargs['models_to_use'] = models
            
        model = MachineLearningModel(symbol, historical_data, **kwargs)
        return model.predict()
        
    except Exception as e:
        return {"error": f"Machine learning prediction failed: {e}"}
