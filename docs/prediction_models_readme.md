# Modular Price Prediction System

A comprehensive, modular price prediction system with 6 different prediction models that can be used independently or combined through an orchestrator.

## Architecture

```
stock_screener/prediction_models/
├── __init__.py                    # Package initialization and exports
├── base_model.py                  # Abstract base class for all models
├── technical_analysis_model.py    # Technical indicators and chart analysis
├── fundamental_analysis_model.py  # Financial metrics and valuation
├── machine_learning_model.py      # ML algorithms for price prediction  
├── time_series_model.py           # Time series analysis and forecasting
├── pattern_recognition_model.py   # Chart patterns and technical patterns
├── volume_analysis_model.py       # Volume-based analysis
└── prediction_orchestrator.py     # Coordinates all models
```

## Models Overview

### 1. Technical Analysis Model
- **Moving Averages**: SMA, EMA crossovers
- **Momentum Indicators**: MACD, RSI, Stochastic
- **Volatility**: Bollinger Bands, ATR
- **Support/Resistance**: Price levels and breakouts

### 2. Fundamental Analysis Model  
- **Valuation Ratios**: P/E, P/B, PEG
- **DCF Analysis**: Discounted Cash Flow valuation
- **Graham Formula**: Benjamin Graham's intrinsic value
- **Growth Analysis**: Revenue and earnings growth

### 3. Machine Learning Model
- **Linear Regression**: Simple linear trends
- **Ridge/Lasso**: Regularized regression
- **Random Forest**: Ensemble decision trees
- **Gradient Boosting**: XGBoost-style boosting
- **Support Vector Regression**: Non-linear relationships

### 4. Time Series Model
- **ARIMA**: Auto-regressive integrated moving average
- **Exponential Smoothing**: Trend and seasonality
- **Seasonal Decomposition**: Pattern identification
- **Trend Analysis**: Long-term direction

### 5. Pattern Recognition Model
- **Chart Patterns**: Triangles, flags, head & shoulders
- **Breakout Detection**: Support/resistance breaks
- **Moving Average Patterns**: Golden/death cross
- **Volume Confirmation**: Pattern validation

### 6. Volume Analysis Model
- **On-Balance Volume (OBV)**: Volume-price momentum
- **Volume Breakouts**: Unusual volume activity
- **Accumulation/Distribution**: Smart money flow
- **Price-Volume Correlation**: Relationship analysis

## Usage Examples

### Quick Start (Orchestrator)

```python
from stock_screener.prediction_models import predict_stock_price

# Get comprehensive prediction using all 6 models
result = predict_stock_price('RELIANCE.NS', days=30)

print(f"Predicted Price: ₹{result['predicted_price']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Market Signal: {result['market_signal']}")
```

### Individual Model Usage

```python
from stock_screener.prediction_models import TechnicalAnalysisModel
import yfinance as yf

# Fetch data
ticker = yf.Ticker('RELIANCE.NS')
data = ticker.history(period='6mo')

# Use specific model
model = TechnicalAnalysisModel('RELIANCE.NS', data)
prediction = model.predict()

print(f"Technical Analysis: ₹{prediction['predicted_price']}")
```

### Orchestrator Class

```python
from stock_screener.prediction_models import PricePredictionOrchestrator

# Create orchestrator
orchestrator = PricePredictionOrchestrator('RELIANCE.NS')

# Comprehensive prediction
result = orchestrator.predict_comprehensive(days=30)

# Individual model results
tech_result = orchestrator.get_individual_model_prediction('technical')
ml_result = orchestrator.get_individual_model_prediction('machine_learning')

# Compare all models
comparison = orchestrator.get_model_comparison()
```

### Standalone Functions

```python
from stock_screener.prediction_models import (
    predict_price_technical,
    predict_price_ml,
    predict_price_timeseries,
    compare_prediction_models
)

# Individual predictions
tech = predict_price_technical('RELIANCE.NS', days=15)
ml = predict_price_ml('RELIANCE.NS', days=15)
ts = predict_price_timeseries('RELIANCE.NS', days=15)

# Model comparison
comparison = compare_prediction_models('RELIANCE.NS')
```

## Configuration

### Model Weights (Orchestrator)

```python
# Custom model weights
weights = {
    'technical': 0.25,
    'fundamental': 0.20,
    'machine_learning': 0.25,
    'time_series': 0.15,
    'pattern_recognition': 0.10,
    'volume_analysis': 0.05
}

orchestrator = PricePredictionOrchestrator(
    'RELIANCE.NS', 
    model_weights=weights
)
```

### Model Parameters

```python
# Technical Analysis parameters
model = TechnicalAnalysisModel(
    'RELIANCE.NS', 
    data,
    short_window=12,    # Short MA period
    long_window=26,     # Long MA period
    rsi_period=14,      # RSI calculation period
    bb_period=20        # Bollinger Bands period
)

# Machine Learning parameters  
model = MachineLearningModel(
    'RELIANCE.NS',
    data, 
    models=['linear', 'random_forest', 'gradient_boosting'],
    feature_days=30,    # Historical days for features
    test_size=0.2       # Train/test split
)
```

## API Reference

### Base Model Interface

All models inherit from `BasePredictionModel`:

```python
class BasePredictionModel:
    def __init__(self, symbol: str, historical_data: pd.DataFrame, **kwargs)
    def validate_inputs(self) -> bool
    def predict(self) -> Dict[str, Any]  
    def get_confidence(self) -> float
    def get_current_price(self) -> float
    def get_support_resistance_levels(self) -> Dict[str, float]
```

### Prediction Result Format

```python
{
    "predicted_price": 2450.75,
    "current_price": 2380.50, 
    "price_change": 70.25,
    "price_change_pct": 2.95,
    "confidence": 0.72,
    "method": "Technical Analysis",
    "prediction_date": "2024-09-15",
    "additional_metrics": {
        # Model-specific metrics
    }
}
```

### Orchestrator Result Format

```python
{
    "predicted_price": 2445.30,
    "current_price": 2380.50,
    "price_change": 64.80,
    "price_change_pct": 2.72,
    "confidence": 0.68,
    "market_signal": "BUY",
    "method": "Ensemble (6 Models)",
    "models_used": 6,
    "individual_predictions": {
        "technical": {...},
        "fundamental": {...},
        # ... other models
    },
    "prediction_statistics": {
        "mean_prediction": 2445.30,
        "std_deviation": 45.20,
        "prediction_range": 120.50,
        # ... more stats
    },
    "risk_assessment": {
        "risk_level": "MEDIUM",
        "prediction_volatility": 0.089,
        "recommendation": "Moderate risk - maintain normal position sizing"
    }
}
```

## Integration with Existing Code

### Replace Original PricePrediction Service

```python
# Old way
from stock_screener.services.pricePrediction import PricePrediction
predictor = PricePrediction('RELIANCE.NS')
result = predictor.predict_price()

# New way (backward compatible)
from stock_screener.prediction_models import PricePredictionOrchestrator
orchestrator = PricePredictionOrchestrator('RELIANCE.NS')
result = orchestrator.predict_price()  # Same interface
```

### Dashboard Integration

```python
# In dashboard.py
from stock_screener.prediction_models import predict_stock_price

def get_price_prediction(symbol):
    """Get price prediction for dashboard display."""
    result = predict_stock_price(symbol, days=365)  # 12-month prediction
    
    if 'error' not in result:
        return {
            'target_price': result['predicted_price'],
            'upside_pct': result['price_change_pct'],
            'confidence': result['confidence'],
            'signal': result['market_signal']
        }
    return None
```

## Dependencies

```bash
pip install pandas numpy yfinance scikit-learn statsmodels talib
```

## Error Handling

All models include comprehensive error handling:

```python
result = predict_stock_price('INVALID.SYMBOL')
if 'error' in result:
    print(f"Prediction failed: {result['error']}")
else:
    print(f"Prediction: {result['predicted_price']}")
```

## Testing

```bash
# Run the example script
python examples/prediction_examples.py

# Test individual models
python -c "
from stock_screener.prediction_models import predict_price_technical
result = predict_price_technical('RELIANCE.NS')
print(f'Technical Analysis: {result}')
"
```

## Performance Notes

- **Data Requirements**: Each model needs different amounts of historical data
- **Computation Time**: ML models take longer; technical analysis is fastest
- **Memory Usage**: Orchestrator loads all models; use individual models for memory efficiency
- **Caching**: Historical data is fetched once per orchestrator instance

## Extensibility

To add a new prediction model:

1. Inherit from `BasePredictionModel`
2. Implement required methods (`predict`, `get_confidence`)
3. Add to orchestrator model list
4. Update `__init__.py` exports

```python
class MyCustomModel(BasePredictionModel):
    def predict(self) -> Dict[str, Any]:
        # Your prediction logic
        return {
            "predicted_price": calculated_price,
            "confidence": confidence_score,
            "method": "My Custom Method"
        }
    
    def get_confidence(self) -> float:
        # Your confidence calculation
        return confidence_value
```
