# Price Prediction Feature - User Guide

## 🎯 Overview

The Stock Screener now includes advanced **Price Prediction** capabilities that provide 30-day forward price targets using multiple analytical methods. This feature is seamlessly integrated into both the unified screener workflow and available as a standalone tool.

## 🚀 Key Features

### Multi-Method Price Prediction
1. **Technical Analysis**: Moving averages, trend analysis, support/resistance
2. **Fundamental Analysis**: DCF approximation, P/E valuation, Graham formula
3. **Machine Learning**: Linear regression and random forest models (when scikit-learn available)
4. **Time Series**: ARIMA modeling (when statsmodels available)  
5. **Pattern Recognition**: Price pattern and momentum analysis
6. **Volume Analysis**: Volume-price relationship analysis
7. **Ensemble Method**: Combines all methods with weighted confidence scoring

### AI-Enhanced Integration
- Price predictions are automatically integrated into AI analysis
- Enhanced AI prompts include price prediction context
- Target price recommendations consider both fundamental and technical factors

## 📊 Output Columns Added

The screener now includes these additional columns in Excel reports:

| Column | Description |
|--------|-------------|
| **Predicted Price (30d)** | 30-day forward price prediction |
| **Price Change %** | Expected percentage change |
| **Prediction Confidence** | Confidence score (0-100%) |
| **Prediction Method** | Method used for prediction |
| **Target Price** | AI-enhanced target price (combines prediction + analysis) |

## 🛠️ Usage Methods

### 1. Integrated with Unified Screener (Recommended)

```bash
# Price predictions are automatically included in standard workflow
python scripts/run_screener.py

# Price prediction columns will appear in comprehensive_analysis.xlsx
```

### 2. Standalone Price Prediction Tool

#### Quick Prediction (Single Stock)
```bash
python scripts/predict_prices.py RELIANCE.NS
```

#### Batch Predictions (Multiple Stocks)
```bash
python scripts/predict_prices.py --batch RELIANCE.NS TCS.NS INFY.NS
```

#### Comprehensive Analysis (All Methods)
```bash
python scripts/predict_prices.py --comprehensive RELIANCE.NS
```

#### Custom Prediction Period
```bash
python scripts/predict_prices.py --days 60 RELIANCE.NS
```

#### JSON Output (for API integration)
```bash
python scripts/predict_prices.py --json RELIANCE.NS
```

## 📈 Example Output

### Quick Prediction
```
📈 Stock: RELIANCE.NS
💰 Current Price: ₹1367.80
🎯 Predicted Price: ₹2801.15
📊 Price Change: 104.79%
🎪 Confidence: 65.0%
📋 Recommendation: Strong Buy
🔍 Method: Quick Prediction (Technical + Fundamental)
```

### Comprehensive Analysis
```
🚀 COMPREHENSIVE PRICE PREDICTIONS FOR RELIANCE.NS
📅 Current Price: ₹1367.80
🎯 Prediction Date: 2025-09-08 (30 days)

🎪 ENSEMBLE PREDICTION (Recommended)
Predicted Price: ₹2881.31
Confidence: 37.0%
Recommendation: Strong Buy
Price Range: ₹1287.18 - ₹4475.44
Methods Used: 4

📊 INDIVIDUAL METHOD PREDICTIONS
Technical: ₹1299.41 (60.0%)
Fundamental: ₹4088.35 (70.0%)
Pattern: ₹1364.92 (54.0%)
Volume: ₹1367.50 (30.0%)

⚠️ RISK ASSESSMENT
Risk Level: Low
Volatility (30d): 17.56%
Max Drawdown: 27.18%
```

## ⚙️ Configuration

### Optional Dependencies for Enhanced Features

```bash
# For machine learning predictions
pip install scikit-learn

# For time series analysis
pip install statsmodels

# For advanced technical indicators
pip install talib
```

### Performance Notes
- **Quick predictions** (Technical + Fundamental): ~2-3 seconds per stock
- **Comprehensive analysis** (All methods): ~5-10 seconds per stock
- **Batch predictions**: Parallel processing for multiple stocks

## 🎨 Integration with AI Analysis

### Enhanced AI Prompts
The AI analysis now receives price prediction context:

```
PRICE PREDICTION ANALYSIS:
Current Price: ₹1367.80
Predicted Price (30 days): ₹2881.31
Expected Change: 110.6%
Prediction Confidence: 37.0%

ENHANCED ANALYSIS REQUIRED:
1. How the predicted price movement aligns with fundamental analysis
2. Whether current valuation supports the predicted direction
3. Risk factors that could impact prediction accuracy
4. Timeline considerations for achieving target price
```

### Improved Target Price Accuracy
- Combines fundamental valuation with technical momentum
- Considers recent news and market developments
- Provides confidence-weighted recommendations

## 📋 Prediction Methods Explained

### 1. Technical Analysis
- **Indicators**: SMA, EMA, MACD, RSI, Bollinger Bands
- **Trend Analysis**: Moving average crossovers and momentum
- **Support/Resistance**: Key price levels from historical data

### 2. Fundamental Analysis
- **P/E Valuation**: Using sector-average P/E ratios
- **P/B Valuation**: Book value multiple approach
- **DCF Approximation**: Simplified discounted cash flow
- **Graham Formula**: Benjamin Graham's growth formula

### 3. Machine Learning (Optional)
- **Features**: Price, volume, technical indicators, ratios
- **Models**: Linear Regression, Random Forest
- **Validation**: Time series cross-validation

### 4. Time Series Analysis (Optional)
- **ARIMA Models**: Auto-regressive integrated moving average
- **Seasonal Patterns**: Identifying cyclical price movements
- **Forecast Intervals**: Prediction confidence bands

### 5. Ensemble Method
- **Weighted Average**: Confidence-based combination
- **Risk Adjustment**: Volatility-based confidence scaling
- **Consensus Prediction**: Most reliable combined forecast

## ⚠️ Important Disclaimers

1. **Educational Purpose**: Predictions are for educational/research purposes only
2. **Not Investment Advice**: Always consult financial advisors for investment decisions
3. **Market Risk**: Stock markets are inherently unpredictable
4. **Accuracy Limitations**: Past performance doesn't guarantee future results
5. **External Factors**: Predictions don't account for unexpected market events

## 🚀 Advanced Usage

### Custom Prediction Models
You can extend the `PricePredictionService` class to add your own prediction methods:

```python
from stock_screener.services.pricePrediction import PricePredictionService

class CustomPredictor(PricePredictionService):
    def _custom_prediction_method(self):
        # Your custom prediction logic
        pass
```

### API Integration
Use JSON output for integrating with other applications:

```python
import json
from stock_screener.services.pricePrediction import get_batch_predictions

symbols = ['RELIANCE.NS', 'TCS.NS']
results = get_batch_predictions(symbols)
print(json.dumps(results, indent=2))
```

## 📊 Performance Metrics

The system tracks prediction accuracy and provides confidence scores based on:
- Historical model performance
- Market volatility
- Data availability
- Method consensus

Higher confidence scores indicate more reliable predictions based on multiple confirming methods and stable market conditions.

---

**💡 Pro Tip**: Use the comprehensive analysis mode for important investment decisions, as it provides the most thorough evaluation using all available methods and includes detailed risk assessment.
