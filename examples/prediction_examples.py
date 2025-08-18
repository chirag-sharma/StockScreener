"""
Price Prediction Examples
========================

Examples showing how to use the modular prediction system.
Can be run independently or imported into other scripts.
"""

import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stock_screener.prediction_models import (
    PricePredictionOrchestrator,
    TechnicalAnalysisModel,
    FundamentalAnalysisModel,
    predict_stock_price,
    compare_prediction_models
)


def example_orchestrator_usage():
    """Example of using the PricePredictionOrchestrator."""
    print("=== Orchestrator Example ===")
    
    # Comprehensive prediction using all 6 models
    result = predict_stock_price('RELIANCE.NS', days=30)
    
    if 'error' not in result:
        print(f"Symbol: {result['symbol']}")
        print(f"Current Price: ₹{result['current_price']}")
        print(f"Predicted Price: ₹{result['predicted_price']}")
        print(f"Expected Change: {result['price_change_pct']:.2f}%")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Market Signal: {result['market_signal']}")
        print(f"Models Used: {result['models_used']}/6")
    else:
        print(f"Error: {result['error']}")


def example_individual_model_usage():
    """Example of using individual models."""
    print("\n=== Individual Model Examples ===")
    
    import yfinance as yf
    
    # Fetch data
    ticker = yf.Ticker('RELIANCE.NS')
    historical_data = ticker.history(period='6mo')
    
    if not historical_data.empty:
        # Technical Analysis Model
        print("\n--- Technical Analysis ---")
        tech_model = TechnicalAnalysisModel('RELIANCE.NS', historical_data)
        tech_result = tech_model.predict()
        
        print(f"Technical Prediction: ₹{tech_result['predicted_price']}")
        print(f"Confidence: {tech_result['confidence']:.2f}")
        print(f"Signals: {tech_result.get('signals', {})}")
        
        # Fundamental Analysis Model  
        print("\n--- Fundamental Analysis ---")
        fund_model = FundamentalAnalysisModel('RELIANCE.NS', historical_data)
        fund_result = fund_model.predict()
        
        print(f"Fundamental Prediction: ₹{fund_result['predicted_price']}")
        print(f"Confidence: {fund_result['confidence']:.2f}")
        print(f"Valuation Methods: {fund_result.get('valuation_methods', [])}")
    
    else:
        print("Could not fetch historical data")


def example_model_comparison():
    """Example of comparing all models."""
    print("\n=== Model Comparison Example ===")
    
    comparison = compare_prediction_models('RELIANCE.NS', days=30)
    
    if 'error' not in comparison:
        print(f"Current Price: ₹{comparison['current_price']}")
        print("\nModel Predictions:")
        print("-" * 50)
        
        for model_name, prediction in comparison['prediction_comparison'].items():
            if 'error' not in prediction:
                print(f"{model_name.title():20} | ₹{prediction['predicted_price']:8.2f} | {prediction['price_change_pct']:6.1f}% | Conf: {prediction['confidence']:.2f}")
            else:
                print(f"{model_name.title():20} | Error: {prediction['error']}")
    else:
        print(f"Error: {comparison['error']}")


def example_standalone_functions():
    """Example of using standalone prediction functions."""
    print("\n=== Standalone Functions Example ===")
    
    from stock_screener.prediction_models import (
        predict_price_technical,
        predict_price_ml,
        predict_price_timeseries
    )
    
    # Technical analysis
    tech_result = predict_price_technical('RELIANCE.NS', days=15)
    print(f"Technical (15d): ₹{tech_result.get('predicted_price', 'N/A')}")
    
    # Machine learning
    ml_result = predict_price_ml('RELIANCE.NS', days=15)
    print(f"ML (15d): ₹{ml_result.get('predicted_price', 'N/A')}")
    
    # Time series
    ts_result = predict_price_timeseries('RELIANCE.NS', days=15)
    print(f"Time Series (15d): ₹{ts_result.get('predicted_price', 'N/A')}")


def main():
    """Run all examples."""
    print("Stock Price Prediction - Modular System Examples")
    print("=" * 55)
    
    try:
        # Run examples
        example_orchestrator_usage()
        example_individual_model_usage() 
        example_model_comparison()
        example_standalone_functions()
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have the required dependencies installed:")
        print("pip install yfinance pandas numpy scikit-learn statsmodels talib")


if __name__ == "__main__":
    main()
