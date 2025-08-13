#!/usr/bin/env python3
"""
Test script for multi-period price predictions (6-12 months)
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from stock_screener.services.pricePrediction import PricePredictionService, get_multi_period_batch_predictions
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_single_stock_multi_period():
    """Test multi-period predictions for a single stock"""
    print("\n" + "="*80)
    print("🔮 TESTING MULTI-PERIOD PRICE PREDICTIONS (6-12 MONTHS)")
    print("="*80)
    
    # Test with TCS
    symbol = "TCS.NS"
    print(f"\n📊 Analyzing: {symbol}")
    print("-" * 50)
    
    try:
        # Initialize predictor
        predictor = PricePredictionService(symbol)
        
        # Get simplified multi-period predictions
        predictions = predictor.get_simplified_multi_period_predictions()
        
        if "error" in predictions:
            print(f"❌ Error: {predictions['error']}")
            return
        
        # Display current price
        print(f"💰 Current Price: ₹{predictions['current_price']:.2f}")
        print(f"📅 Analysis Date: {predictions['analysis_date']}")
        print()
        
        # Display predictions for each period
        print("🎯 MULTI-PERIOD PREDICTIONS:")
        print("-" * 50)
        
        for period_name, pred in predictions['predictions'].items():
            months = pred['months']
            price = pred['predicted_price']
            confidence = pred['confidence']
            growth = pred['growth_percent']
            date = pred['prediction_date']
            
            growth_icon = "📈" if growth > 0 else "📉" if growth < 0 else "➡️"
            confidence_icon = "🟢" if confidence > 0.6 else "🟡" if confidence > 0.4 else "🔴"
            
            print(f"{months:2d} Months: ₹{price:8.2f} {growth_icon} {growth:+6.2f}% {confidence_icon} ({confidence:.2f}) | {date}")
        
        # Display summary if available
        if "summary" in predictions and "price_range" in predictions["summary"]:
            summary = predictions["summary"]
            print("\n📊 SUMMARY STATISTICS:")
            print("-" * 30)
            print(f"Price Range: ₹{summary['price_range']['min']:.2f} - ₹{summary['price_range']['max']:.2f}")
            print(f"Average Price: ₹{summary['price_range']['average']:.2f}")
            
            if "growth_projection" in summary:
                growth = summary["growth_projection"]
                print(f"6-Month Growth: {growth['6_month_growth']:+.2f}%")
                print(f"12-Month Growth: {growth['12_month_growth']:+.2f}%")
        
        print("\n✅ Multi-period prediction completed successfully!")
        
    except Exception as e:
        logger.error(f"Error testing multi-period predictions: {e}")
        print(f"❌ Error: {e}")

def test_multiple_stocks_multi_period():
    """Test multi-period predictions for multiple stocks"""
    print("\n" + "="*80)
    print("🚀 TESTING BATCH MULTI-PERIOD PREDICTIONS")
    print("="*80)
    
    # Test with a few IT stocks
    symbols = ["TCS.NS", "INFY.NS", "WIPRO.NS"]
    
    try:
        results = get_multi_period_batch_predictions(symbols)
        
        for symbol, prediction in results.items():
            print(f"\n📊 {symbol}")
            print("-" * 40)
            
            if "error" in prediction:
                print(f"❌ Error: {prediction['error']}")
                continue
            
            current_price = prediction.get('current_price', 0)
            print(f"Current: ₹{current_price:.2f}")
            
            # Show 6-month and 12-month predictions
            predictions_data = prediction.get('predictions', {})
            if '6_months' in predictions_data and '12_months' in predictions_data:
                six_month = predictions_data['6_months']
                twelve_month = predictions_data['12_months']
                
                print(f"6M:  ₹{six_month['predicted_price']:8.2f} ({six_month['growth_percent']:+6.2f}%)")
                print(f"12M: ₹{twelve_month['predicted_price']:8.2f} ({twelve_month['growth_percent']:+6.2f}%)")
        
        print("\n✅ Batch multi-period predictions completed!")
        
    except Exception as e:
        logger.error(f"Error in batch multi-period predictions: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🧪 MULTI-PERIOD PRICE PREDICTION TEST SUITE")
    print("Testing 6-12 month predictions with 1-month intervals")
    
    # Test single stock
    test_single_stock_multi_period()
    
    # Test multiple stocks
    test_multiple_stocks_multi_period()
    
    print("\n🎉 All tests completed!")
