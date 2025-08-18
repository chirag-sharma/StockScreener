#!/usr/bin/env python3
"""
Debug script to trace monthly prediction data flow
"""

import sys
import os
sys.path.append('/Users/chirag/PycharmProjects/StockScreener')

from stock_screener.prediction_models import PricePredictionOrchestrator
from stock_screener.core.analyzer import DetailedAnalyzer
import json

def test_monthly_prediction_flow():
    """Test the complete flow of monthly predictions"""
    
    print("🔍 DEBUGGING MONTHLY PREDICTION FLOW")
    print("=" * 60)
    
    # Test with one stock
    symbol = "COALINDIA.NS"
    
    # Step 1: Test prediction service directly
    print(f"\n📊 STEP 1: Testing PricePredictionService for {symbol}")
    print("-" * 50)
    
    try:
        orchestrator = PricePredictionOrchestrator(symbol)
        multi_period = {
            '7d': orchestrator.predict_comprehensive(target_days=7),
            '30d': orchestrator.predict_comprehensive(target_days=30),
            '90d': orchestrator.predict_comprehensive(target_days=90),
            '365d': orchestrator.predict_comprehensive(target_days=365)
        }
        
        print(f"✅ Prediction service data structure:")
        for period, data in multi_period.items():
            if 'predicted_price' in data:
                predicted_price = data['predicted_price']
                print(f"   {period}: ₹{predicted_price}")
            else:
                print(f"   {period}: Error - {data.get('error', 'Unknown')}")
            
    except Exception as e:
        print(f"❌ Error in prediction service: {e}")
        return
    
    # Step 2: Test analyzer's price prediction extraction
    print(f"\n🧠 STEP 2: Testing StockAnalyzer price prediction extraction")
    print("-" * 60)
    
    try:
        analyzer = DetailedAnalyzer()
        price_prediction = analyzer._get_price_prediction(symbol)
        
        print("✅ Analyzer price prediction result:")
        print(f"   Main predicted price: {price_prediction.get('predicted_price', 'N/A')}")
        print(f"   Method: {price_prediction.get('method', 'N/A')}")
        
        multi_period_preds = price_prediction.get('multi_period_predictions', {})
        print(f"   Multi-period predictions found: {len(multi_period_preds)}")
        
        for period, data in multi_period_preds.items():
            if isinstance(data, dict):
                predicted_price = data.get('predicted_price', 'N/A')
                print(f"     {period}: ₹{predicted_price}")
            else:
                print(f"     {period}: {data} (unexpected format)")
                
    except Exception as e:
        print(f"❌ Error in analyzer: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Test full AI analysis integration
    print(f"\n🤖 STEP 3: Testing AI analysis integration")
    print("-" * 50)
    
    try:
        # Mock stock data for the analyzer
        current_price = orchestrator._get_current_price()
        stock_data = {
            'Symbol': symbol,
            'Current Price (₹)': current_price,
            'PE Ratio': 10.0,
            'Value Score': 60.0
        }
        
        # Get AI analysis (without actually calling OpenAI)
        ai_result = {
            'ai_sentiment': 'Neutral',
            'ai_recommendation': 'Hold',
            'investment_thesis': 'Test thesis',
            'value_score_ai': 5
        }
        
        # Enhance with price prediction (this is where the bug might be)
        price_prediction = analyzer._get_price_prediction(symbol)
        
        if price_prediction and not price_prediction.get('error'):
            ai_result['multi_period_predictions'] = price_prediction.get('multi_period_predictions', {})
            
        print("✅ Final AI result multi-period predictions:")
        final_multi_period = ai_result.get('multi_period_predictions', {})
        print(f"   Found {len(final_multi_period)} periods:")
        
        for period, data in final_multi_period.items():
            if isinstance(data, dict):
                predicted_price = data.get('predicted_price', 'N/A')
                print(f"     {period}: ₹{predicted_price}")
            else:
                print(f"     {period}: {data}")
                
    except Exception as e:
        print(f"❌ Error in AI integration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_monthly_prediction_flow()
