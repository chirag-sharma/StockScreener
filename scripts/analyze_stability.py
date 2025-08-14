#!/usr/bin/env python3
"""
Price Prediction Stability Analysis Tool
========================================

Development and QA tool for testing prediction model consistency.

Features:
- Tests prediction stability across multiple runs
- Analyzes variation in price predictions and confidence scores  
- Identifies sources of prediction inconsistency
- Provides recommendations for model improvement

Usage: python scripts/analyze_stability.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from stock_screener.services.pricePrediction import PricePredictionService
import pandas as pd
import numpy as np
from datetime import datetime
import time

def test_prediction_stability():
    """Test prediction consistency and identify variation sources"""
    
    print("🔍 PREDICTION STABILITY ANALYSIS")
    print("=" * 60)
    
    symbol = "RELIANCE.NS"
    runs = 5
    results = []
    
    for i in range(runs):
        print(f"\n📊 Run {i+1}/{runs}:")
        
        # Create predictor
        predictor = PricePredictionService(symbol, prediction_days=30)
        
        # Get current price (this is the main variable)
        current_price = predictor.current_price
        print(f"   Current Price: ₹{current_price:.2f}")
        
        # Get comprehensive predictions
        result = predictor.get_comprehensive_predictions()
        
        if "ensemble" in result:
            predicted_price = result["ensemble"]["predicted_price"]
            confidence = result["ensemble"]["confidence"]
            methods_count = len(result["methods"])
            
            results.append({
                'run': i+1,
                'current_price': current_price,
                'predicted_price': predicted_price,
                'confidence': confidence,
                'methods': methods_count,
                'timestamp': datetime.now().strftime("%H:%M:%S")
            })
            
            print(f"   Predicted: ₹{predicted_price:.2f}")
            print(f"   Confidence: {confidence:.2f}")
            print(f"   Methods: {methods_count}")
        
        # Small delay between runs
        if i < runs - 1:
            time.sleep(1)
    
    # Analysis
    print("\n" + "=" * 60)
    print("📈 STABILITY ANALYSIS:")
    
    if results:
        df = pd.DataFrame(results)
        
        # Current price variation
        current_std = df['current_price'].std()
        current_mean = df['current_price'].mean()
        
        # Prediction variation  
        pred_std = df['predicted_price'].std()
        pred_mean = df['predicted_price'].mean()
        
        # Confidence variation
        conf_std = df['confidence'].std()
        conf_mean = df['confidence'].mean()
        
        print(f"\n🎯 Current Price:")
        print(f"   Mean: ₹{current_mean:.2f}")
        print(f"   Std Dev: ₹{current_std:.4f}")
        print(f"   Coefficient of Variation: {(current_std/current_mean)*100:.4f}%")
        
        print(f"\n🔮 Predicted Price:")
        print(f"   Mean: ₹{pred_mean:.2f}")
        print(f"   Std Dev: ₹{pred_std:.4f}")
        print(f"   Coefficient of Variation: {(pred_std/pred_mean)*100:.4f}%")
        
        print(f"\n📊 Confidence Score:")
        print(f"   Mean: {conf_mean:.3f}")
        print(f"   Std Dev: {conf_std:.4f}")
        
        print(f"\n📋 Detailed Results:")
        for _, row in df.iterrows():
            print(f"   Run {int(row['run'])}: Current=₹{row['current_price']:.2f}, "
                  f"Predicted=₹{row['predicted_price']:.2f}, "
                  f"Confidence={row['confidence']:.2f}, "
                  f"Time={row['timestamp']}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        if current_std > 1.0:
            print("   ⚠️  Current price is highly volatile (market hours)")
            print("   💡 Run predictions after market close for stability")
        else:
            print("   ✅ Current price is stable (market closed/low volatility)")
        
        if pred_std / pred_mean > 0.05:  # >5% variation
            print("   ⚠️  Predictions show high variation")
            print("   💡 Consider using Quick Prediction for more stability")
        else:
            print("   ✅ Predictions are consistent")
        
        if conf_std > 0.1:
            print("   ⚠️  Confidence scores vary significantly")
            print("   💡 Model performance is inconsistent")
        else:
            print("   ✅ Confidence scores are stable")

if __name__ == "__main__":
    test_prediction_stability()
