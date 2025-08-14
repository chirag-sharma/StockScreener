#!/usr/bin/env python3
"""
Price Prediction CLI Tool
========================

Standalone tool for testing price prediction functionality.

Usage:
    python scripts/predict_prices.py RELIANCE.NS
    python scripts/predict_prices.py --batch RELIANCE.NS TCS.NS INFY.NS
    python scripts/predict_prices.py --comprehensive RELIANCE.NS
"""

import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from stock_screener.services.pricePrediction import PricePredictionService, get_batch_predictions
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def format_prediction_output(prediction: dict) -> str:
    """Format prediction output for display"""
    if "error" in prediction:
        return f"❌ Error: {prediction['error']}"
    
    output = []
    output.append(f"📈 Stock: {prediction.get('symbol', 'N/A')}")
    output.append(f"💰 Current Price: ₹{prediction.get('current_price', 'N/A')}")
    output.append(f"🎯 Predicted Price: ₹{prediction.get('predicted_price', 'N/A')}")
    output.append(f"📊 Price Change: {prediction.get('price_change_percent', 'N/A')}%")
    output.append(f"🎪 Confidence: {prediction.get('confidence', 0)*100:.1f}%")
    output.append(f"📋 Recommendation: {prediction.get('recommendation', 'N/A')}")
    output.append(f"🔍 Method: {prediction.get('method', 'N/A')}")
    
    return "\n".join(output)


def format_comprehensive_output(predictions: dict) -> str:
    """Format comprehensive predictions output"""
    if "error" in predictions:
        return f"❌ Error: {predictions['error']}"
    
    output = []
    output.append("=" * 80)
    output.append(f"🚀 COMPREHENSIVE PRICE PREDICTIONS FOR {predictions['symbol']}")
    output.append("=" * 80)
    output.append(f"📅 Current Price: ₹{predictions['current_price']}")
    output.append(f"🎯 Prediction Date: {predictions['prediction_date']} ({predictions['prediction_days']} days)")
    output.append("")
    
    # Ensemble Result
    if 'ensemble' in predictions and 'predicted_price' in predictions['ensemble']:
        ensemble = predictions['ensemble']
        output.append("🎪 ENSEMBLE PREDICTION (Recommended)")
        output.append("-" * 40)
        output.append(f"Predicted Price: ₹{ensemble['predicted_price']}")
        output.append(f"Confidence: {ensemble['confidence']*100:.1f}%")
        output.append(f"Recommendation: {ensemble['recommendation']}")
        if 'price_range' in ensemble:
            output.append(f"Price Range: ₹{ensemble['price_range']['low']} - ₹{ensemble['price_range']['high']}")
        output.append(f"Methods Used: {ensemble['methods_used']}")
        output.append("")
    
    # Individual Methods
    output.append("📊 INDIVIDUAL METHOD PREDICTIONS")
    output.append("-" * 40)
    
    for method_name, result in predictions.get('methods', {}).items():
        if isinstance(result, dict) and 'predicted_price' in result and 'error' not in result:
            output.append(f"{method_name.title().replace('_', ' ')}: ₹{result['predicted_price']} "
                         f"(Confidence: {result.get('confidence', 0)*100:.1f}%)")
        elif 'error' in result:
            output.append(f"{method_name.title().replace('_', ' ')}: ❌ {result['error']}")
    
    # Risk Assessment
    if 'risk_assessment' in predictions:
        risk = predictions['risk_assessment']
        output.append("")
        output.append("⚠️  RISK ASSESSMENT")
        output.append("-" * 40)
        if 'error' not in risk:
            output.append(f"Risk Level: {risk.get('risk_level', 'N/A')}")
            output.append(f"Volatility (30d): {risk.get('volatility_30d', 'N/A')}%")
            output.append(f"Max Drawdown: {risk.get('max_drawdown', 'N/A')}%")
    
    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(description="Stock Price Prediction Tool")
    parser.add_argument('symbols', nargs='*', help="Stock symbols (e.g., RELIANCE.NS)")
    parser.add_argument('--batch', action='store_true', help="Batch prediction mode")
    parser.add_argument('--comprehensive', action='store_true', help="Comprehensive analysis mode")
    parser.add_argument('--days', type=int, default=30, help="Prediction days (default: 30)")
    parser.add_argument('--json', action='store_true', help="Output in JSON format")
    
    args = parser.parse_args()
    
    if not args.symbols:
        parser.print_help()
        print("\nExample usage:")
        print("  python scripts/predict_prices.py RELIANCE.NS")
        print("  python scripts/predict_prices.py --batch RELIANCE.NS TCS.NS")
        print("  python scripts/predict_prices.py --comprehensive RELIANCE.NS")
        return
    
    try:
        if args.batch:
            print("🔄 Running batch price predictions...")
            results = get_batch_predictions(args.symbols, args.days)
            
            if args.json:
                print(json.dumps(results, indent=2))
            else:
                for symbol, prediction in results.items():
                    print(f"\n{'='*60}")
                    print(format_prediction_output(prediction))
                    
        elif args.comprehensive:
            if len(args.symbols) > 1:
                print("⚠️  Comprehensive mode only supports one symbol at a time")
                return
                
            symbol = args.symbols[0]
            print(f"🔄 Running comprehensive analysis for {symbol}...")
            
            predictor = PricePredictionService(symbol, args.days)
            predictions = predictor.get_comprehensive_predictions()
            
            if args.json:
                print(json.dumps(predictions, indent=2))
            else:
                print(format_comprehensive_output(predictions))
                
        else:
            # Quick prediction mode
            for symbol in args.symbols:
                print(f"\n🔄 Getting quick prediction for {symbol}...")
                
                predictor = PricePredictionService(symbol, args.days)
                prediction = predictor.get_quick_prediction()
                
                if args.json:
                    print(json.dumps(prediction, indent=2))
                else:
                    print(format_prediction_output(prediction))
                    
    except KeyboardInterrupt:
        print("\n⏹️  Prediction cancelled by user")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
