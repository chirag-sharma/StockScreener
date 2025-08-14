#!/usr/bin/env python3
"""
Integration test for the weighted scoring system
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from services.stockAnalyzer import StockAnalyzer

def test_weighted_scoring_integration():
    """Test the new weighted scoring system with a few known stocks"""
    test_symbols = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS']  # Popular Indian stocks
    
    print("Testing New Weighted Value Investing Scoring System")
    print("=" * 60)
    
    all_passed = True
    
    for symbol in test_symbols:
        print(f"\n🔍 Analyzing {symbol}...")
        
        try:
            analyzer = StockAnalyzer(symbol)
            analyzer.fetch_data()
            result = analyzer.analyze()
            
            if result:
                score = result.get('Value Score', 'N/A')
                recommendation = result.get('Investment Recommendation', 'N/A')
                
                # Basic validation checks
                if isinstance(score, (int, float)) and 0 <= score <= 100:
                    print(f"✅ {symbol}: Score = {score}")
                    print(f"   Recommendation: {recommendation}")
                    print(f"   PE Ratio: {result.get('PE Ratio', 'N/A')}")
                    print(f"   ROE: {result.get('ROE', 'N/A')}%")
                    print(f"   Debt/Equity: {result.get('Debt/Equity', 'N/A')}")
                    print(f"   Current Ratio: {result.get('Current Ratio', 'N/A')}")
                else:
                    print(f"❌ {symbol}: Invalid score - {score}")
                    all_passed = False
            else:
                print(f"⚠️ {symbol}: No data available")
        except Exception as e:
            print(f"❌ {symbol}: Error - {e}")
            all_passed = False
    
    print(f"\n{'='*60}")
    if all_passed:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
    
    return all_passed

if __name__ == "__main__":
    test_weighted_scoring_integration()
