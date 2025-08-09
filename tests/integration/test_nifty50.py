#!/usr/bin/env python3
"""
Integration test for Nifty 50 analysis
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from services.stockAnalyzer import StockAnalyzer
from services.excelExporter import ExcelExporter
import json

def test_nifty50_integration():
    """Test with a sample of Nifty 50 stocks"""
    
    # Use absolute path to load Nifty 50 tickers
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    ticker_file = os.path.join(project_root, 'data', 'input', 'tickers', 'nifty_50.json')
    
    try:
        with open(ticker_file, 'r') as f:
            ticker_data = json.load(f)
            
        # Get tickers from the JSON structure
        if 'tickers' in ticker_data:
            symbols = ticker_data['tickers'][:3]  # Test with first 3 for faster testing
        else:
            symbols = ticker_data[:3]  # Fallback structure
            
        print("🎯 Testing Nifty 50 Integration")
        print("=" * 60)
        print(f"Testing with: {', '.join(symbols)}")
        
        all_results = []
        successful_analyses = 0
        
        for symbol in symbols:
            print(f"\n📊 Analyzing {symbol}...")
            
            try:
                analyzer = StockAnalyzer(symbol)
                analyzer.fetch_data()
                result = analyzer.analyze()
                
                if result:
                    all_results.append(result)
                    successful_analyses += 1
                    score = result.get('Value Score', 'N/A')
                    recommendation = result.get('Investment Recommendation', 'N/A')
                    pe_ratio = result.get('PE Ratio', 'N/A')
                    roe = result.get('ROE', 'N/A')
                    
                    print(f"   ✅ Score: {score}/100")
                    print(f"   📈 Recommendation: {recommendation}")
                    print(f"   💰 PE Ratio: {pe_ratio}")
                    print(f"   🏆 ROE: {roe}%")
                else:
                    print(f"   ❌ No data available for {symbol}")
            except Exception as e:
                print(f"   ❌ Error analyzing {symbol}: {e}")
        
        # Test Excel export
        if all_results:
            output_file = os.path.join(project_root, "data", "output", "test_analysis.xlsx")
            try:
                exporter = ExcelExporter(output_file)
                exporter.write_data(all_results)
                print(f"\n📋 Excel file created: {output_file}")
            except Exception as e:
                print(f"\n❌ Excel export failed: {e}")
                return False
        
        print(f"\n✅ Integration test completed: {successful_analyses}/{len(symbols)} stocks analyzed successfully")
        return successful_analyses > 0
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_nifty50_integration()
    sys.exit(0 if success else 1)
