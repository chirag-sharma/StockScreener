#!/usr/bin/env python3
"""
Detailed breakdown of the weighted value investing scoring system
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from services.stockAnalyzer import StockAnalyzer
import json

def detailed_analysis():
    """Show detailed breakdown of how the weighted scoring works"""
    
    # Test with one representative stock
    symbol = "ICICIBANK.NS"  # This had the highest score from our test
    
    print("🔍 DETAILED WEIGHTED VALUE INVESTING ANALYSIS")
    print("=" * 60)
    print(f"Stock: {symbol}")
    print("=" * 60)
    
    analyzer = StockAnalyzer(symbol)
    analyzer.fetch_data()
    result = analyzer.analyze()
    
    if not result:
        print("❌ No data available")
        return False
    
    print(f"\n🎯 OVERALL SCORE: {result.get('Value Score', 'N/A')}/100")
    print(f"📊 RECOMMENDATION: {result.get('Investment Recommendation', 'N/A')}")
    
    print(f"\n{'='*60}")
    print("📈 FINANCIAL METRICS BREAKDOWN")
    print(f"{'='*60}")
    
    # Core metrics with their categories
    core_value_metrics = {
        'PE Ratio': (15.0, 'Lower is better'),
        'Price to Book': (10.0, 'Lower is better'),
        'EV/EBITDA': (10.0, 'Lower is better'),
        'Margin of Safety (%)': (5.0, 'Higher is better')
    }
    
    profitability_metrics = {
        'ROE': (10.0, 'Higher is better'),
        'Net Profit Margin': (8.0, 'Higher is better'),
        'Operating Margin': (7.0, 'Higher is better'),
        'Return on Assets (ROA)': (5.0, 'Higher is better')
    }
    
    financial_strength_metrics = {
        'Current Ratio': (6.0, 'Higher is better'),
        'Debt/Equity': (6.0, 'Lower is better'),
        'Quick Ratio': (4.0, 'Higher is better'),
        'Interest Coverage Ratio': (4.0, 'Higher is better')
    }
    
    growth_metrics = {
        'Free Cash Flow': (4.0, 'Higher is better'),
        'EPS Growth (%)': (3.0, 'Higher is better'),
        'Revenue Growth (%)': (3.0, 'Higher is better')
    }
    
    def print_category(title, metrics_dict, emoji):
        print(f"\n{emoji} {title}")
        print("-" * 50)
        total_weight = sum(weight for weight, _ in metrics_dict.values())
        
        for metric, (weight, direction) in metrics_dict.items():
            value = result.get(metric, 'N/A')
            pass_fail = result.get(f"{metric} Pass", 'N/A')
            
            # Calculate individual metric score
            try:
                individual_score = analyzer._score_metric(metric, value) if value != 'N/A' else 0
            except:
                individual_score = 0
            
            value_str = str(value) if value is not None else 'N/A'
            status_emoji = "✅" if pass_fail else "❌" if pass_fail is False else "⚪"
            
            print(f"  {status_emoji} {metric:<25}: {value_str:<12} (Weight: {weight}%) Score: {individual_score}/100")
            print(f"      Direction: {direction}")
        
        print(f"  📊 Category Total Weight: {total_weight}%")
    
    try:
        print_category("CORE VALUE METRICS (40%)", core_value_metrics, "💰")
        print_category("PROFITABILITY & QUALITY (30%)", profitability_metrics, "🏆")
        print_category("FINANCIAL STRENGTH (20%)", financial_strength_metrics, "🛡️")
        print_category("GROWTH & CASH GENERATION (10%)", growth_metrics, "📈")
        
        print(f"\n{'='*60}")
        print("🎯 VALUE INVESTING INTERPRETATION")
        print(f"{'='*60}")
        
        score = result.get('Value Score', 0)
        if score >= 90:
            print("🌟 EXCEPTIONAL VALUE: Strong fundamentals with significant margin of safety")
        elif score >= 70:
            print("✅ GOOD VALUE: Solid fundamentals with reasonable valuation")
        elif score >= 50:
            print("⚖️  FAIR VALUE: Mixed signals, requires careful analysis")
        elif score >= 30:
            print("⚠️  OVERVALUED: Limited upside, consider waiting for better entry")
        else:
            print("🚫 POOR VALUE: Weak fundamentals or significantly overvalued")
        
        print(f"\n{'='*60}")
        print("📚 VALUE INVESTING PRINCIPLES APPLIED")
        print(f"{'='*60}")
        print("✓ Warren Buffett: Focus on quality businesses (ROE, Profit Margins)")
        print("✓ Benjamin Graham: Margin of Safety & Asset Protection (P/B, Current Ratio)")
        print("✓ Peter Lynch: Reasonable Growth at Reasonable Price (PEG-like scoring)")
        print("✓ Joel Greenblatt: Quality + Value (ROE + Earnings Yield considerations)")
        
        return True
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return False

if __name__ == "__main__":
    success = detailed_analysis()
    sys.exit(0 if success else 1)
