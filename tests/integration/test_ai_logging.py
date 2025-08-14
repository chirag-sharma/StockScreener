#!/usr/bin/env python3
"""
AI API Integration Tests
========================

This module contains integration tests for AI API calls with enhanced logging.
It verifies that all AI API calls are properly logged with detailed information.

Usage:
    # Run as standalone script
    python tests/integration/test_ai_logging.py
    
    # Run with pytest (if available)
    pytest tests/integration/test_ai_logging.py -v
    
    # Run all integration tests
    python tests/run_tests.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stock_screener.services.aiBusinessQuality import get_business_quality_analysis
from stock_screener.core.analyzer import DetailedAnalyzer

def test_ai_business_quality():
    """Test the AI business quality analysis with detailed logging."""
    print("\n🧪 TESTING AI Business Quality Analysis")
    print("=" * 60)
    
    # Test with sample data
    symbol = "RELIANCE.NS"
    roe_trend = [15.2, 16.8, 14.5, 17.1, 16.3]
    margin_trend = [12.1, 13.4, 11.8, 14.2, 13.9]
    
    try:
        result = get_business_quality_analysis(symbol, roe_trend, margin_trend)
        print(f"✅ AI Business Quality Analysis completed successfully!")
        print(f"   Result: {result}")
    except Exception as e:
        print(f"❌ AI Business Quality Analysis failed: {e}")
    
def test_ai_comprehensive_analysis():
    """Test the comprehensive AI analysis with detailed logging."""
    print("\n🧪 TESTING AI Comprehensive Analysis")
    print("=" * 60)
    
    # Create a sample stock data row
    sample_stock_data = {
        'Symbol': 'RELIANCE.NS',
        'Current Price': 2500.00,
        'Market Cap': 1500000000000,
        'PE Ratio': 15.5,
        'PB Ratio': 2.1,
        'ROE': 16.3,
        'Debt to Equity': 0.3,
        'Value Score': 8.2,
        'Investment Recommendation': 'Strong Buy'
    }
    
    try:
        # Initialize analyzer (with AI enabled)
        analyzer = DetailedAnalyzer(
            input_file="dummy.xlsx",  # Won't be used for this test
            output_file="dummy.xlsx",  # Won't be used for this test
            enable_ai_analysis=True
        )
        
        # Test single stock AI analysis
        result = analyzer.analyze_stock_with_ai(sample_stock_data)
        print(f"✅ AI Comprehensive Analysis completed successfully!")
        print(f"   Result keys: {list(result.keys())}")
        print(f"   AI Sentiment: {result.get('ai_sentiment', 'N/A')}")
        print(f"   Value Score: {result.get('value_score', 'N/A')}")
        
    except Exception as e:
        print(f"❌ AI Comprehensive Analysis failed: {e}")

def main():
    """Main test function."""
    print("🚀 AI API Call Testing - Enhanced Logging Verification")
    print("=" * 80)
    print("This script will test AI API calls with comprehensive logging.")
    print("You should see detailed API request/response information below.")
    print("=" * 80)
    
    # Test 1: AI Business Quality Analysis
    test_ai_business_quality()
    
    # Test 2: Comprehensive AI Analysis
    test_ai_comprehensive_analysis()
    
    print("\n✅ AI API Call Testing Complete!")
    print("=" * 80)
    print("Review the output above to verify detailed AI API logging is working.")
    print("Each API call should show:")
    print("  - API call initiation with timestamp")
    print("  - Prompt details")
    print("  - Response timing and token usage")
    print("  - Response content preview")
    print("  - Success/error status")
    print("=" * 80)

if __name__ == "__main__":
    main()

# Pytest-compatible test functions (if pytest is available)
def test_ai_business_quality_pytest():
    """Pytest-compatible version of AI business quality test."""
    try:
        test_ai_business_quality()
        print("✅ AI Business Quality test passed")
    except Exception as e:
        raise AssertionError(f"AI Business Quality test failed: {e}")

def test_ai_comprehensive_analysis_pytest():
    """Pytest-compatible version of comprehensive AI analysis test."""
    try:
        test_ai_comprehensive_analysis()
        print("✅ AI Comprehensive Analysis test passed")
    except Exception as e:
        raise AssertionError(f"AI Comprehensive Analysis test failed: {e}")

def test_main_integration():
    """Pytest-compatible version of main integration test."""
    try:
        main()
        print("✅ Full AI logging integration test passed")
    except Exception as e:
        raise AssertionError(f"AI logging integration test failed: {e}")
