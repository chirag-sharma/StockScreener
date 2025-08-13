#!/usr/bin/env python3
"""
Test dashboard functionality and data loading capabilities.
Validates dashboard components before launching the full interface.
"""

import sys
from pathlib import Path
import pandas as pd
import os

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_dashboard_dependencies():
    """Test if all required dashboard dependencies are available."""
    
    print("🔍 TESTING DASHBOARD DEPENDENCIES")
    print("=" * 45)
    
    try:
        import streamlit as st
        print("✅ Streamlit: Available")
    except ImportError:
        print("❌ Streamlit: Missing (pip install streamlit)")
        return False
    
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        print("✅ Plotly: Available")
    except ImportError:
        print("❌ Plotly: Missing (pip install plotly)")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy: Available")
    except ImportError:
        print("❌ NumPy: Missing (pip install numpy)")
        return False
    
    return True

def test_data_availability():
    """Test if analysis data files are available."""
    
    print("\n📊 TESTING DATA AVAILABILITY")
    print("=" * 45)
    
    output_dir = project_root / "data" / "output"
    
    if not output_dir.exists():
        print("❌ Output directory not found: data/output/")
        return False
    
    # Look for comprehensive analysis files
    analysis_files = list(output_dir.glob("comprehensive_analysis_*.xlsx"))
    
    if not analysis_files:
        print("❌ No comprehensive analysis files found")
        print("💡 Run analysis first: python scripts/run_screener.py")
        return False
    
    # Get the latest file
    latest_file = max(analysis_files, key=os.path.getctime)
    print(f"✅ Found analysis file: {latest_file.name}")
    
    # Try to load and validate the data
    try:
        df = pd.read_excel(latest_file)
        print(f"✅ Data loaded successfully: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Check for key columns
        key_columns = ['Symbol', 'PE Ratio', 'ROE', 'Value Score (1-10)', 'Investment Recommendation']
        missing_columns = [col for col in key_columns if col not in df.columns]
        
        if missing_columns:
            print(f"⚠️  Some expected columns missing: {missing_columns}")
        else:
            print("✅ All key columns present")
        
        # Show available AI and prediction columns
        ai_columns = [col for col in df.columns if 'AI' in col or 'Prediction' in col or 'Predicted' in col]
        if ai_columns:
            print(f"✅ AI/Prediction columns found: {len(ai_columns)}")
            for col in ai_columns[:5]:  # Show first 5
                print(f"   • {col}")
            if len(ai_columns) > 5:
                print(f"   • ... and {len(ai_columns) - 5} more")
        else:
            print("⚠️  No AI/Prediction columns found")
        
        return True, df
        
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return False, None

def test_dashboard_components():
    """Test dashboard component generation."""
    
    print("\n🧪 TESTING DASHBOARD COMPONENTS")
    print("=" * 45)
    
    # Test data loading
    data_test = test_data_availability()
    if not data_test[0]:
        return False
    
    df = data_test[1]
    
    # Test basic data operations
    try:
        # Test filtering
        if 'PE Ratio' in df.columns:
            filtered_data = df[df['PE Ratio'] <= 25]
            print(f"✅ Filtering works: {len(filtered_data)} stocks with PE <= 25")
        
        # Test grouping for charts
        if 'Investment Recommendation' in df.columns:
            grade_counts = df['Investment Recommendation'].value_counts()
            print(f"✅ Investment grade grouping: {len(grade_counts)} categories")
        
        # Test numeric operations
        if 'Value Score (1-10)' in df.columns:
            avg_score = df['Value Score (1-10)'].mean()
            print(f"✅ AI score calculation: Average = {avg_score:.2f}")
        
        print("✅ Dashboard components ready")
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {str(e)}")
        return False

def main():
    """Run all dashboard tests."""
    
    print("🚀 DASHBOARD READINESS TEST")
    print("=" * 50)
    print()
    
    # Test dependencies
    if not test_dashboard_dependencies():
        print("\n❌ DEPENDENCY TEST FAILED")
        print("📋 Please install missing dependencies and try again")
        sys.exit(1)
    
    # Test data availability
    if not test_data_availability()[0]:
        print("\n❌ DATA TEST FAILED")
        print("📋 Please run stock analysis first:")
        print("   python scripts/run_screener.py")
        sys.exit(1)
    
    # Test dashboard components
    if not test_dashboard_components():
        print("\n❌ COMPONENT TEST FAILED")
        sys.exit(1)
    
    print("\n🎉 ALL TESTS PASSED!")
    print("✅ Dashboard is ready to launch")
    print("\n🚀 To start the dashboard:")
    print("   python scripts/run_dashboard.py")
    print("\n🌐 Or launch directly:")
    print("   streamlit run stock_screener/dashboard/dashboard.py")
    
    # Show data summary
    print(f"\n📊 DATA SUMMARY:")
    data_test = test_data_availability()
    if data_test[0]:
        df = data_test[1]
        print(f"   • Total Stocks: {len(df)}")
        if 'Investment Recommendation' in df.columns:
            strong_buy = len(df[df['Investment Recommendation'] == 'Strong Buy'])
            print(f"   • Strong Buy Recommendations: {strong_buy}")
        if 'Value Score (1-10)' in df.columns:
            avg_ai_score = df['Value Score (1-10)'].mean()
            print(f"   • Average AI Score: {avg_ai_score:.1f}/10")
    
    print("\n🎯 Dashboard Features Available:")
    print("   • 📊 Interactive charts and visualizations")
    print("   • 🔍 Advanced filtering and search")
    print("   • 🔮 Multi-period price prediction analysis")
    print("   • 🧠 AI-powered investment insights")
    print("   • 📥 Data export capabilities")

if __name__ == "__main__":
    main()
