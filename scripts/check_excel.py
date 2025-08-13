#!/usr/bin/env python3
"""Quick check of the Excel output"""

import pandas as pd
import sys
from pathlib import Path

try:
    df = pd.read_excel('/Users/chirag/PycharmProjects/StockScreener/data/output/comprehensive_analysis.xlsx')
    
    print("📊 EXCEL FILE ANALYSIS COMPLETE!")
    print("=" * 50)
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print()
    
    print("🎯 PREDICTION-RELATED COLUMNS:")
    print("-" * 30)
    prediction_cols = []
    for i, col in enumerate(df.columns):
        if any(keyword in col.lower() for keyword in ['price', 'target', 'growth', 'predicted', 'confidence']):
            prediction_cols.append(col)
            print(f"   ✅ {col}")
    
    print(f"\n📈 Found {len(prediction_cols)} prediction-related columns")
    
    # Show TCS data
    tcs_data = df[df['Symbol'] == 'TCS.NS']
    if not tcs_data.empty:
        tcs_row = tcs_data.iloc[0]
        print(f"\n🎯 TCS.NS SAMPLE DATA:")
        print("-" * 25)
        for col in prediction_cols:
            value = tcs_row.get(col, 'N/A')
            print(f"   {col}: {value}")
    
    print("\n✅ Excel analysis complete!")
    
except Exception as e:
    print(f"❌ Error reading Excel: {e}")
    sys.exit(1)
