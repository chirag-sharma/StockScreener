#!/usr/bin/env python3

import pandas as pd
import numpy as np

def analyze_excel_data():
    """Analyze the Excel data to understand upside calculation issues"""
    try:
        # Read the Excel file
        df = pd.read_excel('data/output/comprehensive_analysis.xlsx')
        
        print('=' * 60)
        print('EXCEL DATA ANALYSIS')
        print('=' * 60)
        
        print(f'Data shape: {df.shape}')
        print(f'\nColumn names ({len(df.columns)} total):')
        for i, col in enumerate(df.columns.tolist(), 1):
            print(f'{i:2d}. {col}')
        
        # Check for price-related columns
        price_cols = [col for col in df.columns if any(keyword in col.lower() 
                     for keyword in ['target', 'price', 'current', 'ltp'])]
        print(f'\nPrice-related columns: {price_cols}')
        
        if price_cols:
            print('\nFirst 5 rows of price columns:')
            print(df[price_cols].head())
            
            print(f'\nData types:')
            print(df[price_cols].dtypes)
            
            print(f'\nNull values:')
            print(df[price_cols].isnull().sum())
            
            print(f'\nDescriptive statistics:')
            print(df[price_cols].describe())
            
            # Check specific target price column
            target_col = None
            current_col = None
            
            for col in df.columns:
                if 'target' in col.lower():
                    target_col = col
                if 'current' in col.lower() or 'ltp' in col.lower():
                    current_col = col
            
            print(f'\nTarget price column: {target_col}')
            print(f'Current price column: {current_col}')
            
            if target_col:
                print(f'\nUnique values in {target_col} (first 20):')
                print(df[target_col].value_counts().head(20))
                
                print(f'\nZero values in {target_col}: {(df[target_col] == 0).sum()}')
                print(f'NaN values in {target_col}: {df[target_col].isna().sum()}')
                
            if current_col:
                print(f'\nUnique values in {current_col} (first 20):')
                print(df[current_col].value_counts().head(20))
                
                print(f'\nZero values in {current_col}: {(df[current_col] == 0).sum()}')
                print(f'NaN values in {current_col}: {df[current_col].isna().sum()}')
                
            # Check upside calculation manually
            if target_col and current_col:
                print(f'\nManual upside calculation check:')
                # Use the correct symbol column name
                symbol_col = 'Symbol' if 'Symbol' in df.columns else 'symbol'
                current_price_col = 'Current Price (₹)' if 'Current Price (₹)' in df.columns else current_col
                
                sample_data = df[[target_col, current_price_col, symbol_col]].head(10)
                print(sample_data)
                
                for idx, row in sample_data.iterrows():
                    symbol = row[symbol_col]
                    target = row[target_col]
                    current = row[current_price_col]
                    
                    if pd.notna(target) and pd.notna(current) and current > 0:
                        upside = ((target - current) / current) * 100
                        print(f'{symbol}: Target={target}, Current={current}, Upside={upside:.2f}%')
                    else:
                        print(f'{symbol}: Target={target}, Current={current}, Upside=N/A (invalid data)')
        
    except Exception as e:
        print(f'Error analyzing data: {e}')

if __name__ == '__main__':
    analyze_excel_data()
