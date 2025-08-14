"""
Column Mapping Configuration for Enhanced Dashboard
=================================================

Maps the actual Excel column names to the expected dashboard column names.
"""

import pandas as pd
import numpy as np

# Column name mapping from Excel to dashboard
COLUMN_MAPPING = {
    # Basic info
    'Symbol': 'symbol',
    'Company Name': 'company_name',
    'Current Price (₹)': 'current_price',
    'Market Cap (Cr)': 'market_cap',
    
    # Financial metrics
    'PE Ratio': 'pe_ratio',
    'Price to Book': 'pb_ratio',
    'EV/EBITDA': 'ev_ebitda',
    'Debt/Equity': 'debt_equity',
    'ROE': 'roe',
    'Return on Assets (ROA)': 'roa',
    'Net Profit Margin': 'net_profit_margin',
    'EPS Growth (%)': 'eps_growth',
    'Revenue Growth (%)': 'revenue_growth',
    
    # Valuation
    'Value Score': 'value_score',
    'Investment Recommendation': 'investment_grade',
    'Margin of Safety (%)': 'margin_of_safety',
    
    # AI Analysis
    'AI Sentiment': 'ai_sentiment',
    'AI Reasoning': 'ai_analysis',
    'Business Quality': 'ai_business_quality_score',
    'Financial Health': 'financial_health',
    'Risk Level': 'risk_level',
    
    # Predictions
    'Target Price (12M)': 'target_price',
    'Price Target (6M)': 'target_price_6m',
    'Predicted Price (30d)': 'predicted_price_30d',
    'Growth 12M (%)': 'growth_12m',
    'Growth 6M (%)': 'growth_6m',
    'Price Change % (30d)': 'price_change_30d',
    'Prediction Confidence': 'prediction_confidence',
    
    # Technical
    'Volume': 'volume',
    'Day High (₹)': 'day_high',
    'Day Low (₹)': 'day_low',
    '52W High (₹)': 'week_52_high',
    '52W Low (₹)': 'week_52_low',
}

# Reverse mapping for display purposes
DISPLAY_MAPPING = {v: k for k, v in COLUMN_MAPPING.items()}

def normalize_dataframe(df):
    """Normalize dataframe column names for dashboard compatibility"""
    # Create a copy to avoid modifying original
    normalized_df = df.copy()
    
    # Rename columns using mapping
    normalized_df = normalized_df.rename(columns=COLUMN_MAPPING)
    
    # Create derived columns that the dashboard expects
    if 'value_score' in normalized_df.columns:
        # Use value_score as final_score if available
        normalized_df['final_score'] = normalized_df['value_score']
    else:
        # Create a composite score from available metrics
        score_components = []
        weights = []
        
        # Add ROE component (weight: 0.3)
        if 'roe' in normalized_df.columns:
            score_components.append(normalized_df['roe'].fillna(0))
            weights.append(0.3)
        
        # Add margin of safety component (weight: 0.3)
        if 'margin_of_safety' in normalized_df.columns:
            score_components.append(normalized_df['margin_of_safety'].fillna(0))
            weights.append(0.3)
        
        # Add AI business quality component (weight: 0.4)
        if 'ai_business_quality_score' in normalized_df.columns:
            # Convert text scores to numeric if needed
            ai_scores = pd.to_numeric(normalized_df['ai_business_quality_score'], errors='coerce').fillna(50)
            score_components.append(ai_scores)
            weights.append(0.4)
        
        # Calculate composite score
        if score_components:
            weights = np.array(weights) / sum(weights)  # Normalize weights
            composite_score = sum(w * comp for w, comp in zip(weights, score_components))
            normalized_df['final_score'] = composite_score
        else:
            # Fallback: use a constant score
            normalized_df['final_score'] = 50.0
    
    # Handle investment grade normalization
    if 'investment_grade' in normalized_df.columns:
        # Map common investment recommendations to standard format
        grade_mapping = {
            'Strong Buy': 'STRONG_BUY',
            'Buy': 'BUY',
            'Hold': 'HOLD',
            'Sell': 'AVOID',
            'Avoid': 'AVOID',
            'STRONG BUY': 'STRONG_BUY',
        }
        normalized_df['investment_grade'] = normalized_df['investment_grade'].map(
            lambda x: grade_mapping.get(str(x), str(x)) if pd.notna(x) else 'HOLD'
        )
    else:
        # Create investment grades based on final_score
        def score_to_grade(score):
            if score >= 80: return 'STRONG_BUY'
            elif score >= 65: return 'BUY'
            elif score >= 45: return 'HOLD'
            else: return 'AVOID'
        
        normalized_df['investment_grade'] = normalized_df['final_score'].apply(score_to_grade)
    
    # Convert numeric columns
    numeric_columns = [
        'current_price', 'market_cap', 'pe_ratio', 'pb_ratio', 'roe', 'roa',
        'final_score', 'target_price', 'value_score', 'margin_of_safety'
    ]
    
    for col in numeric_columns:
        if col in normalized_df.columns:
            normalized_df[col] = pd.to_numeric(normalized_df[col], errors='coerce').fillna(0)
    
    # Ensure target_price exists
    if 'target_price' not in normalized_df.columns and 'target_price_6m' in normalized_df.columns:
        normalized_df['target_price'] = normalized_df['target_price_6m']
    elif 'target_price' not in normalized_df.columns:
        # Create target price based on current price and some upside
        normalized_df['target_price'] = normalized_df['current_price'] * 1.15  # 15% upside assumption
    
    return normalized_df

def get_available_columns(df):
    """Get list of available columns after normalization"""
    normalized_df = normalize_dataframe(df)
    return list(normalized_df.columns)

def safe_column_access(df, column_name, default_value=0):
    """Safely access column with fallback"""
    if column_name in df.columns:
        return df[column_name].fillna(default_value)
    else:
        return pd.Series([default_value] * len(df), index=df.index)
