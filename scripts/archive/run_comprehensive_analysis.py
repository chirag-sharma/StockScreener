#!/usr/bin/env python3
"""
Comprehensive Analysis Runner
============================

This script takes the temp_basic_analysis.xlsx file and enhances it with:
1. AI-powered stock analysis and recommendations
2. Multi-period price predictions (6M, 12M)
3. Advanced sentiment analysis
4. Investment thesis generation
5. Risk assessment

Usage: python run_comprehensive_analysis.py
"""

import pandas as pd
import logging
import sys
from datetime import datetime
from pathlib import Path
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('comprehensive_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from stock_screener.core.analyzer import DetailedAnalyzer

def main():
    """Run comprehensive analysis on temp basic analysis file"""
    
    print("🚀 Starting Comprehensive Analysis Enhancement")
    print("=" * 60)
    
    # File paths
    input_file = "data/output/temp_basic_analysis.xlsx"
    output_file = f"data/output/comprehensive_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ ERROR: Input file not found: {input_file}")
        return 1
    
    try:
        # Load the basic analysis data
        print(f"📊 Loading basic analysis data from: {input_file}")
        df = pd.read_excel(input_file)
        print(f"✅ Loaded {len(df)} stocks with {len(df.columns)} columns")
        
        # Initialize the detailed analyzer
        print("🧠 Initializing AI-powered analyzer...")
        analyzer = DetailedAnalyzer(
            input_file=input_file,
            output_file=output_file,
            enable_ai_analysis=True,
            preferred_ai_provider='auto'
        )
        
        # Add company names if missing (required for AI analysis)
        if 'Company Name' not in df.columns:
            print("🏢 Adding company names...")
            df['Company Name'] = df['Symbol'].apply(lambda x: x.replace('.NS', '').replace('.BO', '') + ' Limited')
        
        # Ensure required columns exist
        required_columns = ['Symbol', 'Company Name']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"❌ ERROR: Missing required columns: {missing_columns}")
            return 1
        
        print("🎯 Starting comprehensive AI analysis...")
        print("   This may take some time due to:")
        print("   - AI-powered stock analysis")
        print("   - Multi-period price predictions (6M, 12M)")
        print("   - News sentiment analysis")
        print("   - Investment thesis generation")
        print()
        
        # Add comprehensive AI analysis
        enhanced_df = analyzer.add_ai_analysis_to_dataframe(df)
        
        # Save the comprehensive analysis
        print(f"💾 Saving comprehensive analysis to: {output_file}")
        analyzer.save_detailed_analysis(enhanced_df)
        
        print("=" * 60)
        print("✅ COMPREHENSIVE ANALYSIS COMPLETE!")
        print()
        print("📈 Enhanced Features Added:")
        
        # Check what new columns were added
        new_columns = set(enhanced_df.columns) - set(df.columns)
        if new_columns:
            print(f"   • {len(new_columns)} new analysis columns")
            key_new_columns = [col for col in new_columns if any(keyword in col for keyword in 
                             ['AI', 'Prediction', 'Target', 'Growth', 'Sentiment', 'Investment'])]
            for col in sorted(key_new_columns)[:10]:  # Show top 10
                print(f"     - {col}")
            if len(key_new_columns) > 10:
                print(f"     - ... and {len(key_new_columns) - 10} more columns")
        
        print()
        print(f"📊 Final Results:")
        print(f"   • Total stocks analyzed: {len(enhanced_df)}")
        print(f"   • Total columns: {len(enhanced_df.columns)}")
        print(f"   • Output file: {output_file}")
        
        # Quick stats on AI recommendations
        if 'AI Recommendation' in enhanced_df.columns:
            recommendation_counts = enhanced_df['AI Recommendation'].value_counts()
            print(f"   • AI Recommendations: {dict(recommendation_counts)}")
        
        # Quick stats on value scores
        if 'Value Score (1-10)' in enhanced_df.columns:
            valid_scores = enhanced_df['Value Score (1-10)'][enhanced_df['Value Score (1-10)'] != 'N/A']
            if len(valid_scores) > 0:
                valid_scores_numeric = pd.to_numeric(valid_scores, errors='coerce').dropna()
                if len(valid_scores_numeric) > 0:
                    avg_score = valid_scores_numeric.mean()
                    print(f"   • Average Value Score: {avg_score:.1f}/10")
        
        print()
        print("🎉 Ready for investment analysis!")
        return 0
        
    except Exception as e:
        print(f"❌ ERROR during analysis: {e}")
        logger.error(f"Comprehensive analysis failed: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
