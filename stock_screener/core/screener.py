#!/usr/bin/env python3
"""
Unified Stock Screener with AI Analysis
=======================================

This module combines basic stock screening with detailed AI analysis in a single workflow.
No more intermediate files - direct screening to comprehensive analysis.

Features:
- Configuration-driven AI settings (no command-line arguments needed)
- Real-time news integration
- Value investing principles (Graham & Buffett)
- Multi-provider AI support (OpenAI, Gemini, Claude, Ollama)
- Professional Excel output with color coding

Usage:
    python -m stock_screener.core.screener
    
Configuration:
    Edit stock_screener/config/screener_config.properties to adjust:
    - AI provider and settings
    - Investment criteria thresholds
    - News integration preferences
"""

import pandas as pd
import sys
import os
from pathlib import Path
import logging
from datetime import datetime, timedelta
import configparser
import time

# Add the project root to Python path for module imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Core services
from stock_screener.services.excelExporter import ExcelExporter
from stock_screener.services.stockAnalyzer import StockAnalyzer
from stock_screener.utils.configLoader import load_symbols_from_config
from stock_screener.utils.tickerLoader import load_tickers

# Import the comprehensive analysis functionality
from stock_screener.core.analyzer import DetailedAnalyzer

# Import constants for proper path handling
from stock_screener.core.constants import BASE_OUTPUT_DIR

# Environment variable loading
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class UnifiedScreener:
    """
    Unified stock screener that combines basic screening with detailed AI analysis.
    Configuration-driven instead of command-line dependent.
    """
    
    def __init__(self, config_file=None):
        """Initialize with configuration file."""
        self.config_file = config_file or 'stock_screener/config/screener_config.properties'
        self.config = self._load_config()
        
        # Set output file path using constants
        self.output_file = os.path.join(BASE_OUTPUT_DIR, 'comprehensive_analysis.xlsx')
        
        # Initialize AI settings from config
        self.ai_enabled = self.config.getboolean('DEFAULT', 'ai_analysis_enabled', fallback=True)
        self.ai_provider = self.config.get('DEFAULT', 'ai_provider', fallback='auto')
        self.news_enabled = self.config.getboolean('DEFAULT', 'news_integration_enabled', fallback=True)
        self.news_days_back = self.config.getint('DEFAULT', 'news_days_back', fallback=30)
        
        logger.info(f"🤖 AI Analysis: {'ENABLED' if self.ai_enabled else 'DISABLED'}")
        if self.ai_enabled:
            logger.info(f"   - Provider: {self.ai_provider}")
            logger.info(f"   - News Integration: {'ENABLED' if self.news_enabled else 'DISABLED'}")
    
    def _load_config(self):
        """Load configuration from properties file."""
        config = configparser.ConfigParser()
        if os.path.exists(self.config_file):
            config.read(self.config_file)
            logger.info(f"Loaded configuration from: {self.config_file}")
        else:
            logger.warning(f"Config file not found: {self.config_file}. Using defaults.")
            # Add default section for fallback
            config.add_section('DEFAULT')
        return config
    
    def run_unified_screening(self):
        """
        Execute unified screening workflow:
        1. Load stock symbols
        2. Analyze each symbol 
        3. Apply comprehensive AI analysis
        4. Generate single comprehensive report
        """
        try:
            logger.info("🚀 Starting Unified Stock Screening with AI Analysis...")
            
            # Step 1: Load symbols
            symbols = load_symbols_from_config()
            if not symbols:
                logger.error("No symbols loaded from configuration")
                return
            
            logger.info(f"📊 Loaded {len(symbols)} symbols for analysis")
            
            # Step 2: Basic screening
            logger.info("🔍 Phase 1: Basic Stock Analysis...")
            all_results = []
            
            for i, symbol in enumerate(symbols, 1):
                logger.info(f"   Analyzing {symbol} ({i}/{len(symbols)})")
                
                analyzer = StockAnalyzer(symbol)
                analyzer.fetch_data()
                result = analyzer.analyze()
                
                if result:
                    all_results.append(result)
                
                # Small delay to avoid overwhelming APIs
                time.sleep(0.5)
            
            if not all_results:
                logger.error("❌ No valid analysis results obtained")
                return
            
            logger.info(f"✅ Basic analysis completed for {len(all_results)} stocks")
            
            # Step 3: Create temporary Excel file for DetailedAnalyzer
            temp_file = os.path.join(BASE_OUTPUT_DIR, 'temp_basic_analysis.xlsx')
            os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
            
            exporter = ExcelExporter(temp_file)
            exporter.write_data(all_results)
            
            # Step 4: Enhanced AI Analysis
            logger.info("🧠 Phase 2: Comprehensive AI Analysis...")
            
            detailed_analyzer = DetailedAnalyzer(
                input_file=temp_file,
                output_file=self.output_file,
                enable_ai_analysis=self.ai_enabled,
                preferred_ai_provider=self.ai_provider
            )
            
            # Run comprehensive analysis
            summary = detailed_analyzer.run_analysis()
            
            # Clean up temporary file
            try:
                os.remove(temp_file)
                logger.debug("Cleaned up temporary analysis file")
            except:
                pass
            
            # Display results
            self._display_summary(summary)
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Unified screening failed: {e}")
            raise
    
    def _display_summary(self, summary):
        """Display comprehensive analysis summary."""
        print("\n" + "="*60)
        print("🎯 COMPREHENSIVE SCREENING COMPLETE!")
        print("="*60)
        print(f"📊 Total stocks analyzed: {summary.get('total_stocks_analyzed', 0)}")
        print(f"📈 Average readiness score: {summary.get('avg_readiness_score', 'N/A')}/100")
        
        if self.ai_enabled:
            print(f"🧠 Average AI value score: {summary.get('avg_ai_value_score', 'N/A')}/10")
        
        print(f"⭐ Excellent candidates: {summary.get('excellent_candidates', 0)}")
        
        print("\n📋 Investment Recommendations:")
        print(f"   - Strong Buy: {summary.get('strong_buy_count', 0)}")
        print(f"   - Buy: {summary.get('buy_count', 0)}")
        print(f"   - Hold/Others: {summary.get('hold_count', 0)}")
        
        print(f"\n📄 Comprehensive report: {summary.get('output_file', self.output_file)}")
        
        if self.ai_enabled:
            print("\n🤖 Enhanced with comprehensive AI analysis:")
            print("   - Value investing scorecard (Graham & Buffett principles)")
            print("   - Financial health & business quality assessment")
            print("   - Risk analysis & margin of safety calculation")
            print("   - Investment thesis & growth catalysts")
            print("   - Target price estimates & valuation analysis")
            if self.news_enabled:
                print("   - Real-time news integration for factual recommendations")


def main():
    """Main execution function."""
    try:
        # Initialize unified screener
        screener = UnifiedScreener()
        
        # Run comprehensive analysis
        summary = screener.run_unified_screening()
        
        if summary:
            logger.info("✅ Unified screening completed successfully")
        else:
            logger.error("❌ Unified screening failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("⏹️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
