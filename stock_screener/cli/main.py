#!/usr/bin/env python3
"""
Command Line Interface for Stock Screener
==========================================

Main entry point for the stock screener CLI application.
"""

import sys
import os
import argparse
from pathlib import Path

# Add package to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from stock_screener.core.screener import UnifiedScreener
from stock_screener.core.analyzer import DetailedAnalyzer


def create_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description='Stock Screener with AI Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Run unified screening with config settings
  %(prog)s --config custom.properties # Use custom config file
  %(prog)s --legacy                  # Run legacy detailed analysis only
  %(prog)s --help                    # Show this help message
        """
    )
    
    parser.add_argument(
        '--config',
        metavar='FILE',
        help='Path to configuration file (default: config/screener_config.properties)'
    )
    
    parser.add_argument(
        '--legacy',
        action='store_true',
        help='Run legacy detailed analysis mode instead of unified screening'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='Stock Screener 2.0.0'
    )
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        if args.legacy:
            # Legacy mode - detailed analysis only
            print("🔄 Running in legacy mode...")
            analyzer = DetailedAnalyzer()
            summary = analyzer.run_analysis()
        else:
            # Modern mode - unified screening
            print("🚀 Running unified stock screening...")
            screener = UnifiedScreener(config_file=args.config)
            summary = screener.run_unified_screening()
        
        if summary:
            print("\n✅ Analysis completed successfully!")
            return 0
        else:
            print("\n❌ Analysis failed!")
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️ Analysis interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
