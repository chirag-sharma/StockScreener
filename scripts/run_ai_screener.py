#!/usr/bin/env python3
"""
Simple AI-Powered Stock Screener Runner
========================================

A simplified script to run the comprehensive AI-powered stock screener.
This script calls the real AI-enabled screener with OpenAI integration.

Features:
- Real AI analysis using OpenAI GPT
- News integration and sentiment analysis
- Price predictions and target calculations
- Comprehensive Excel report generation

Usage: python scripts/run_ai_screener.py
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Run the AI-powered stock screener."""
    
    print("🚀 AI-POWERED STOCK SCREENER")
    print("=" * 50)
    print("🤖 Using Real OpenAI Integration")
    print("📰 With News Sentiment Analysis")
    print("📊 Comprehensive Financial Analysis")
    print("=" * 50)
    print()
    
    # Get project root and virtual environment
    project_root = Path(__file__).parent.parent
    venv_python = project_root / ".venv" / "bin" / "python"
    
    # Check if virtual environment exists
    if not venv_python.exists():
        print("❌ Virtual environment not found!")
        print("Please run: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt")
        return 1
    
    try:
        print("🔄 Starting AI analysis (this may take 1-2 minutes)...")
        print("⏱️  Please be patient while we fetch data and analyze stocks...")
        print()
        
        # Run the real AI screener
        result = subprocess.run([
            str(venv_python), 
            "-m", 
            "stock_screener.core.screener"
        ], cwd=project_root, check=True)
        
        print()
        print("✅ AI analysis completed successfully!")
        print("📄 Check the data/output/ directory for Excel reports")
        print("🎯 You can also run: python scripts/run_dashboard.py")
        
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"❌ AI screener failed with exit code {e.returncode}")
        print("💡 Make sure your API keys are configured in .env file")
        return e.returncode
    except FileNotFoundError:
        print("❌ Python executable not found in virtual environment")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
