#!/usr/bin/env python3
"""
Simple Basic Stock Screener Runner
===================================

A simplified script to run the basic stock screener (without AI).
This script calls the existing run_ai_screener.py with basic mode for quick analysis.

Features:
- Fast execution (3-5 seconds)
- Basic financial analysis
- Rule-based recommendations
- Excel report generation

Usage: python scripts/run_basic_screener.py
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Run the basic stock screener."""
    
    print("📊 BASIC STOCK SCREENER")
    print("=" * 40)
    print("⚡ Fast Analysis (No AI)")
    print("📈 Rule-based Recommendations") 
    print("📋 Basic Financial Metrics")
    print("=" * 40)
    print()
    
    # Get project root and virtual environment
    project_root = Path(__file__).parent.parent
    venv_python = project_root / ".venv" / "bin" / "python"
    run_screener_path = project_root / "scripts" / "run_ai_screener.py"
    
    # Check if virtual environment exists
    if not venv_python.exists():
        print("❌ Virtual environment not found!")
        print("Please run: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt")
        return 1
    
    # Check if run_ai_screener.py exists
    if not run_screener_path.exists():
        print("❌ run_ai_screener.py not found!")
        return 1
    
    try:
        print("🔄 Starting basic analysis...")
        print("⏱️  This should complete in 3-5 seconds...")
        print()
        
        # Run the basic screener without AI analysis
        result = subprocess.run([
            str(venv_python), 
            str(run_screener_path),
            "--basic"  # Add basic mode flag to skip AI analysis
        ], cwd=project_root, check=True)
        
        print()
        print("✅ Basic analysis completed!")
        print("📄 Check the data/output/ directory for Excel reports")
        print("🎯 You can also run: python scripts/run_dashboard.py")
        
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Basic screener failed with exit code {e.returncode}")
        return e.returncode
    except FileNotFoundError:
        print("❌ Python executable not found in virtual environment")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
