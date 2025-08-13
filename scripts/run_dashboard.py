#!/usr/bin/env python3
"""
Launch script for the Stock Screener AI-powered dashboard.
Provides easy access to comprehensive analysis visualization.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Launch the Streamlit dashboard with proper configuration."""
    
    print("🚀 LAUNCHING AI-POWERED STOCK SCREENER DASHBOARD")
    print("=" * 55)
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    dashboard_path = project_root / "stock_screener" / "dashboard" / "dashboard.py"
    
    # Check if dashboard file exists
    if not dashboard_path.exists():
        print(f"❌ Dashboard file not found: {dashboard_path}")
        sys.exit(1)
    
    # Check if data files exist
    output_dir = project_root / "data" / "output"
    
    # Look for analysis files with different patterns
    analysis_files = []
    analysis_files.extend(list(output_dir.glob("comprehensive_analysis_*.xlsx")))
    analysis_files.extend(list(output_dir.glob("comprehensive_analysis.xlsx")))
    
    if not analysis_files:
        print("❌ No comprehensive analysis files found!")
        print("📋 Please run the screener first:")
        print("   python scripts/run_screener.py")
        print("📁 Or check if files exist in data/output/")
        sys.exit(1)
    
    latest_file = max(analysis_files, key=os.path.getctime)
    print(f"📊 Found analysis data: {latest_file.name}")
    
    # Change to project root for proper relative path resolution
    os.chdir(project_root)
    
    print("\n🌐 Starting Streamlit dashboard...")
    print("📱 Dashboard will open in your default web browser")
    print("🔗 URL: http://localhost:8501")
    print("\n💡 Dashboard Features:")
    print("   • 📊 Interactive stock analysis visualization")
    print("   • 🔮 Multi-period price prediction charts") 
    print("   • 🧠 AI-powered investment insights")
    print("   • 🔍 Advanced filtering and search")
    print("   • 📥 Data export capabilities")
    print("\n⏹️  Press Ctrl+C to stop the dashboard")
    print("=" * 55)
    
    try:
        # Launch Streamlit dashboard
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_path),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--server.headless", "false",
            "--browser.serverAddress", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"\n❌ Error launching dashboard: {str(e)}")
        print("\n🔧 Troubleshooting:")
        print("   1. Ensure Streamlit is installed: pip install streamlit")
        print("   2. Ensure Plotly is installed: pip install plotly")
        print("   3. Check that analysis data exists in data/output/")
        sys.exit(1)

if __name__ == "__main__":
    main()
