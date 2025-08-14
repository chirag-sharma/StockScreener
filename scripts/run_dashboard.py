#!/usr/bin/env python3
"""
Stock Screener Dashboard Launcher
================================

Professional AI-powered stock analysis dashboard with:
- Modern UI with advanced styling
- Real-time price updates and predictions
- Interactive portfolio management
- Comprehensive analytics and AI insights
- Mobile-responsive design
- Multi-period forecasting
"""

import subprocess
import sys
import os
from pathlib import Path
import webbrowser
import time

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = ['streamlit', 'plotly', 'yfinance', 'pandas', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        for package in missing_packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print("✅ All packages installed successfully!")

def check_data_availability():
    """Check if analysis data is available"""
    output_dir = Path("data/output")
    
    if not output_dir.exists():
        print("❌ Output directory not found: data/output/")
        print("📋 Please run the screener first:")
        print("   python scripts/run_screener.py")
        return False
    
    # Look for analysis files
    analysis_files = list(output_dir.glob("comprehensive_analysis_*.xlsx"))
    analysis_files.extend(list(output_dir.glob("comprehensive_analysis.xlsx")))
    
    if not analysis_files:
        print("❌ No analysis data found!")
        print("📋 Please run the screener first:")
        print("   python scripts/run_screener.py")
        return False
    
    latest_file = max(analysis_files, key=os.path.getmtime)
    print(f"✅ Found analysis data: {latest_file.name}")
    return True

def launch_dashboard():
    """Launch the enhanced Streamlit dashboard"""
    print("🚀 Launching AI Stock Screener Pro Dashboard...")
    print("=" * 60)
    print("🎯 Features:")
    print("   • Professional UI with modern styling")
    print("   • Real-time price updates")
    print("   • Interactive charts and analytics")
    print("   • AI insights visualization")
    print("   • Mobile-responsive design")
    print("   • Advanced filtering options")
    print("=" * 60)
    
    dashboard_path = Path("stock_screener/dashboard/dashboard.py")
    
    if not dashboard_path.exists():
        print(f"❌ Dashboard file not found: {dashboard_path}")
        return False
    
    try:
        # Launch Streamlit with enhanced configuration
        cmd = [
            "streamlit", "run", str(dashboard_path),
            "--server.port=8501",
            "--server.address=localhost",
            "--browser.gatherUsageStats=false",
            "--server.headless=true",  # Disable automatic browser opening
            "--theme.primaryColor=#3498db",
            "--theme.backgroundColor=#ffffff",
            "--theme.secondaryBackgroundColor=#f8f9fa",
            "--theme.textColor=#2c3e50"
        ]
        
        print("🌐 Starting dashboard server...")
        print("📍 URL: http://localhost:8501")
        print("⚡ Press Ctrl+C to stop the dashboard")
        print("-" * 40)
        
        # Auto-open browser after a short delay
        def open_browser():
            time.sleep(3)
            webbrowser.open("http://localhost:8501")
        
        import threading
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Run Streamlit
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching dashboard: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    return True

def main():
    """Main function to launch the enhanced dashboard"""
    print("🚀 AI Stock Screener Pro Dashboard Launcher")
    print("=" * 50)
    
    # Change to project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    
    # Check dependencies
    print("1. Checking dependencies...")
    check_dependencies()
    
    # Check data availability
    print("2. Checking data availability...")
    if not check_data_availability():
        print("\n💡 Quick Start Guide:")
        print("   1. Run: python scripts/run_screener.py")
        print("   2. Wait for analysis to complete")
        print("   3. Run: python scripts/run_dashboard.py")
        return
    
    # Launch dashboard
    print("3. Launching dashboard...")
    launch_dashboard()

if __name__ == "__main__":
    main()
