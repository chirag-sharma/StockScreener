#!/usr/bin/env python3
"""
Stock Screener Launcher
========================

Main launcher script that provides options to run different types of analysis.

Usage: python scripts/launcher.py
"""

import sys
import subprocess
from pathlib import Path

def print_banner():
    """Print the application banner."""
    print("🚀" * 25)
    print("📈 STOCK SCREENER LAUNCHER 📈")
    print("🚀" * 25)
    print()

def print_options():
    """Print available analysis options."""
    print("Choose your analysis type:")
    print()
    print("1️⃣  AI-Powered Analysis")
    print("   🤖 Real OpenAI integration")
    print("   📰 News sentiment analysis")
    print("   🎯 AI price predictions")
    print("   ⏱️  Takes 1-2 minutes")
    print()
    print("2️⃣  Basic Analysis")
    print("   ⚡ Fast rule-based analysis")
    print("   📊 Financial metrics only")
    print("   🏃 Completes in 3-5 seconds")
    print()
    print("3️⃣  Dashboard")
    print("   🌐 Interactive web interface")
    print("   📋 View previous results")
    print("   📊 Visualizations & charts")
    print()
    print("0️⃣  Exit")
    print()

def run_ai_screener():
    """Run the AI-powered screener."""
    script_path = Path(__file__).parent / "run_ai_screener.py"
    return subprocess.call([sys.executable, str(script_path)])

def run_basic_screener():
    """Run the basic screener."""
    script_path = Path(__file__).parent / "run_basic_screener.py"
    return subprocess.call([sys.executable, str(script_path)])

def run_dashboard():
    """Run the dashboard."""
    script_path = Path(__file__).parent / "run_dashboard.py"
    return subprocess.call([sys.executable, str(script_path)])

def main():
    """Main launcher function."""
    
    while True:
        print_banner()
        print_options()
        
        try:
            choice = input("Enter your choice (1/2/3/0): ").strip()
            print()
            
            if choice == "1":
                print("🚀 Launching AI-Powered Analysis...")
                print("-" * 40)
                return run_ai_screener()
                
            elif choice == "2":
                print("⚡ Launching Basic Analysis...")
                print("-" * 40)
                return run_basic_screener()
                
            elif choice == "3":
                print("🌐 Launching Dashboard...")
                print("-" * 40)
                return run_dashboard()
                
            elif choice == "0":
                print("👋 Goodbye!")
                return 0
                
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 0.")
                input("Press Enter to continue...")
                print("\n" * 2)
                continue
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            return 0
        except Exception as e:
            print(f"❌ Error: {e}")
            return 1

if __name__ == "__main__":
    sys.exit(main())
