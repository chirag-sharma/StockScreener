#!/usr/bin/env python3
"""
Utility script to update dates in ticker files without changing the ticker content.
This prevents the screener from thinking cache files are invalid and overwriting them.
"""

import os
import json
from datetime import datetime
from pathlib import Path

def update_ticker_file_dates(ticker_dir):
    """Update dates in all ticker JSON files to today's date"""
    ticker_dir = Path(ticker_dir)
    today = datetime.today().strftime("%Y-%m-%d")
    updated_files = []
    
    print("🔄 TICKER FILE DATE UPDATER")
    print("=" * 40)
    print(f"Target date: {today}")
    print()
    
    for json_file in ticker_dir.glob("*.json"):
        try:
            # Read existing data
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Check if it needs updating
            current_date = data.get('date', 'N/A')
            ticker_count = len(data.get('tickers', []))
            
            if current_date != today:
                # Update the date
                data['date'] = today
                
                # Write back to file
                with open(json_file, 'w') as f:
                    json.dump(data, f, indent=2)
                
                updated_files.append(json_file.name)
                print(f"✅ {json_file.name:20s} | {current_date:10s} → {today:10s} | {ticker_count:2d} tickers")
            else:
                print(f"⭐ {json_file.name:20s} | {current_date:10s} (current) | {ticker_count:2d} tickers")
                
        except Exception as e:
            print(f"❌ {json_file.name:20s} | ERROR: {e}")
    
    print()
    if updated_files:
        print(f"🎯 Updated {len(updated_files)} files: {', '.join(updated_files)}")
    else:
        print("🎯 All files already have current date")
    
    return updated_files

if __name__ == "__main__":
    ticker_directory = "data/input/tickers"
    
    if not os.path.exists(ticker_directory):
        print(f"❌ Ticker directory not found: {ticker_directory}")
        print("Run this script from the project root directory")
        exit(1)
    
    print("🔧 Stock Screener - Ticker Date Updater")
    print("This tool updates dates in ticker files to prevent cache overwrites")
    print()
    
    updated = update_ticker_file_dates(ticker_directory)
    
    print()
    print("✅ Date update complete!")
    print("Your ticker files are now protected from cache overwrites.")
