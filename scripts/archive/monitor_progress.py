#!/usr/bin/env python3
"""
Progress Monitor for Comprehensive Analysis
===========================================

This script monitors the progress of the comprehensive analysis
and provides estimated completion time.
"""

import os
import time
from datetime import datetime, timedelta

def monitor_progress():
    """Monitor the comprehensive analysis progress"""
    
    log_file = "comprehensive_analysis.log"
    total_stocks = 480
    
    if not os.path.exists(log_file):
        print("❌ Log file not found. Analysis may not have started yet.")
        return
    
    print("📊 Comprehensive Analysis Progress Monitor")
    print("=" * 50)
    
    while True:
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            # Count completed stocks
            analyzing_lines = [line for line in content.split('\n') if 'Analyzing stock' in line and '/480:' in line]
            
            if analyzing_lines:
                # Get the latest stock being analyzed
                latest_line = analyzing_lines[-1]
                # Extract stock number (e.g., "Analyzing stock 5/480:")
                stock_num = int(latest_line.split('Analyzing stock ')[1].split('/')[0])
                
                # Calculate progress
                progress_percent = (stock_num / total_stocks) * 100
                completed = stock_num - 1  # Previous stock was completed
                remaining = total_stocks - stock_num
                
                # Estimate completion time (assuming ~2 minutes per stock)
                avg_time_per_stock = 2  # minutes
                estimated_remaining_time = remaining * avg_time_per_stock
                estimated_completion = datetime.now() + timedelta(minutes=estimated_remaining_time)
                
                print(f"\n🔄 Analysis Progress:")
                print(f"   Current Stock: {stock_num}/{total_stocks}")
                print(f"   Completed: {completed} stocks")
                print(f"   Progress: {progress_percent:.1f}%")
                print(f"   Remaining: {remaining} stocks")
                print(f"   Estimated Time Remaining: {estimated_remaining_time:.0f} minutes")
                print(f"   Estimated Completion: {estimated_completion.strftime('%H:%M:%S')}")
                
                # Show progress bar
                bar_length = 40
                filled_length = int(bar_length * progress_percent / 100)
                bar = '█' * filled_length + '-' * (bar_length - filled_length)
                print(f"   Progress: |{bar}| {progress_percent:.1f}%")
                
                # Check if completed
                if stock_num >= total_stocks:
                    print("\n🎉 Analysis appears to be complete!")
                    break
                
            else:
                print("⏳ Analysis starting... No stocks processed yet.")
            
            # Wait before next check
            time.sleep(30)  # Check every 30 seconds
            
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped by user.")
            break
        except Exception as e:
            print(f"❌ Error monitoring progress: {e}")
            time.sleep(30)

if __name__ == "__main__":
    monitor_progress()
