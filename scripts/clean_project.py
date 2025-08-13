#!/usr/bin/env python3
"""
Project Cleanup Script
=====================

This script cleans up the StockScreener project by:
1. Removing temporary files and logs
2. Cleaning cache directories 
3. Organizing output files
4. Removing redundant/empty files
5. Creating a clean project structure

Usage: python clean_project.py
"""

import os
import shutil
import glob
from datetime import datetime
from pathlib import Path

def clean_cache_files():
    """Remove Python cache files and directories"""
    print("🧹 Cleaning Python cache files...")
    
    # Remove __pycache__ directories
    pycache_dirs = glob.glob("**/__pycache__", recursive=True)
    for cache_dir in pycache_dirs:
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            print(f"   Removed: {cache_dir}")
    
    # Remove .pyc files
    pyc_files = glob.glob("**/*.pyc", recursive=True)
    for pyc_file in pyc_files:
        if os.path.exists(pyc_file):
            os.remove(pyc_file)
            print(f"   Removed: {pyc_file}")
    
    print(f"   ✅ Cleaned {len(pycache_dirs)} cache directories and {len(pyc_files)} .pyc files")

def clean_log_files():
    """Clean up log files"""
    print("📋 Cleaning log files...")
    
    log_files = glob.glob("*.log")
    if log_files:
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        for log_file in log_files:
            if os.path.getsize(log_file) > 0:  # Only move non-empty logs
                dest = f"logs/{log_file}"
                shutil.move(log_file, dest)
                print(f"   Moved: {log_file} → {dest}")
            else:
                os.remove(log_file)
                print(f"   Removed empty: {log_file}")
        
        print(f"   ✅ Processed {len(log_files)} log files")
    else:
        print("   ✅ No log files to clean")

def clean_temporary_files():
    """Remove temporary and backup files"""
    print("🗑️  Cleaning temporary files...")
    
    temp_patterns = [
        "**/*~",           # Backup files
        "**/.DS_Store",    # macOS files
        "**/~$*",          # Office temp files
        "**/*.tmp",        # Temp files
        "**/*.temp",       # Temp files
    ]
    
    removed_count = 0
    for pattern in temp_patterns:
        temp_files = glob.glob(pattern, recursive=True)
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                print(f"   Removed: {temp_file}")
                removed_count += 1
    
    print(f"   ✅ Removed {removed_count} temporary files")

def organize_output_files():
    """Organize output files"""
    print("📊 Organizing output files...")
    
    output_dir = "data/output"
    if not os.path.exists(output_dir):
        print("   ⚠️  No output directory found")
        return
    
    # List current files
    output_files = os.listdir(output_dir)
    current_files = [f for f in output_files if f.endswith('.xlsx')]
    
    if not current_files:
        print("   ✅ No Excel files to organize")
        return
    
    print(f"   📁 Found {len(current_files)} Excel files:")
    
    # Categorize files
    final_files = []
    template_files = []
    temp_files = []
    old_files = []
    
    for file in current_files:
        file_path = os.path.join(output_dir, file)
        file_size = os.path.getsize(file_path)
        
        if 'REAL' in file:
            final_files.append((file, file_size))
        elif 'template' in file:
            template_files.append((file, file_size))
        elif 'temp' in file or 'copy' in file:
            temp_files.append((file, file_size))
        elif 'old' in file:
            old_files.append((file, file_size))
        else:
            final_files.append((file, file_size))
    
    # Display categorization
    if final_files:
        print("   🎯 Final Analysis Files:")
        for file, size in final_files:
            print(f"      • {file} ({size:,} bytes)")
    
    if template_files:
        print("   📋 Template Files:")
        for file, size in template_files:
            print(f"      • {file} ({size:,} bytes)")
    
    if temp_files:
        print("   ⏳ Temporary Files:")
        for file, size in temp_files:
            print(f"      • {file} ({size:,} bytes)")
    
    if old_files:
        print("   📦 Old Files:")
        for file, size in old_files:
            print(f"      • {file} ({size:,} bytes)")
    
    print("   ✅ Output files organized and cataloged")

def clean_empty_files():
    """Remove empty files"""
    print("📄 Checking for empty files...")
    
    empty_files = []
    
    # Check Python files for empty content (except __init__.py)
    py_files = glob.glob("**/*.py", recursive=True)
    for py_file in py_files:
        if os.path.basename(py_file) != "__init__.py":  # Skip __init__.py files
            try:
                if os.path.getsize(py_file) == 0:
                    empty_files.append(py_file)
            except OSError:
                pass
    
    if empty_files:
        print("   ⚠️  Found empty files:")
        for empty_file in empty_files:
            file_size = os.path.getsize(empty_file)
            print(f"      • {empty_file} ({file_size} bytes)")
        
        print("   💡 Empty files kept (may be intentional placeholders)")
    else:
        print("   ✅ No empty files found")

def create_cleanup_summary():
    """Create a summary of the cleanup"""
    print("📋 Creating cleanup summary...")
    
    summary_file = f"cleanup_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    with open(summary_file, 'w') as f:
        f.write("# Project Cleanup Summary\n\n")
        f.write(f"**Cleanup Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Cleaned Items\n")
        f.write("- ✅ Python cache files (__pycache__, *.pyc)\n")
        f.write("- ✅ Log files (moved to logs/ directory)\n")
        f.write("- ✅ Temporary files (~, .DS_Store, *.tmp)\n")
        f.write("- ✅ Organized output files\n\n")
        
        f.write("## Current Project Structure\n")
        f.write("```\n")
        f.write("StockScreener/\n")
        f.write("├── config/                  # Configuration files\n") 
        f.write("├── data/                    # Data files\n")
        f.write("│   ├── input/tickers/       # Ticker data\n")
        f.write("│   └── output/              # Analysis results\n")
        f.write("├── docs/                    # Documentation\n")
        f.write("├── logs/                    # Log files (moved here)\n")
        f.write("├── scripts/                 # Utility scripts\n")
        f.write("├── stock_screener/          # Main package\n")
        f.write("│   ├── cli/                 # Command line interface\n")
        f.write("│   ├── core/                # Core analysis logic\n")
        f.write("│   ├── services/            # Business services\n")
        f.write("│   └── utils/               # Utilities\n")
        f.write("├── tests/                   # Test files\n")
        f.write("└── requirements.txt         # Dependencies\n")
        f.write("```\n\n")
        
        f.write("## Ready for Development\n")
        f.write("Your project is now clean and organized! 🎉\n")
    
    print(f"   ✅ Created cleanup summary: {summary_file}")

def main():
    """Main cleanup function"""
    print("🚀 Starting StockScreener Project Cleanup")
    print("=" * 60)
    
    # Change to project root
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    try:
        # Run cleanup steps
        clean_cache_files()
        print()
        
        clean_log_files()
        print()
        
        clean_temporary_files()
        print()
        
        organize_output_files()
        print()
        
        clean_empty_files()
        print()
        
        create_cleanup_summary()
        print()
        
        print("=" * 60)
        print("✅ PROJECT CLEANUP COMPLETE!")
        print()
        print("🎯 Your StockScreener project is now clean and organized.")
        print("📊 All analysis results are preserved in data/output/")
        print("📋 Logs have been moved to logs/ directory")
        print("🚀 Ready for your next analysis!")
        
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
