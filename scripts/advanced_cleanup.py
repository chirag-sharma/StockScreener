#!/usr/bin/env python3
"""
Advanced Project Cleanup - Root Directory Organization
=====================================================

This script organizes remaining files in the root directory by:
1. Moving utility scripts to appropriate folders
2. Organizing documentation files
3. Removing empty/redundant files
4. Creating a final clean structure
"""

import os
import shutil
from pathlib import Path

def organize_utility_scripts():
    """Move utility scripts to scripts directory"""
    print("🔧 Organizing utility scripts...")
    
    utility_scripts = [
        'analyze_stability.py',
        'check_excel.py', 
        'prediction_consistency.py',
        'update_ticker_dates.py',
        'clean_project.py'
    ]
    
    moved = 0
    for script in utility_scripts:
        if os.path.exists(script):
            dest = f"scripts/{script}"
            shutil.move(script, dest)
            print(f"   Moved: {script} → {dest}")
            moved += 1
    
    print(f"   ✅ Moved {moved} utility scripts")

def organize_documentation():
    """Organize documentation files"""
    print("📚 Organizing documentation...")
    
    # Move reports to docs
    doc_files = [
        'COMPREHENSIVE_AI_ANALYSIS_REPORT.md',
        'README_UNIFIED.md', 
        'TRANSFORMATION_SUMMARY.md',
        'cleanup_summary_20250813_222725.md'
    ]
    
    moved = 0
    for doc_file in doc_files:
        if os.path.exists(doc_file):
            # Check if file is empty
            if os.path.getsize(doc_file) == 0:
                os.remove(doc_file)
                print(f"   Removed empty: {doc_file}")
            else:
                dest = f"docs/{doc_file}"
                shutil.move(doc_file, dest)
                print(f"   Moved: {doc_file} → {dest}")
                moved += 1
    
    print(f"   ✅ Processed {moved} documentation files")

def organize_shell_scripts():
    """Organize shell scripts"""
    print("🐚 Organizing shell scripts...")
    
    shell_scripts = [
        'run_screener.sh',
        'setup_api_keys.sh'
    ]
    
    moved = 0
    for script in shell_scripts:
        if os.path.exists(script):
            # Check if file is empty
            if os.path.getsize(script) == 0:
                os.remove(script)
                print(f"   Removed empty: {script}")
            else:
                dest = f"scripts/{script}"
                shutil.move(script, dest)
                print(f"   Moved: {script} → {dest}")
                moved += 1
    
    print(f"   ✅ Processed {moved} shell scripts")

def remove_redundant_files():
    """Remove redundant or empty files"""
    print("🗑️  Removing redundant files...")
    
    # Check for redundant __init__.py in root
    if os.path.exists("__init__.py"):
        # Root __init__.py is usually not needed
        os.remove("__init__.py")
        print("   Removed: __init__.py (not needed in root)")
    
    print("   ✅ Cleaned redundant files")

def organize_requirements():
    """Organize requirements files"""
    print("📦 Organizing requirements...")
    
    # Keep main requirements.txt in root, but organize others
    req_files = [
        'requirements-prediction.txt'
    ]
    
    # Create requirements directory if it doesn't exist
    os.makedirs("config", exist_ok=True)
    
    moved = 0
    for req_file in req_files:
        if os.path.exists(req_file):
            dest = f"config/{req_file}"
            shutil.move(req_file, dest)
            print(f"   Moved: {req_file} → {dest}")
            moved += 1
    
    print(f"   ✅ Organized {moved} requirements files")

def final_root_check():
    """Show final root directory contents"""
    print("📋 Final root directory contents:")
    
    root_files = [f for f in os.listdir(".") if os.path.isfile(f)]
    root_dirs = [d for d in os.listdir(".") if os.path.isdir(d) and not d.startswith('.')]
    
    print("   📁 Directories:")
    for directory in sorted(root_dirs):
        file_count = len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])
        print(f"      • {directory}/ ({file_count} files)")
    
    print("   📄 Files:")
    essential_files = []
    for file in sorted(root_files):
        if not file.startswith('.'):
            size = os.path.getsize(file)
            essential_files.append((file, size))
            print(f"      • {file} ({size:,} bytes)")
    
    print(f"   ✅ Root directory now has {len(essential_files)} essential files")

def main():
    """Main organization function"""
    print("🧹 Advanced Root Directory Cleanup")
    print("=" * 50)
    
    # Change to project root
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    try:
        organize_utility_scripts()
        print()
        
        organize_documentation()
        print()
        
        organize_shell_scripts()
        print()
        
        organize_requirements()
        print()
        
        remove_redundant_files()
        print()
        
        final_root_check()
        print()
        
        print("=" * 50)
        print("✅ ADVANCED CLEANUP COMPLETE!")
        print()
        print("🎯 Root directory is now minimal and organized:")
        print("   • Configuration files in config/")
        print("   • Scripts in scripts/")
        print("   • Documentation in docs/")
        print("   • Only essential files remain in root")
        print()
        print("🚀 Professional project structure achieved!")
        
    except Exception as e:
        print(f"❌ Error during organization: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
