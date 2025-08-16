#!/usr/bin/env python3
"""
Prompts Library Migration Script
===============================

This script helps migrate from legacy prompt implementations to the new 
centralized prompts library. It can also be used to validate existing 
integrations and update prompt versions.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple

def find_legacy_prompt_patterns(directory: str) -> List[Tuple[str, str, int]]:
    """Find legacy prompt patterns in Python files"""
    
    patterns = [
        r'prompt\s*=\s*f?""".*?"""',  # Multi-line string prompts
        r'prompt\s*=\s*f?".*?"',      # Single-line string prompts
        r'def.*prompt.*\(.*\):',      # Function definitions with 'prompt' in name
    ]
    
    findings = []
    python_files = Path(directory).rglob("*.py")
    
    for file_path in python_files:
        if 'prompts' in str(file_path):  # Skip our prompts library files
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            for pattern in patterns:
                matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    findings.append((str(file_path), match.group(0)[:100], line_num))
                    
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")
    
    return findings

def check_prompts_library_usage(directory: str) -> Dict[str, bool]:
    """Check which files are already using the prompts library"""
    
    usage_status = {}
    python_files = Path(directory).rglob("*.py")
    
    for file_path in python_files:
        if 'prompts' in str(file_path) and 'prompts' in str(file_path).split('/')[-2:]:
            continue  # Skip prompts library itself
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check for prompts library imports
            has_import = bool(re.search(r'from.*prompts.*import', content) or 
                            re.search(r'import.*prompts', content))
            
            usage_status[str(file_path)] = has_import
            
        except Exception:
            continue
    
    return usage_status

def generate_migration_report(base_directory: str) -> None:
    """Generate a comprehensive migration report"""
    
    print("🔍 StockScreener Prompts Migration Report")
    print("=" * 50)
    
    # Find legacy patterns
    print("\n1. Legacy Prompt Patterns Found:")
    print("-" * 35)
    
    legacy_patterns = find_legacy_prompt_patterns(base_directory)
    
    if not legacy_patterns:
        print("   ✅ No legacy prompt patterns found!")
    else:
        for file_path, pattern, line_num in legacy_patterns[:10]:  # Show first 10
            relative_path = os.path.relpath(file_path, base_directory)
            print(f"   📄 {relative_path}:{line_num}")
            print(f"      {pattern}...")
            print()
        
        if len(legacy_patterns) > 10:
            print(f"   ... and {len(legacy_patterns) - 10} more patterns found")
    
    # Check current usage
    print("\n2. Prompts Library Usage Status:")
    print("-" * 35)
    
    usage_status = check_prompts_library_usage(base_directory)
    using_library = sum(1 for status in usage_status.values() if status)
    total_files = len(usage_status)
    
    print(f"   📊 Files using prompts library: {using_library}/{total_files}")
    
    if using_library > 0:
        print("   ✅ Files already migrated:")
        for file_path, is_using in usage_status.items():
            if is_using:
                relative_path = os.path.relpath(file_path, base_directory)
                print(f"      • {relative_path}")
    
    not_using = [f for f, status in usage_status.items() if not status]
    if not_using:
        print("   ⚠️  Files not yet migrated:")
        for file_path in not_using[:5]:  # Show first 5
            relative_path = os.path.relpath(file_path, base_directory)
            print(f"      • {relative_path}")
        if len(not_using) > 5:
            print(f"      ... and {len(not_using) - 5} more files")
    
    # Migration recommendations
    print("\n3. Migration Recommendations:")
    print("-" * 30)
    
    if legacy_patterns:
        print("   🔧 Actions needed:")
        print("      1. Replace string-based prompts with PromptManager calls")
        print("      2. Add prompts library imports")
        print("      3. Use PromptType enum for consistency")
        print("      4. Add parameter validation")
        print("      5. Implement fallback handling")
    else:
        print("   🎉 Migration appears complete!")
        print("      • All prompt patterns use the library")
        print("      • Consider adding more prompt types as needed")
    
    # Best practices
    print("\n4. Best Practices Checklist:")
    print("-" * 30)
    print("   ✅ Use PromptManager for all prompt generation")
    print("   ✅ Validate parameters before calling get_prompt()")
    print("   ✅ Implement graceful fallbacks for errors")
    print("   ✅ Use PromptType enum instead of strings")
    print("   ✅ Add logging for prompt generation")
    print("   ✅ Document custom prompt requirements")

def main():
    """Main migration script"""
    
    # Detect if we're in the StockScreener project
    current_dir = os.getcwd()
    if 'StockScreener' in current_dir:
        base_dir = current_dir
    else:
        base_dir = input("Enter path to StockScreener project: ").strip()
    
    if not os.path.exists(os.path.join(base_dir, 'stock_screener')):
        print("❌ Error: StockScreener project not found at specified path")
        return
    
    # Generate migration report
    generate_migration_report(base_dir)
    
    print(f"\n📋 Report completed for: {base_dir}")
    print("\n🚀 Next Steps:")
    print("   1. Review legacy patterns found")
    print("   2. Update imports in non-migrated files")
    print("   3. Test all functionality after migration")
    print("   4. Consider adding new prompt types for specific needs")

if __name__ == "__main__":
    main()
