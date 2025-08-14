#!/usr/bin/env python3
"""
Comprehensive test suite runner for Stock Screener
"""
import sys
import os
import subprocess
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_command(command, description):
    """Run a command and return success status"""
    print(f"\n{'='*60}")
    print(f"🧪 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(command)}")
    
    start_time = time.time()
    try:
        result = subprocess.run(
            command, 
            check=True, 
            capture_output=True, 
            text=True,
            cwd=project_root
        )
        end_time = time.time()
        
        print(f"✅ SUCCESS ({end_time - start_time:.2f}s)")
        if result.stdout.strip():
            print("\nOutput:")
            print(result.stdout)
        return True
        
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        print(f"❌ FAILED ({end_time - start_time:.2f}s)")
        
        if e.stdout:
            print("\nStdout:")
            print(e.stdout)
        if e.stderr:
            print("\nStderr:")  
            print(e.stderr)
        return False

def check_dependencies():
    """Check if required testing dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    try:
        import pytest
        import pandas
        import yfinance
        print("✅ All required dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def run_unit_tests():
    """Run unit tests"""
    return run_command(
        [sys.executable, "-m", "pytest", "tests/unit/", "-v", "--tb=short"],
        "Running Unit Tests"
    )

def run_integration_tests():
    """Run integration tests"""
    return run_command(
        [sys.executable, "-m", "pytest", "tests/integration/", "-v", "--tb=short", "-s"],
        "Running Integration Tests"
    )

def run_core_functionality_test():
    """Test core screener functionality"""
    return run_command(
        [sys.executable, "src/screener.py"],
        "Testing Core Screener Functionality"
    )

def run_all_tests():
    """Run all tests with coverage"""
    return run_command(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "--cov=src", "--cov=services", "--cov=utils", "--cov-report=term-missing"],
        "Running All Tests with Coverage"
    )

def main():
    """Run the comprehensive test suite"""
    print("🚀 Stock Screener - Comprehensive Test Suite")
    print("=" * 60)
    
    start_time = time.time()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    results = []
    
    # Run tests in order
    test_functions = [
        ("Unit Tests", run_unit_tests),
        ("Integration Tests", run_integration_tests), 
        ("Core Functionality", run_core_functionality_test),
        ("All Tests with Coverage", run_all_tests)
    ]
    
    for test_name, test_func in test_functions:
        success = test_func()
        results.append((test_name, success))
        
        if not success:
            print(f"\n⚠️ {test_name} failed, but continuing...")
    
    # Final summary
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:<25}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print(f"Time: {total_time:.2f} seconds")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Your stock screener is working perfectly!")
        sys.exit(0)
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please check the output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
