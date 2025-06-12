#!/usr/bin/env python3
"""
MAIF ACID Benchmark Runner
=========================

Simple script to run all ACID implementation benchmarks and compare results.

Usage:
    python run_acid_benchmarks.py

This will test:
1. Basic ACID implementation
2. Truly optimized ACID implementation  
3. Security-hardened ACID implementation

And show performance comparison results.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def run_benchmark(script_name, description):
    """Run a benchmark script and capture results."""
    print(f"\n{'='*60}")
    print(f"🚀 Running {description}")
    print(f"{'='*60}")
    
    script_path = Path(__file__).parent / script_name
    
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return False
    
    try:
        # Run the benchmark script
        start_time = time.time()
        result = subprocess.run([sys.executable, str(script_path)], 
                              capture_output=True, text=True, timeout=300)
        end_time = time.time()
        
        print(f"⏱️  Benchmark completed in {end_time - start_time:.1f} seconds")
        
        if result.returncode == 0:
            print("✅ Benchmark successful!")
            print("\n📊 Results:")
            print(result.stdout)
            return True
        else:
            print("❌ Benchmark failed!")
            print("Error output:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Benchmark timed out (5 minutes)")
        return False
    except Exception as e:
        print(f"❌ Error running benchmark: {e}")
        return False

def main():
    """Run all ACID benchmarks."""
    print("🔐 MAIF ACID Benchmark Suite")
    print("=" * 80)
    print("Testing all ACID implementations for performance and security")
    
    # Change to the script directory
    os.chdir(Path(__file__).parent)
    
    benchmarks = [
        ("test_acid_implementation.py", "Basic ACID Implementation"),
        ("maif/acid_truly_optimized.py", "Truly Optimized ACID Implementation"),
        ("maif/acid_secure.py", "Security-Hardened ACID Implementation"),
        ("test_final_acid_comparison.py", "Comprehensive ACID Comparison"),
        ("test_stream_access_control_comprehensive.py", "Stream Access Control Tests"),
        ("test_stream_tamper_prevention.py", "Stream Tamper Prevention Tests")
    ]
    
    results = {}
    
    for script, description in benchmarks:
        success = run_benchmark(script, description)
        results[description] = success
    
    # Summary
    print("\n" + "="*80)
    print("📈 BENCHMARK SUMMARY")
    print("="*80)
    
    for description, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {description}: {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print(f"\n🎯 Overall Results: {total_passed}/{total_tests} benchmarks passed")
    
    if total_passed == total_tests:
        print("🎉 All benchmarks completed successfully!")
        print("\n💡 Next Steps:")
        print("   • Review performance results above")
        print("   • Check security test results")
        print("   • Compare different ACID implementations")
        print("   • Choose appropriate ACID level for your use case")
        return 0
    else:
        print("⚠️  Some benchmarks failed - check error messages above")
        return 1

if __name__ == "__main__":
    sys.exit(main())