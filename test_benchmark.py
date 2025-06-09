#!/usr/bin/env python3
"""
Quick test to verify the benchmark suite works correctly.
"""

import sys
import tempfile
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_basic_maif_functionality():
    """Test basic MAIF functionality before running full benchmark."""
    print("Testing basic MAIF functionality...")
    
    try:
        from maif.core import MAIFEncoder, MAIFDecoder
        
        # Test encoding
        encoder = MAIFEncoder()
        text_hash = encoder.add_text_block("Hello, MAIF!")
        embeddings_hash = encoder.add_embeddings_block([[1.0, 2.0, 3.0]])
        
        # Test building
        with tempfile.TemporaryDirectory() as tmpdir:
            maif_path = os.path.join(tmpdir, "test.maif")
            manifest_path = os.path.join(tmpdir, "test_manifest.json")
            
            encoder.build_maif(maif_path, manifest_path)
            
            # Test decoding
            decoder = MAIFDecoder(maif_path, manifest_path)
            texts = decoder.get_text_blocks()
            embeddings = decoder.get_embeddings()
            
            # Verify
            assert len(texts) == 1
            assert texts[0] == "Hello, MAIF!"
            assert len(embeddings) == 1
            assert embeddings[0] == [1.0, 2.0, 3.0]
            
            print("✓ Basic MAIF functionality works")
            
    except Exception as e:
        print(f"✗ Basic MAIF test failed: {e}")
        assert False, f"Basic MAIF test failed: {e}"

def test_benchmark_imports():
    """Test that benchmark imports work."""
    print("Testing benchmark imports...")
    
    try:
        from benchmarks.maif_benchmark_suite import MAIFBenchmarkSuite, BenchmarkResult
        print("✓ Benchmark imports work")
    except Exception as e:
        print(f"✗ Benchmark import failed: {e}")
        assert False, f"Benchmark import failed: {e}"

def test_quick_benchmark():
    """Run a very quick benchmark test."""
    print("Running quick benchmark test...")
    
    try:
        from benchmarks.maif_benchmark_suite import MAIFBenchmarkSuite
        
        # Create benchmark suite with minimal settings
        suite = MAIFBenchmarkSuite("test_results")
        suite.text_sizes = [1024]  # Just 1KB
        suite.embedding_counts = [10]  # Just 10 embeddings
        suite.file_counts = [5]  # Just 5 files
        
        # Run just one benchmark
        suite._benchmark_compression_ratios()
        
        if len(suite.results) > 0 and suite.results[0].success:
            print("✓ Quick benchmark test passed")
        else:
            print("✗ Quick benchmark test failed")
            assert False, "Quick benchmark test failed"
            
    except Exception as e:
        print(f"✗ Quick benchmark failed: {e}")
        assert False, f"Quick benchmark failed: {e}"

def main():
    """Run all tests."""
    print("MAIF Benchmark Test Suite")
    print("=" * 40)
    
    tests = [
        test_basic_maif_functionality,
        test_benchmark_imports,
        test_quick_benchmark
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! The benchmark suite is ready to run.")
        print("\nTo run the full benchmark suite:")
        print("  python run_benchmark.py")
        print("\nTo run a quick benchmark:")
        print("  python run_benchmark.py --quick")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())