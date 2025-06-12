#!/usr/bin/env python3
"""
Test script to verify streaming throughput improvements.
"""

import os
import sys
import time
import tempfile
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from maif.core import MAIFEncoder
from maif.streaming import MAIFStreamReader, StreamingConfig
from maif.streaming_optimized import HighThroughputMAIFReader, OptimizedStreamingConfig

def generate_test_data(size_mb: int = 50):
    """Generate test data of specified size."""
    data = []
    chunk_size = 1024 * 1024  # 1MB chunks
    
    for i in range(size_mb):
        # Generate varied content for realistic compression
        chunk = f"Test data block {i:04d} " * (chunk_size // 20)
        chunk = chunk[:chunk_size]  # Ensure exact size
        data.append(chunk)
    
    return data

def create_test_file(tmpdir: str, size_mb: int = 50):
    """Create a test MAIF file."""
    print(f"Creating {size_mb}MB test file...")
    
    encoder = MAIFEncoder()
    test_data = generate_test_data(size_mb)
    
    for i, chunk in enumerate(test_data):
        encoder.add_text_block(chunk, metadata={"chunk_id": i})
    
    maif_path = os.path.join(tmpdir, "throughput_test.maif")
    manifest_path = os.path.join(tmpdir, "throughput_test_manifest.json")
    
    encoder.build_maif(maif_path, manifest_path)
    
    file_size = os.path.getsize(maif_path)
    print(f"Created test file: {file_size / (1024*1024):.1f}MB")
    
    return maif_path

def test_original_streaming(maif_path: str):
    """Test original streaming implementation."""
    print("\n=== Testing Original Streaming ===")
    
    config = StreamingConfig(
        chunk_size=8192,  # Original small chunk size
        max_workers=4,
        buffer_size=65536  # Original small buffer
    )
    
    start_time = time.time()
    bytes_read = 0
    blocks_read = 0
    
    try:
        with MAIFStreamReader(maif_path, config) as reader:
            for block_type, data in reader.stream_blocks():
                bytes_read += len(data)
                blocks_read += 1
        
        duration = time.time() - start_time
        throughput_mbps = (bytes_read / (1024 * 1024)) / duration if duration > 0 else 0
        
        print(f"Original Implementation:")
        print(f"  Blocks read: {blocks_read}")
        print(f"  Bytes read: {bytes_read:,}")
        print(f"  Duration: {duration:.3f}s")
        print(f"  Throughput: {throughput_mbps:.1f} MB/s")
        
        return throughput_mbps
        
    except Exception as e:
        print(f"Original streaming failed: {e}")
        return 0

def test_optimized_streaming(maif_path: str):
    """Test optimized streaming implementation."""
    print("\n=== Testing Optimized Streaming ===")
    
    config = OptimizedStreamingConfig(
        chunk_size=1024 * 1024,  # 1MB chunks
        max_workers=16,
        buffer_size=16 * 1024 * 1024,  # 16MB buffer
        use_memory_mapping=True,
        batch_size=32
    )
    
    methods = [
        ("Memory Mapped", "stream_blocks_ultra_fast"),
        ("Parallel Optimized", "stream_blocks_parallel_optimized")
    ]
    
    best_throughput = 0
    
    for method_name, method_func in methods:
        try:
            start_time = time.time()
            bytes_read = 0
            blocks_read = 0
            
            with HighThroughputMAIFReader(maif_path, config) as reader:
                stream_method = getattr(reader, method_func)
                for block_type, data in stream_method():
                    bytes_read += len(data)
                    blocks_read += 1
            
            duration = time.time() - start_time
            throughput_mbps = (bytes_read / (1024 * 1024)) / duration if duration > 0 else 0
            
            print(f"{method_name}:")
            print(f"  Blocks read: {blocks_read}")
            print(f"  Bytes read: {bytes_read:,}")
            print(f"  Duration: {duration:.3f}s")
            print(f"  Throughput: {throughput_mbps:.1f} MB/s")
            
            if throughput_mbps > best_throughput:
                best_throughput = throughput_mbps
                
        except Exception as e:
            print(f"{method_name} failed: {e}")
    
    return best_throughput

def main():
    """Main test function."""
    print("🚀 MAIF Streaming Throughput Test")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test file
        maif_path = create_test_file(tmpdir, size_mb=50)
        
        # Test original implementation
        original_throughput = test_original_streaming(maif_path)
        
        # Test optimized implementation
        optimized_throughput = test_optimized_streaming(maif_path)
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 RESULTS SUMMARY")
        print("=" * 50)
        print(f"Original Throughput:  {original_throughput:.1f} MB/s")
        print(f"Optimized Throughput: {optimized_throughput:.1f} MB/s")
        
        if optimized_throughput > 0 and original_throughput > 0:
            improvement = optimized_throughput / original_throughput
            print(f"Improvement Factor:   {improvement:.1f}x")
        
        # Check if we meet the 500 MB/s target
        target_throughput = 500.0
        print(f"\nTarget: {target_throughput} MB/s")
        
        if optimized_throughput >= target_throughput:
            print("✅ TARGET ACHIEVED!")
        else:
            gap = target_throughput - optimized_throughput
            print(f"❌ Gap: {gap:.1f} MB/s remaining")
            
            # Provide recommendations
            print("\n🔧 RECOMMENDATIONS:")
            if optimized_throughput < 100:
                print("- Check if memory mapping is working")
                print("- Verify file system performance")
                print("- Consider SSD storage")
            elif optimized_throughput < 300:
                print("- Increase buffer size further")
                print("- Try different batch sizes")
                print("- Check CPU utilization")
            else:
                print("- Fine-tune worker count")
                print("- Consider async I/O")
                print("- Profile for remaining bottlenecks")

if __name__ == "__main__":
    main()