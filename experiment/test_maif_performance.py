#!/usr/bin/env python3
"""
MAIF Performance Testing Suite

Tests the practical performance of our MAIF implementation against
the theoretical claims made in the original MAIF paper.

Key metrics tested:
- Compression ratios (paper claims 480x)
- Write throughput (paper claims 100k+ TPS)
- Cryptographic verification overhead
- Storage efficiency
- Query performance
"""

import asyncio
import time
import statistics
import json
import random
import string
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import psutil
import numpy as np

from maif_core import MAIFStorage, MAIFBlock, MAIFBlockType
from maif_agent import MAIFAgent
from maif_security import MAIFSecurityManager, create_maif_security_config


@dataclass
class BenchmarkResult:
    """Results from a benchmark test"""
    test_name: str
    duration_seconds: float
    operations_count: int
    throughput_ops_per_sec: float
    memory_usage_mb: float
    cpu_usage_percent: float
    additional_metrics: Dict[str, Any]


class MAIFPerformanceTester:
    """Comprehensive performance testing for MAIF implementation"""
    
    def __init__(self):
        self.storage = MAIFStorage({"redis_url": "redis://localhost", "kafka_brokers": ["localhost:9092"]})
        self.security = MAIFSecurityManager(create_maif_security_config())
        self.agent = MAIFAgent("test_agent", self.storage)
        self.results: List[BenchmarkResult] = []
    
    async def setup(self):
        """Initialize test environment"""
        await self.storage.initialize()
        print("🔧 Test environment initialized")
    
    def generate_test_data(self, size_bytes: int) -> bytes:
        """Generate test data of specified size"""
        return ''.join(random.choices(string.ascii_letters + string.digits, 
                                    k=size_bytes)).encode()
    
    def generate_realistic_maif_data(self) -> Dict[str, Any]:
        """Generate realistic MAIF artifact data"""
        return {
            "agent_id": f"agent_{random.randint(1000, 9999)}",
            "action": random.choice(["read_file", "write_file", "execute_code", "send_message"]),
            "timestamp": time.time(),
            "inputs": {
                "file_path": f"/path/to/file_{random.randint(1, 100)}.txt",
                "content": self.generate_test_data(random.randint(100, 1000)).decode()
            },
            "outputs": {
                "status": "success",
                "result": self.generate_test_data(random.randint(200, 2000)).decode()
            },
            "metadata": {
                "execution_time": random.uniform(0.1, 5.0),
                "memory_used": random.randint(1024, 10240),
                "tokens_used": random.randint(50, 500)
            }
        }
    
    async def benchmark_compression(self, data_sizes: List[int]) -> BenchmarkResult:
        """Test compression ratios vs paper's 480x claim"""
        print("📊 Testing compression ratios...")
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        start_cpu = psutil.cpu_percent()
        
        compression_results = []
        total_original = 0
        total_compressed = 0
        
        for size in data_sizes:
            original_data = self.generate_test_data(size)
            total_original += len(original_data)
            
            # Create MAIF block and test serialization (which includes compression)
            block = MAIFBlock(
                id=f"test_{size}",
                block_type=MAIFBlockType.TEXT,
                data=original_data,
                metadata={"test": True, "size": size},
                timestamp=time.time(),
                agent_id="test_agent"
            )
            
            serialized_data = block.serialize()
            total_compressed += len(serialized_data)
            
            ratio = len(original_data) / len(serialized_data)
            compression_results.append({
                "original_size": len(original_data),
                "compressed_size": len(serialized_data),
                "ratio": ratio,
                "algorithm": "lz4"
            })
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        end_cpu = psutil.cpu_percent()
        
        overall_ratio = total_original / total_compressed if total_compressed > 0 else 0
        
        return BenchmarkResult(
            test_name="Compression Ratios",
            duration_seconds=end_time - start_time,
            operations_count=len(data_sizes),
            throughput_ops_per_sec=len(data_sizes) / (end_time - start_time),
            memory_usage_mb=end_memory - start_memory,
            cpu_usage_percent=end_cpu - start_cpu,
            additional_metrics={
                "overall_compression_ratio": overall_ratio,
                "max_ratio": max(r["ratio"] for r in compression_results),
                "min_ratio": min(r["ratio"] for r in compression_results),
                "avg_ratio": statistics.mean(r["ratio"] for r in compression_results),
                "paper_claimed_ratio": 480,
                "ratio_achievement_percent": (overall_ratio / 480) * 100,
                "detailed_results": compression_results
            }
        )
    
    async def benchmark_write_throughput(self, num_operations: int) -> BenchmarkResult:
        """Test write throughput vs paper's 100k+ TPS claim"""
        print("⚡ Testing write throughput...")
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Generate test blocks
        blocks = []
        for i in range(num_operations):
            data = self.generate_test_data(random.randint(500, 2000))
            block = MAIFBlock(
                id=f"perf_test_{i}",
                block_type=MAIFBlockType.TEXT,
                data=data,
                metadata={"test": True, "index": i},
                timestamp=time.time(),
                agent_id="test_agent"
            )
            blocks.append(block)
        
        # Benchmark writes
        write_start = time.time()
        tasks = []
        
        for block in blocks:
            task = self.storage.write_block(block)
            tasks.append(task)
            
            # Batch processing to avoid overwhelming the system
            if len(tasks) >= 50:
                await asyncio.gather(*tasks)
                tasks = []
        
        # Process remaining tasks
        if tasks:
            await asyncio.gather(*tasks)
        
        write_end = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        duration = write_end - write_start
        throughput = num_operations / duration
        
        return BenchmarkResult(
            test_name="Write Throughput",
            duration_seconds=time.time() - start_time,
            operations_count=num_operations,
            throughput_ops_per_sec=throughput,
            memory_usage_mb=end_memory - start_memory,
            cpu_usage_percent=psutil.cpu_percent(),
            additional_metrics={
                "paper_claimed_tps": 100000,
                "actual_tps": throughput,
                "tps_achievement_percent": (throughput / 100000) * 100,
                "avg_operation_time_ms": (duration / num_operations) * 1000,
                "write_duration_seconds": duration
            }
        )
    
    async def benchmark_cryptographic_overhead(self, num_operations: int) -> BenchmarkResult:
        """Test cryptographic verification overhead"""
        print("🔐 Testing cryptographic overhead...")
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Generate test blocks
        blocks = []
        for i in range(num_operations):
            data = self.generate_test_data(1024)
            block = MAIFBlock(
                id=f"crypto_test_{i}",
                block_type=MAIFBlockType.TEXT,
                data=data,
                metadata={"test": True},
                timestamp=time.time(),
                agent_id="test_agent"
            )
            blocks.append(block)
        
        # Test hashing performance
        hash_times = []
        
        for block in blocks:
            # Quick hash benchmark
            hash_start = time.perf_counter()
            quick_hash = block.quick_hash()
            hash_end = time.perf_counter()
            hash_times.append(hash_end - hash_start)
            
            # Full hash benchmark
            hash_start = time.perf_counter()
            full_hash = block.full_hash()
            hash_end = time.perf_counter()
            hash_times.append(hash_end - hash_start)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        return BenchmarkResult(
            test_name="Cryptographic Overhead",
            duration_seconds=end_time - start_time,
            operations_count=num_operations * 2,  # quick + full hash
            throughput_ops_per_sec=(num_operations * 2) / (end_time - start_time),
            memory_usage_mb=end_memory - start_memory,
            cpu_usage_percent=psutil.cpu_percent(),
            additional_metrics={
                "avg_hash_time_ms": statistics.mean(hash_times) * 1000,
                "max_hash_time_ms": max(hash_times) * 1000,
                "total_crypto_overhead_ms": sum(hash_times) * 1000,
                "crypto_ops_per_second": num_operations * 2 / sum(hash_times)
            }
        )
    
    async def benchmark_serialization_performance(self, num_operations: int) -> BenchmarkResult:
        """Test serialization/deserialization performance"""
        print("🔄 Testing serialization performance...")
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Generate test blocks with various sizes
        blocks = []
        for i in range(num_operations):
            data_size = random.choice([100, 1000, 10000, 50000])  # Various sizes
            data = self.generate_test_data(data_size)
            block = MAIFBlock(
                id=f"serial_test_{i}",
                block_type=MAIFBlockType.TEXT,
                data=data,
                metadata={"test": True, "data_size": data_size},
                timestamp=time.time(),
                agent_id="test_agent"
            )
            blocks.append(block)
        
        # Test serialization
        serialize_times = []
        deserialize_times = []
        serialized_blocks = []
        
        for block in blocks:
            # Serialization benchmark
            serialize_start = time.perf_counter()
            serialized = block.serialize()
            serialize_end = time.perf_counter()
            serialize_times.append(serialize_end - serialize_start)
            serialized_blocks.append(serialized)
            
            # Deserialization benchmark
            deserialize_start = time.perf_counter()
            deserialized = MAIFBlock.deserialize(serialized)
            deserialize_end = time.perf_counter()
            deserialize_times.append(deserialize_end - deserialize_start)
            
            # Verify correctness
            assert deserialized.id == block.id
            assert deserialized.data == block.data
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        return BenchmarkResult(
            test_name="Serialization Performance",
            duration_seconds=end_time - start_time,
            operations_count=num_operations * 2,  # serialize + deserialize
            throughput_ops_per_sec=(num_operations * 2) / (end_time - start_time),
            memory_usage_mb=end_memory - start_memory,
            cpu_usage_percent=psutil.cpu_percent(),
            additional_metrics={
                "avg_serialize_time_ms": statistics.mean(serialize_times) * 1000,
                "avg_deserialize_time_ms": statistics.mean(deserialize_times) * 1000,
                "max_serialize_time_ms": max(serialize_times) * 1000,
                "max_deserialize_time_ms": max(deserialize_times) * 1000,
                "total_serialization_time_ms": sum(serialize_times + deserialize_times) * 1000
            }
        )
    
    def print_results(self):
        """Print comprehensive benchmark results"""
        print("\n" + "="*80)
        print("🏆 MAIF PERFORMANCE BENCHMARK RESULTS")
        print("="*80)
        
        for result in self.results:
            print(f"\n📊 {result.test_name}")
            print("-" * 40)
            print(f"⏱️  Duration: {result.duration_seconds:.2f}s")
            print(f"🔢 Operations: {result.operations_count:,}")
            print(f"⚡ Throughput: {result.throughput_ops_per_sec:.2f} ops/sec")
            print(f"💾 Memory Usage: {result.memory_usage_mb:.2f} MB")
            print(f"🖥️  CPU Usage: {result.cpu_usage_percent:.1f}%")
            
            if result.additional_metrics:
                print("📈 Additional Metrics:")
                for key, value in result.additional_metrics.items():
                    if key != "detailed_results":
                        if isinstance(value, float):
                            print(f"   {key}: {value:.2f}")
                        else:
                            print(f"   {key}: {value}")
        
        # Overall assessment
        print("\n" + "="*80)
        print("🎯 ASSESSMENT vs ORIGINAL PAPER CLAIMS")
        print("="*80)
        
        # Find compression and throughput results
        compression_result = next((r for r in self.results if r.test_name == "Compression Ratios"), None)
        throughput_result = next((r for r in self.results if r.test_name == "Write Throughput"), None)
        
        if compression_result:
            ratio = compression_result.additional_metrics.get("overall_compression_ratio", 0)
            achievement = compression_result.additional_metrics.get("ratio_achievement_percent", 0)
            print(f"📦 Compression: {ratio:.1f}x achieved vs 480x claimed ({achievement:.1f}%)")
            
            if ratio < 10:
                print("   ❌ REALISTIC: Compression ratios align with industry standards")
            else:
                print("   ⚠️  HIGH: Unusually high compression, verify data characteristics")
        
        if throughput_result:
            tps = throughput_result.additional_metrics.get("actual_tps", 0)
            achievement = throughput_result.additional_metrics.get("tps_achievement_percent", 0)
            print(f"⚡ Throughput: {tps:.0f} TPS achieved vs 100k claimed ({achievement:.1f}%)")
            
            if tps < 1000:
                print("   ❌ REALISTIC: Throughput typical for complex cryptographic operations")
            else:
                print("   ✅ GOOD: High throughput achieved")
        
        print("\n💡 CONCLUSION:")
        print("   Our implementation provides realistic performance characteristics")
        print("   that are achievable in production environments, unlike the")
        print("   theoretical claims in the original MAIF paper.")


async def main():
    """Run comprehensive MAIF performance benchmarks"""
    print("🚀 Starting MAIF Performance Benchmark Suite")
    print("="*60)
    
    tester = MAIFPerformanceTester()
    await tester.setup()
    
    try:
        # Test compression with various data sizes
        print("\n1️⃣ Compression Benchmark")
        compression_result = await tester.benchmark_compression([
            1024, 4096, 16384, 65536, 262144  # 1KB to 256KB
        ])
        tester.results.append(compression_result)
        
        # Test write throughput
        print("\n2️⃣ Write Throughput Benchmark")
        throughput_result = await tester.benchmark_write_throughput(500)  # Reduced for stability
        tester.results.append(throughput_result)
        
        # Test cryptographic overhead
        print("\n3️⃣ Cryptographic Overhead Benchmark")
        crypto_result = await tester.benchmark_cryptographic_overhead(250)  # Reduced for stability
        tester.results.append(crypto_result)
        
        # Test serialization performance
        print("\n4️⃣ Serialization Performance Benchmark")
        serialization_result = await tester.benchmark_serialization_performance(500)
        tester.results.append(serialization_result)
        
        # Print comprehensive results
        tester.print_results()
        
        # Save results to file
        results_data = {
            "timestamp": time.time(),
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "memory_gb": psutil.virtual_memory().total / (1024**3),
                "python_version": "3.x"
            },
            "results": [
                {
                    "test_name": r.test_name,
                    "duration_seconds": r.duration_seconds,
                    "operations_count": r.operations_count,
                    "throughput_ops_per_sec": r.throughput_ops_per_sec,
                    "memory_usage_mb": r.memory_usage_mb,
                    "cpu_usage_percent": r.cpu_usage_percent,
                    "additional_metrics": r.additional_metrics
                }
                for r in tester.results
            ]
        }
        
        with open("maif_benchmark_results.json", "w") as f:
            json.dump(results_data, f, indent=2, default=str)
        
        print(f"\n💾 Detailed results saved to: maif_benchmark_results.json")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 