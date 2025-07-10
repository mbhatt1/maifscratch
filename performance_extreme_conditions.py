#!/usr/bin/env python3
"""
MAIF Performance Testing Under Extreme Conditions
================================================

This script tests the MAIF system under extreme conditions to ensure it can handle
high load, large files, and concurrent operations.
"""

import os
import time
import random
import threading
import multiprocessing
import argparse
import json
import hashlib
import statistics
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Import MAIF modules
from maif.block_storage import BlockStorage
from maif.acid_transactions import ACIDTransactionManager, ACIDLevel, AcidMAIFEncoder
from maif.signature_verification import create_default_verifier, sign_block_data
from maif.aws_kms_integration import create_kms_verifier
from maif.aws_bedrock_integration import create_bedrock_integration


class ExtremeConditionTest:
    """Base class for extreme condition tests."""
    
    def __init__(self, test_name: str, output_dir: str = "results"):
        """
        Initialize test.
        
        Args:
            test_name: Test name
            output_dir: Output directory for results
        """
        self.test_name = test_name
        self.output_dir = output_dir
        self.results: Dict[str, Any] = {
            "test_name": test_name,
            "timestamp": time.time(),
            "metrics": {}
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def setup(self):
        """Set up test environment."""
        pass
    
    def run(self):
        """Run test."""
        pass
    
    def cleanup(self):
        """Clean up test environment."""
        pass
    
    def save_results(self):
        """Save test results."""
        # Add timestamp
        self.results["end_timestamp"] = time.time()
        
        # Save results to file
        result_file = os.path.join(self.output_dir, f"{self.test_name}_results.json")
        with open(result_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Generate plots if metrics are available
        self._generate_plots()
        
        print(f"Results saved to {result_file}")
    
    def _generate_plots(self):
        """Generate plots from test results."""
        if not self.results.get("metrics"):
            return
        
        for metric_name, metric_data in self.results["metrics"].items():
            if isinstance(metric_data, list) and len(metric_data) > 1:
                plt.figure(figsize=(10, 6))
                plt.plot(metric_data)
                plt.title(f"{self.test_name} - {metric_name}")
                plt.xlabel("Operation")
                plt.ylabel(metric_name)
                plt.grid(True)
                plt.savefig(os.path.join(self.output_dir, f"{self.test_name}_{metric_name}.png"))
                plt.close()


class HighConcurrencyTest(ExtremeConditionTest):
    """Test MAIF performance under high concurrency."""
    
    def __init__(self, num_threads: int = 100, operations_per_thread: int = 1000, 
                output_dir: str = "results"):
        """
        Initialize high concurrency test.
        
        Args:
            num_threads: Number of concurrent threads
            operations_per_thread: Number of operations per thread
            output_dir: Output directory for results
        """
        super().__init__(f"high_concurrency_{num_threads}x{operations_per_thread}", output_dir)
        self.num_threads = num_threads
        self.operations_per_thread = operations_per_thread
        self.maif_path = os.path.join(output_dir, f"concurrency_test_{num_threads}.maif")
        self.encoder = None
        self.thread_results: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
    
    def setup(self):
        """Set up test environment."""
        # Create MAIF encoder with ACID transactions
        self.encoder = AcidMAIFEncoder(
            maif_path=self.maif_path,
            acid_level=ACIDLevel.FULL_ACID
        )
        
        # Initialize thread results
        self.thread_results = []
    
    def _worker_thread(self, thread_id: int):
        """Worker thread function."""
        thread_start_time = time.time()
        operation_times = []
        
        for i in range(self.operations_per_thread):
            # Generate random data
            data_size = random.randint(100, 10000)
            data = os.urandom(data_size)
            
            # Start transaction
            start_time = time.time()
            transaction_id = self.encoder.begin_transaction()
            
            # Add binary block
            block_id = self.encoder.add_binary_block(
                data=data,
                block_type="test",
                metadata={
                    "thread_id": thread_id,
                    "operation": i,
                    "timestamp": time.time()
                }
            )
            
            # Commit transaction
            self.encoder.commit_transaction()
            
            # Record operation time
            end_time = time.time()
            operation_time = end_time - start_time
            operation_times.append(operation_time)
        
        # Record thread results
        thread_end_time = time.time()
        thread_duration = thread_end_time - thread_start_time
        thread_result = {
            "thread_id": thread_id,
            "operations": self.operations_per_thread,
            "duration": thread_duration,
            "avg_operation_time": statistics.mean(operation_times),
            "min_operation_time": min(operation_times),
            "max_operation_time": max(operation_times),
            "p95_operation_time": np.percentile(operation_times, 95),
            "operations_per_second": self.operations_per_thread / thread_duration
        }
        
        # Add to thread results
        with self.lock:
            self.thread_results.append(thread_result)
    
    def run(self):
        """Run high concurrency test."""
        print(f"Running high concurrency test with {self.num_threads} threads, {self.operations_per_thread} operations per thread")
        
        # Create and start threads
        threads = []
        for i in range(self.num_threads):
            thread = threading.Thread(target=self._worker_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in tqdm(threads, desc="Threads"):
            thread.join()
        
        # Calculate aggregate metrics
        all_ops_per_second = [r["operations_per_second"] for r in self.thread_results]
        all_avg_times = [r["avg_operation_time"] for r in self.thread_results]
        
        total_operations = self.num_threads * self.operations_per_thread
        total_duration = max(r["duration"] for r in self.thread_results)
        total_ops_per_second = total_operations / total_duration
        
        # Record results
        self.results["metrics"] = {
            "total_operations": total_operations,
            "total_duration": total_duration,
            "operations_per_second": total_ops_per_second,
            "thread_ops_per_second": all_ops_per_second,
            "thread_avg_times": all_avg_times,
            "avg_operation_time": statistics.mean(all_avg_times),
            "min_operation_time": min(r["min_operation_time"] for r in self.thread_results),
            "max_operation_time": max(r["max_operation_time"] for r in self.thread_results),
            "p95_operation_time": np.percentile([r["p95_operation_time"] for r in self.thread_results], 95)
        }
        
        print(f"High concurrency test completed: {total_ops_per_second:.2f} operations/second")
    
    def cleanup(self):
        """Clean up test environment."""
        # Close encoder
        if self.encoder:
            self.encoder._transaction_manager.close()
        
        # Remove test file
        if os.path.exists(self.maif_path):
            os.remove(self.maif_path)


class LargeFileTest(ExtremeConditionTest):
    """Test MAIF performance with large files."""
    
    def __init__(self, file_sizes: List[int], num_iterations: int = 5, 
                output_dir: str = "results"):
        """
        Initialize large file test.
        
        Args:
            file_sizes: List of file sizes to test (in MB)
            num_iterations: Number of iterations per file size
            output_dir: Output directory for results
        """
        super().__init__(f"large_file_test", output_dir)
        self.file_sizes = file_sizes
        self.num_iterations = num_iterations
        self.results_by_size: Dict[int, Dict[str, Any]] = {}
    
    def setup(self):
        """Set up test environment."""
        # Create test directory
        os.makedirs(os.path.join(self.output_dir, "large_files"), exist_ok=True)
    
    def run(self):
        """Run large file test."""
        print(f"Running large file test with sizes: {self.file_sizes} MB")
        
        for size_mb in self.file_sizes:
            print(f"Testing with {size_mb} MB files")
            
            # Convert MB to bytes
            size_bytes = size_mb * 1024 * 1024
            
            # Create test file path
            maif_path = os.path.join(self.output_dir, f"large_files/test_{size_mb}mb.maif")
            
            # Initialize metrics
            write_times = []
            read_times = []
            
            for i in range(self.num_iterations):
                # Generate random data
                data = os.urandom(size_bytes)
                
                # Create encoder
                encoder = AcidMAIFEncoder(
                    maif_path=maif_path,
                    acid_level=ACIDLevel.PERFORMANCE  # Use performance mode for large files
                )
                
                # Measure write time
                start_time = time.time()
                
                # Begin transaction
                transaction_id = encoder.begin_transaction()
                
                # Add binary block
                block_id = encoder.add_binary_block(
                    data=data,
                    block_type="large_file",
                    metadata={
                        "size_mb": size_mb,
                        "iteration": i,
                        "timestamp": time.time()
                    }
                )
                
                # Commit transaction
                encoder.commit_transaction()
                
                # Record write time
                write_time = time.time() - start_time
                write_times.append(write_time)
                
                # Close encoder
                encoder._transaction_manager.close()
                
                # Create block storage for reading
                storage = BlockStorage(maif_path)
                
                # Measure read time
                start_time = time.time()
                
                with storage:
                    # Read block
                    result = storage.get_block(block_id)
                
                # Record read time
                read_time = time.time() - start_time
                read_times.append(read_time)
                
                # Remove test file
                if os.path.exists(maif_path):
                    os.remove(maif_path)
            
            # Calculate metrics
            write_throughput = [(size_bytes / t) / (1024 * 1024) for t in write_times]  # MB/s
            read_throughput = [(size_bytes / t) / (1024 * 1024) for t in read_times]  # MB/s
            
            # Record results for this size
            self.results_by_size[size_mb] = {
                "size_mb": size_mb,
                "write_times": write_times,
                "read_times": read_times,
                "avg_write_time": statistics.mean(write_times),
                "avg_read_time": statistics.mean(read_times),
                "avg_write_throughput": statistics.mean(write_throughput),
                "avg_read_throughput": statistics.mean(read_throughput),
                "max_write_throughput": max(write_throughput),
                "max_read_throughput": max(read_throughput)
            }
        
        # Aggregate results
        self.results["metrics"] = {
            "file_sizes": self.file_sizes,
            "avg_write_throughput": [self.results_by_size[s]["avg_write_throughput"] for s in self.file_sizes],
            "avg_read_throughput": [self.results_by_size[s]["avg_read_throughput"] for s in self.file_sizes],
            "max_write_throughput": [self.results_by_size[s]["max_write_throughput"] for s in self.file_sizes],
            "max_read_throughput": [self.results_by_size[s]["max_read_throughput"] for s in self.file_sizes],
            "results_by_size": self.results_by_size
        }
        
        print("Large file test completed")
    
    def _generate_plots(self):
        """Generate plots from test results."""
        super()._generate_plots()
        
        # Generate throughput vs file size plot
        if self.file_sizes and "avg_write_throughput" in self.results["metrics"]:
            plt.figure(figsize=(10, 6))
            plt.plot(self.file_sizes, self.results["metrics"]["avg_write_throughput"], label="Write")
            plt.plot(self.file_sizes, self.results["metrics"]["avg_read_throughput"], label="Read")
            plt.title("Throughput vs File Size")
            plt.xlabel("File Size (MB)")
            plt.ylabel("Throughput (MB/s)")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.output_dir, f"{self.test_name}_throughput.png"))
            plt.close()


class HighThroughputTest(ExtremeConditionTest):
    """Test MAIF performance under high throughput conditions."""
    
    def __init__(self, duration: int = 60, block_size: int = 1024, 
                num_processes: int = 4, output_dir: str = "results"):
        """
        Initialize high throughput test.
        
        Args:
            duration: Test duration in seconds
            block_size: Block size in bytes
            num_processes: Number of concurrent processes
            output_dir: Output directory for results
        """
        super().__init__(f"high_throughput_{duration}s_{num_processes}p", output_dir)
        self.duration = duration
        self.block_size = block_size
        self.num_processes = num_processes
        self.maif_path = os.path.join(output_dir, f"throughput_test.maif")
        self.stop_event = multiprocessing.Event()
        self.process_results: List[Dict[str, Any]] = []
    
    def _worker_process(self, process_id: int, result_queue: multiprocessing.Queue):
        """Worker process function."""
        # Create encoder
        encoder = AcidMAIFEncoder(
            maif_path=f"{self.maif_path}_{process_id}",
            acid_level=ACIDLevel.PERFORMANCE  # Use performance mode for throughput
        )
        
        # Initialize metrics
        operations = 0
        start_time = time.time()
        operation_times = []
        
        # Run until stop event is set
        while not self.stop_event.is_set():
            # Generate random data
            data = os.urandom(self.block_size)
            
            # Measure operation time
            op_start_time = time.time()
            
            # Begin transaction
            transaction_id = encoder.begin_transaction()
            
            # Add binary block
            block_id = encoder.add_binary_block(
                data=data,
                block_type="throughput",
                metadata={
                    "process_id": process_id,
                    "operation": operations,
                    "timestamp": time.time()
                }
            )
            
            # Commit transaction
            encoder.commit_transaction()
            
            # Record operation time
            op_end_time = time.time()
            operation_time = op_end_time - op_start_time
            operation_times.append(operation_time)
            
            # Increment operation counter
            operations += 1
        
        # Calculate metrics
        end_time = time.time()
        duration = end_time - start_time
        
        # Put results in queue
        result_queue.put({
            "process_id": process_id,
            "operations": operations,
            "duration": duration,
            "operations_per_second": operations / duration,
            "throughput_bytes_per_second": (operations * self.block_size) / duration,
            "avg_operation_time": statistics.mean(operation_times) if operation_times else 0,
            "min_operation_time": min(operation_times) if operation_times else 0,
            "max_operation_time": max(operation_times) if operation_times else 0,
            "p95_operation_time": np.percentile(operation_times, 95) if len(operation_times) > 20 else 0
        })
        
        # Close encoder
        encoder._transaction_manager.close()
    
    def run(self):
        """Run high throughput test."""
        print(f"Running high throughput test for {self.duration} seconds with {self.num_processes} processes")
        
        # Create result queue
        result_queue = multiprocessing.Queue()
        
        # Create and start processes
        processes = []
        for i in range(self.num_processes):
            process = multiprocessing.Process(target=self._worker_process, args=(i, result_queue))
            processes.append(process)
            process.start()
        
        # Wait for duration
        time.sleep(self.duration)
        
        # Set stop event
        self.stop_event.set()
        
        # Wait for all processes to complete
        for process in processes:
            process.join()
        
        # Collect results
        self.process_results = []
        while not result_queue.empty():
            self.process_results.append(result_queue.get())
        
        # Calculate aggregate metrics
        total_operations = sum(r["operations"] for r in self.process_results)
        total_bytes = total_operations * self.block_size
        max_duration = max(r["duration"] for r in self.process_results)
        total_ops_per_second = total_operations / max_duration
        total_throughput = total_bytes / max_duration
        
        # Record results
        self.results["metrics"] = {
            "total_operations": total_operations,
            "total_bytes": total_bytes,
            "duration": max_duration,
            "operations_per_second": total_ops_per_second,
            "throughput_bytes_per_second": total_throughput,
            "throughput_mb_per_second": total_throughput / (1024 * 1024),
            "process_results": self.process_results
        }
        
        print(f"High throughput test completed: {total_ops_per_second:.2f} ops/s, {total_throughput / (1024 * 1024):.2f} MB/s")
    
    def cleanup(self):
        """Clean up test environment."""
        # Remove test files
        for i in range(self.num_processes):
            if os.path.exists(f"{self.maif_path}_{i}"):
                os.remove(f"{self.maif_path}_{i}")


class SecurityOverheadTest(ExtremeConditionTest):
    """Test MAIF performance with different security features enabled."""
    
    def __init__(self, num_operations: int = 1000, block_size: int = 1024,
                output_dir: str = "results"):
        """
        Initialize security overhead test.
        
        Args:
            num_operations: Number of operations per configuration
            block_size: Block size in bytes
            output_dir: Output directory for results
        """
        super().__init__(f"security_overhead_test", output_dir)
        self.num_operations = num_operations
        self.block_size = block_size
        self.maif_path = os.path.join(output_dir, f"security_test.maif")
        self.configurations = [
            {"name": "baseline", "acid": ACIDLevel.PERFORMANCE, "signature": False},
            {"name": "acid", "acid": ACIDLevel.FULL_ACID, "signature": False},
            {"name": "signature", "acid": ACIDLevel.PERFORMANCE, "signature": True},
            {"name": "full_security", "acid": ACIDLevel.FULL_ACID, "signature": True}
        ]
        self.config_results: Dict[str, Dict[str, Any]] = {}
    
    def run(self):
        """Run security overhead test."""
        print(f"Running security overhead test with {self.num_operations} operations per configuration")
        
        # Create signature verifier
        verifier = create_default_verifier()
        
        for config in self.configurations:
            print(f"Testing configuration: {config['name']}")
            
            # Create encoder
            encoder = AcidMAIFEncoder(
                maif_path=f"{self.maif_path}_{config['name']}",
                acid_level=config["acid"]
            )
            
            # Initialize metrics
            operation_times = []
            
            for i in range(self.num_operations):
                # Generate random data
                data = os.urandom(self.block_size)
                
                # Measure operation time
                start_time = time.time()
                
                # Begin transaction
                transaction_id = encoder.begin_transaction()
                
                # Create metadata
                metadata = {
                    "operation": i,
                    "timestamp": time.time()
                }
                
                # Add signature if enabled
                if config["signature"]:
                    signature_metadata = sign_block_data(verifier, data)
                    metadata["signature"] = signature_metadata
                
                # Add binary block
                block_id = encoder.add_binary_block(
                    data=data,
                    block_type="security_test",
                    metadata=metadata
                )
                
                # Commit transaction
                encoder.commit_transaction()
                
                # Record operation time
                end_time = time.time()
                operation_time = end_time - start_time
                operation_times.append(operation_time)
            
            # Calculate metrics
            avg_time = statistics.mean(operation_times)
            throughput = (self.num_operations * self.block_size) / sum(operation_times)
            
            # Record results for this configuration
            self.config_results[config["name"]] = {
                "config": config,
                "avg_operation_time": avg_time,
                "min_operation_time": min(operation_times),
                "max_operation_time": max(operation_times),
                "p95_operation_time": np.percentile(operation_times, 95),
                "throughput_bytes_per_second": throughput,
                "throughput_mb_per_second": throughput / (1024 * 1024),
                "operations_per_second": self.num_operations / sum(operation_times)
            }
            
            # Close encoder
            encoder._transaction_manager.close()
            
            # Remove test file
            if os.path.exists(f"{self.maif_path}_{config['name']}"):
                os.remove(f"{self.maif_path}_{config['name']}")
        
        # Record results
        self.results["metrics"] = {
            "configurations": [c["name"] for c in self.configurations],
            "avg_operation_times": [self.config_results[c["name"]]["avg_operation_time"] for c in self.configurations],
            "throughput_mb_per_second": [self.config_results[c["name"]]["throughput_mb_per_second"] for c in self.configurations],
            "operations_per_second": [self.config_results[c["name"]]["operations_per_second"] for c in self.configurations],
            "config_results": self.config_results
        }
        
        print("Security overhead test completed")
    
    def _generate_plots(self):
        """Generate plots from test results."""
        super()._generate_plots()
        
        # Generate overhead comparison plot
        if self.configurations and "avg_operation_times" in self.results["metrics"]:
            plt.figure(figsize=(10, 6))
            
            # Plot operation times
            plt.subplot(1, 2, 1)
            plt.bar(self.results["metrics"]["configurations"], self.results["metrics"]["avg_operation_times"])
            plt.title("Average Operation Time")
            plt.xlabel("Configuration")
            plt.ylabel("Time (s)")
            plt.grid(True)
            
            # Plot throughput
            plt.subplot(1, 2, 2)
            plt.bar(self.results["metrics"]["configurations"], self.results["metrics"]["throughput_mb_per_second"])
            plt.title("Throughput")
            plt.xlabel("Configuration")
            plt.ylabel("MB/s")
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"{self.test_name}_comparison.png"))
            plt.close()


def run_all_tests(args):
    """Run all extreme condition tests."""
    # Create results directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run high concurrency test
    if args.concurrency:
        concurrency_test = HighConcurrencyTest(
            num_threads=args.threads,
            operations_per_thread=args.operations,
            output_dir=args.output_dir
        )
        concurrency_test.setup()
        concurrency_test.run()
        concurrency_test.cleanup()
        concurrency_test.save_results()
    
    # Run large file test
    if args.large_files:
        large_file_test = LargeFileTest(
            file_sizes=args.file_sizes,
            num_iterations=args.iterations,
            output_dir=args.output_dir
        )
        large_file_test.setup()
        large_file_test.run()
        large_file_test.cleanup()
        large_file_test.save_results()
    
    # Run high throughput test
    if args.throughput:
        throughput_test = HighThroughputTest(
            duration=args.duration,
            block_size=args.block_size,
            num_processes=args.processes,
            output_dir=args.output_dir
        )
        throughput_test.setup()
        throughput_test.run()
        throughput_test.cleanup()
        throughput_test.save_results()
    
    # Run security overhead test
    if args.security:
        security_test = SecurityOverheadTest(
            num_operations=args.operations,
            block_size=args.block_size,
            output_dir=args.output_dir
        )
        security_test.setup()
        security_test.run()
        security_test.cleanup()
        security_test.save_results()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAIF Performance Testing Under Extreme Conditions")
    
    # General options
    parser.add_argument("--output-dir", default="results", help="Output directory for results")
    
    # Test selection
    parser.add_argument("--concurrency", action="store_true", help="Run high concurrency test")
    parser.add_argument("--large-files", action="store_true", help="Run large file test")
    parser.add_argument("--throughput", action="store_true", help="Run high throughput test")
    parser.add_argument("--security", action="store_true", help="Run security overhead test")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    # High concurrency test options
    parser.add_argument("--threads", type=int, default=100, help="Number of threads for concurrency test")
    parser.add_argument("--operations", type=int, default=1000, help="Number of operations per thread/configuration")
    
    # Large file test options
    parser.add_argument("--file-sizes", type=int, nargs="+", default=[10, 50, 100, 500, 1000], help="File sizes in MB")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations per file size")
    
    # High throughput test options
    parser.add_argument("--duration", type=int, default=60, help="Test duration in seconds")
    parser.add_argument("--block-size", type=int, default=1024, help="Block size in bytes")
    parser.add_argument("--processes", type=int, default=4, help="Number of processes for throughput test")
    
    args = parser.parse_args()
    
    # If --all is specified, run all tests
    if args.all:
        args.concurrency = True
        args.large_files = True
        args.throughput = True
        args.security = True
    
    # Run tests
    run_all_tests(args)