"""
MAIF Performance Benchmark
=========================

Comprehensive performance benchmarks for all MAIF implementations:
- Core MAIF
- Event Sourcing
- Columnar Storage
- Dynamic Version Management
- Adaptation Rules Engine
- Enhanced Integration

This benchmark measures performance metrics for various operations
and provides comparative analysis between different implementations.
It also benchmarks performance across different modalities and
compliance levels (raw, ACID, ACID+security).
"""

import os
import time
import json
import random
import tempfile
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from tabulate import tabulate
import gc
import psutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import MAIF components
from maif.core import MAIFEncoder, MAIFDecoder
from maif.event_sourcing import EventLog, EventSourcedMAIF
from maif.columnar_storage import ColumnarFile, ColumnType, EncodingType, CompressionType
from maif.version_management import SchemaRegistry, VersionManager, Schema, SchemaField
from maif.adaptation_rules import AdaptationRulesEngine
from maif.integration_enhanced import EnhancedMAIF
from maif.lifecycle_management_enhanced import EnhancedSelfGoverningMAIF
from maif.acid_transactions import AcidMAIFEncoder
from maif.security import SecurityManager
from maif.acid_truly_optimized import TrulyOptimizedAcidMAIF

# Benchmark configuration
DEFAULT_CONFIG = {
    "num_blocks": 1000,
    "block_size": 1024,  # bytes
    "num_runs": 3,
    "implementations": [
        "core",
        "event_sourcing",
        "columnar_storage",
        "version_management",
        "adaptation_rules",
        "enhanced"
    ],
    "compliance_levels": [
        "raw",
        "acid",
        "acid_security"
    ],
    "modalities": [
        "text",
        "binary",
        "embeddings",
        "video",
        "cross_modal"
    ],
    "operations": [
        "write",
        "read",
        "search",
        "update",
        "delete"
    ]
}

class PerformanceBenchmark:
    """Performance benchmark for MAIF implementations."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize benchmark.
        
        Args:
            config: Benchmark configuration
        """
        self.config = config or DEFAULT_CONFIG
        self.results = {}
        self.temp_dir = Path(tempfile.mkdtemp(prefix="maif_benchmark_"))
        logger.info(f"Using temporary directory: {self.temp_dir}")
    
    def run_benchmarks(self):
        """Run all benchmarks."""
        logger.info("Starting MAIF performance benchmarks")
        
        # Benchmark implementations
        for impl in self.config["implementations"]:
            logger.info(f"Benchmarking {impl} implementation")
            self.results[impl] = self._benchmark_implementation(impl)
        
        # Benchmark compliance levels
        for level in self.config["compliance_levels"]:
            logger.info(f"Benchmarking {level} compliance level")
            self.results[f"compliance_{level}"] = self._benchmark_compliance(level)
        
        # Benchmark modalities
        for modality in self.config["modalities"]:
            logger.info(f"Benchmarking {modality} modality")
            self.results[f"modality_{modality}"] = self._benchmark_modality(modality)
        
        logger.info("All benchmarks completed")
        return self.results
    
    def _benchmark_implementation(self, implementation: str) -> Dict[str, Any]:
        """
        Benchmark specific implementation.
        
        Args:
            implementation: Implementation name
            
        Returns:
            Benchmark results
        """
        results = {}
        
        # Create implementation instance
        instance = self._create_instance(implementation)
        
        # Run operation benchmarks
        for operation in self.config["operations"]:
            if hasattr(self, f"_benchmark_{operation}"):
                logger.info(f"  Running {operation} benchmark")
                
                # Run multiple times and take average
                times = []
                memory_usages = []
                
                for run in range(self.config["num_runs"]):
                    # Clear memory before each run
                    gc.collect()
                    
                    # Measure time and memory
                    start_time = time.time()
                    start_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
                    
                    # Run benchmark
                    benchmark_method = getattr(self, f"_benchmark_{operation}")
                    benchmark_method(instance, implementation)
                    
                    # Measure end time and memory
                    end_time = time.time()
                    end_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
                    
                    # Calculate metrics
                    elapsed_time = end_time - start_time
                    memory_usage = end_memory - start_memory
                    
                    times.append(elapsed_time)
                    memory_usages.append(memory_usage)
                
                # Calculate average metrics
                avg_time = sum(times) / len(times)
                avg_memory = sum(memory_usages) / len(memory_usages)
                
                # Store results
                results[operation] = {
                    "time": avg_time,
                    "memory": avg_memory,
                    "throughput": self.config["num_blocks"] / avg_time if avg_time > 0 else 0
                }
        
        # Clean up
        self._cleanup_instance(instance, implementation)
        
        return results
    
    def _benchmark_compliance(self, compliance_level: str) -> Dict[str, Any]:
        """
        Benchmark specific compliance level.
        
        Args:
            compliance_level: Compliance level (raw, acid, acid_security)
            
        Returns:
            Benchmark results
        """
        results = {}
        
        # Create instance based on compliance level
        if compliance_level == "raw":
            instance = MAIFEncoder()
        elif compliance_level == "acid":
            instance = AcidMAIFEncoder()
        elif compliance_level == "acid_security":
            instance = TrulyOptimizedAcidMAIF(enable_security=True)
        else:
            raise ValueError(f"Unknown compliance level: {compliance_level}")
        
        # Run operation benchmarks
        for operation in self.config["operations"]:
            if hasattr(self, f"_benchmark_{operation}_compliance"):
                logger.info(f"  Running {operation} benchmark for {compliance_level}")
                
                # Run multiple times and take average
                times = []
                memory_usages = []
                
                for run in range(self.config["num_runs"]):
                    # Clear memory before each run
                    gc.collect()
                    
                    # Measure time and memory
                    start_time = time.time()
                    start_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
                    
                    # Run benchmark
                    benchmark_method = getattr(self, f"_benchmark_{operation}_compliance")
                    benchmark_method(instance, compliance_level)
                    
                    # Measure end time and memory
                    end_time = time.time()
                    end_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
                    
                    # Calculate metrics
                    elapsed_time = end_time - start_time
                    memory_usage = end_memory - start_memory
                    
                    times.append(elapsed_time)
                    memory_usages.append(memory_usage)
                
                # Calculate average metrics
                avg_time = sum(times) / len(times)
                avg_memory = sum(memory_usages) / len(memory_usages)
                
                # Store results
                results[operation] = {
                    "time": avg_time,
                    "memory": avg_memory,
                    "throughput": self.config["num_blocks"] / avg_time if avg_time > 0 else 0
                }
        
        # Clean up
        self._cleanup_compliance_instance(instance, compliance_level)
        
        return results
    
    def _benchmark_modality(self, modality: str) -> Dict[str, Any]:
        """
        Benchmark specific modality.
        
        Args:
            modality: Modality type (text, binary, embeddings, video, cross_modal)
            
        Returns:
            Benchmark results
        """
        results = {}
        
        # Create instance for modality benchmarks
        instance = MAIFEncoder()
        
        # Run operation benchmarks
        for operation in self.config["operations"]:
            if hasattr(self, f"_benchmark_{operation}_{modality}"):
                logger.info(f"  Running {operation} benchmark for {modality}")
                
                # Run multiple times and take average
                times = []
                memory_usages = []
                
                for run in range(self.config["num_runs"]):
                    # Clear memory before each run
                    gc.collect()
                    
                    # Measure time and memory
                    start_time = time.time()
                    start_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
                    
                    # Run benchmark
                    benchmark_method = getattr(self, f"_benchmark_{operation}_{modality}")
                    benchmark_method(instance)
                    
                    # Measure end time and memory
                    end_time = time.time()
                    end_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
                    
                    # Calculate metrics
                    elapsed_time = end_time - start_time
                    memory_usage = end_memory - start_memory
                    
                    times.append(elapsed_time)
                    memory_usages.append(memory_usage)
                
                # Calculate average metrics
                avg_time = sum(times) / len(times)
                avg_memory = sum(memory_usages) / len(memory_usages)
                
                # Store results
                results[operation] = {
                    "time": avg_time,
                    "memory": avg_memory,
                    "throughput": self.config["num_blocks"] / avg_time if avg_time > 0 else 0
                }
        
        # Clean up
        self._cleanup_modality_instance(instance, modality)
        
        return results
    
    def _create_instance(self, implementation: str) -> Any:
        """
        Create implementation instance.
        
        Args:
            implementation: Implementation name
            
        Returns:
            Implementation instance
        """
        if implementation == "core":
            return MAIFEncoder()
        
        elif implementation == "event_sourcing":
            event_log_path = self.temp_dir / "event_log.jsonl"
            event_log = EventLog(str(event_log_path))
            return EventSourcedMAIF(
                maif_id="benchmark",
                event_log=event_log,
                agent_id="benchmark-agent"
            )
        
        elif implementation == "columnar_storage":
            columnar_path = self.temp_dir / "columnar.parquet"
            return ColumnarFile(str(columnar_path))
        
        elif implementation == "version_management":
            # Create schema registry
            registry = SchemaRegistry()
            
            # Add schema
            schema = Schema(
                version="1.0.0",
                fields=[
                    SchemaField(name="id", field_type="string", required=True),
                    SchemaField(name="content", field_type="string", required=True),
                    SchemaField(name="metadata", field_type="json", required=False)
                ]
            )
            registry.register_schema(schema)
            
            return VersionManager(registry)
        
        elif implementation == "adaptation_rules":
            return AdaptationRulesEngine()
        
        elif implementation == "enhanced":
            maif_path = self.temp_dir / "enhanced.maif"
            return EnhancedMAIF(
                str(maif_path),
                agent_id="benchmark-agent",
                enable_event_sourcing=True,
                enable_columnar_storage=True,
                enable_version_management=True,
                enable_adaptation_rules=True
            )
        
        else:
            raise ValueError(f"Unknown implementation: {implementation}")
    
    def _cleanup_instance(self, instance: Any, implementation: str):
        """
        Clean up implementation instance.
        
        Args:
            instance: Implementation instance
            implementation: Implementation name
        """
        if implementation == "columnar_storage":
            instance.close()
        elif implementation == "enhanced":
            instance.save()
    
    def _cleanup_compliance_instance(self, instance: Any, compliance_level: str):
        """
        Clean up compliance instance.
        
        Args:
            instance: Compliance instance
            compliance_level: Compliance level
        """
        pass  # No special cleanup needed
    
    def _cleanup_modality_instance(self, instance: Any, modality: str):
        """
        Clean up modality instance.
        
        Args:
            instance: Modality instance
            modality: Modality type
        """
        pass  # No special cleanup needed
    
    # Implementation benchmark methods
    
    def _benchmark_write(self, instance: Any, implementation: str):
        """Benchmark write operation."""
        if implementation == "core":
            for i in range(self.config["num_blocks"]):
                text = f"Block {i}: " + "X" * (self.config["block_size"] - 10)
                instance.add_text_block(text, {"block_id": i})
        
        elif implementation == "event_sourcing":
            for i in range(self.config["num_blocks"]):
                text = f"Block {i}: " + "X" * (self.config["block_size"] - 10)
                instance.add_block(
                    block_id=f"block_{i}",
                    block_type="text",
                    data=text.encode('utf-8'),
                    metadata={"block_id": i}
                )
        
        elif implementation == "columnar_storage":
            data = {}
            for i in range(self.config["num_blocks"]):
                text = f"Block {i}: " + "X" * (self.config["block_size"] - 10)
                if "content" not in data:
                    data["content"] = []
                    data["block_id"] = []
                data["content"].append(text)
                data["block_id"].append(i)
            
            instance.write_batch(data)
        
        elif implementation == "version_management":
            # Not applicable for version management
            pass
        
        elif implementation == "adaptation_rules":
            # Not applicable for adaptation rules
            pass
        
        elif implementation == "enhanced":
            for i in range(self.config["num_blocks"]):
                text = f"Block {i}: " + "X" * (self.config["block_size"] - 10)
                instance.add_text_block(text, {"block_id": i})
    
    def _benchmark_read(self, instance: Any, implementation: str):
        """Benchmark read operation."""
        if implementation == "core":
            # Not applicable for encoder
            pass
        
        elif implementation == "event_sourcing":
            blocks = instance.get_blocks()
            for block_id, block in blocks.items():
                _ = block  # Access block data
        
        elif implementation == "columnar_storage":
            # Read all columns
            for column_name in instance.get_statistics().keys():
                _ = instance.read_column(column_name)
        
        elif implementation == "version_management":
            # Not applicable for version management
            pass
        
        elif implementation == "adaptation_rules":
            # Not applicable for adaptation rules
            pass
        
        elif implementation == "enhanced":
            # Get history
            _ = instance.get_history()
            
            # Get schema version
            _ = instance.get_schema_version()
            
            # Get columnar statistics
            _ = instance.get_columnar_statistics()
    
    def _benchmark_search(self, instance: Any, implementation: str):
        """Benchmark search operation."""
        if implementation == "core":
            # Not applicable for encoder
            pass
        
        elif implementation == "event_sourcing":
            # Search by timestamp
            _ = instance.get_events(
                start_time=time.time() - 3600,
                end_time=time.time()
            )
        
        elif implementation == "columnar_storage":
            # Not applicable for columnar storage
            pass
        
        elif implementation == "version_management":
            # Not applicable for version management
            pass
        
        elif implementation == "adaptation_rules":
            # Evaluate rules
            context = {
                "metrics": {
                    "size_bytes": 1024 * 1024,
                    "block_count": 100,
                    "access_frequency": 5.0,
                    "last_accessed": time.time(),
                    "compression_ratio": 2.0,
                    "fragmentation": 0.3,
                    "age_days": 10.0,
                    "semantic_coherence": 0.8
                },
                "current_time": time.time()
            }
            
            _ = instance.evaluate_rules(context)
        
        elif implementation == "enhanced":
            # Evaluate rules
            _ = instance.evaluate_rules()
    
    def _benchmark_update(self, instance: Any, implementation: str):
        """Benchmark update operation."""
        if implementation == "core":
            # Not applicable for encoder
            pass
        
        elif implementation == "event_sourcing":
            blocks = instance.get_blocks()
            for block_id in list(blocks.keys())[:10]:  # Update first 10 blocks
                instance.update_block(
                    block_id=block_id,
                    data=f"Updated block {block_id}".encode('utf-8')
                )
        
        elif implementation == "columnar_storage":
            # Not applicable for columnar storage
            pass
        
        elif implementation == "version_management":
            # Not applicable for version management
            pass
        
        elif implementation == "adaptation_rules":
            # Not applicable for adaptation rules
            pass
        
        elif implementation == "enhanced":
            # Not applicable for enhanced MAIF in this benchmark
            pass
    
    def _benchmark_delete(self, instance: Any, implementation: str):
        """Benchmark delete operation."""
        if implementation == "core":
            # Not applicable for encoder
            pass
        
        elif implementation == "event_sourcing":
            blocks = instance.get_blocks()
            for block_id in list(blocks.keys())[:10]:  # Delete first 10 blocks
                instance.delete_block(block_id)
        
        elif implementation == "columnar_storage":
            # Not applicable for columnar storage
            pass
        
        elif implementation == "version_management":
            # Not applicable for version management
            pass
        
        elif implementation == "adaptation_rules":
            # Not applicable for adaptation rules
            pass
        
        elif implementation == "enhanced":
            # Not applicable for enhanced MAIF in this benchmark
            pass
    
    # Compliance benchmark methods
    
    def _benchmark_write_compliance(self, instance: Any, compliance_level: str):
        """Benchmark write operation for compliance level."""
        for i in range(self.config["num_blocks"]):
            text = f"Block {i}: " + "X" * (self.config["block_size"] - 10)
            
            if compliance_level == "raw":
                instance.add_text_block(text, {"block_id": i})
            
            elif compliance_level == "acid":
                instance.begin_transaction()
                instance.add_text_block(text, {"block_id": i})
                instance.commit_transaction()
            
            elif compliance_level == "acid_security":
                instance.begin_transaction()
                instance.add_text_block(
                    text, 
                    {"block_id": i},
                    encryption_enabled=True,
                    access_control_enabled=True
                )
                instance.commit_transaction()
    
    def _benchmark_read_compliance(self, instance: Any, compliance_level: str):
        """Benchmark read operation for compliance level."""
        # Not applicable for encoder
        pass
    
    def _benchmark_search_compliance(self, instance: Any, compliance_level: str):
        """Benchmark search operation for compliance level."""
        # Not applicable for encoder
        pass
    
    def _benchmark_update_compliance(self, instance: Any, compliance_level: str):
        """Benchmark update operation for compliance level."""
        # Not applicable for encoder
        pass
    
    def _benchmark_delete_compliance(self, instance: Any, compliance_level: str):
        """Benchmark delete operation for compliance level."""
        # Not applicable for encoder
        pass
    
    # Modality benchmark methods
    
    def _benchmark_write_text(self, instance: Any):
        """Benchmark write operation for text modality."""
        for i in range(self.config["num_blocks"]):
            text = f"Block {i}: " + "X" * (self.config["block_size"] - 10)
            instance.add_text_block(text, {"block_id": i})
    
    def _benchmark_write_binary(self, instance: Any):
        """Benchmark write operation for binary modality."""
        for i in range(self.config["num_blocks"]):
            data = b"X" * self.config["block_size"]
            instance.add_binary_block(data, "binary", {"block_id": i})
    
    def _benchmark_write_embeddings(self, instance: Any):
        """Benchmark write operation for embeddings modality."""
        # Create random embeddings
        embeddings = []
        for i in range(self.config["num_blocks"]):
            embedding = [random.random() for _ in range(384)]  # 384-dim embeddings
            embeddings.append(embedding)
        
        instance.add_embeddings_block(embeddings, {"count": len(embeddings)})
    
    def _benchmark_write_video(self, instance: Any):
        """Benchmark write operation for video modality."""
        # Create dummy video data
        video_data = b"X" * (self.config["block_size"] * 100)  # Larger for video
        
        for i in range(10):  # Fewer blocks for video due to size
            instance.add_video_block(
                video_data,
                {"block_id": i, "duration": 10.0, "fps": 30.0}
            )
    
    def _benchmark_write_cross_modal(self, instance: Any):
        """Benchmark write operation for cross-modal modality."""
        # Create multimodal data
        for i in range(10):  # Fewer blocks for cross-modal due to complexity
            multimodal_data = {
                "text": f"Cross-modal block {i}",
                "image": b"X" * self.config["block_size"],
                "embedding": [random.random() for _ in range(384)]
            }
            
            instance.add_cross_modal_block(
                multimodal_data,
                {"block_id": i, "modalities": ["text", "image", "embedding"]}
            )
    
    def _benchmark_read_text(self, instance: Any):
        """Benchmark read operation for text modality."""
        # Not applicable for encoder
        pass
    
    def _benchmark_read_binary(self, instance: Any):
        """Benchmark read operation for binary modality."""
        # Not applicable for encoder
        pass
    
    def _benchmark_read_embeddings(self, instance: Any):
        """Benchmark read operation for embeddings modality."""
        # Not applicable for encoder
        pass
    
    def _benchmark_read_video(self, instance: Any):
        """Benchmark read operation for video modality."""
        # Not applicable for encoder
        pass
    
    def _benchmark_read_cross_modal(self, instance: Any):
        """Benchmark read operation for cross-modal modality."""
        # Not applicable for encoder
        pass
    
    # Results analysis and visualization
    
    def generate_report(self, output_dir: Optional[str] = None) -> str:
        """
        Generate benchmark report.
        
        Args:
            output_dir: Output directory for report files
            
        Returns:
            Path to report file
        """
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = self.temp_dir
        
        report_file = output_path / "benchmark_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# MAIF Performance Benchmark Report\n\n")
            
            # Write implementation results
            f.write("## Implementation Benchmarks\n\n")
            self._write_implementation_results(f)
            
            # Write compliance results
            f.write("## Compliance Level Benchmarks\n\n")
            self._write_compliance_results(f)
            
            # Write modality results
            f.write("## Modality Benchmarks\n\n")
            self._write_modality_results(f)
            
            # Write comparison charts
            f.write("## Performance Comparisons\n\n")
            self._generate_comparison_charts(output_path)
            f.write(f"![Implementation Comparison]({output_path}/implementation_comparison.png)\n\n")
            f.write(f"![Compliance Comparison]({output_path}/compliance_comparison.png)\n\n")
            f.write(f"![Modality Comparison]({output_path}/modality_comparison.png)\n\n")
        
        logger.info(f"Benchmark report generated at {report_file}")
        return str(report_file)
    
    def _write_implementation_results(self, file):
        """Write implementation benchmark results to file."""
        file.write("### Implementation Performance\n\n")
        
        # Create table headers
        headers = ["Implementation", "Operation", "Time (s)", "Memory (MB)", "Throughput (ops/s)"]
        rows = []
        
        # Add data rows
        for impl, results in self.results.items():
            if not impl.startswith("compliance_") and not impl.startswith("modality_"):
                for op, metrics in results.items():
                    rows.append([
                        impl,
                        op,
                        f"{metrics['time']:.4f}",
                        f"{metrics['memory']:.2f}",
                        f"{metrics['throughput']:.2f}"
                    ])
        
        # Write table
        file.write(tabulate(rows, headers=headers, tablefmt="pipe"))
        file.write("\n\n")
    
    def _write_compliance_results(self, file):
        """Write compliance benchmark results to file."""
        file.write("### Compliance Level Performance\n\n")
        
        # Create table headers
        headers = ["Compliance Level", "Operation", "Time (s)", "Memory (MB)", "Throughput (ops/s)"]
        rows = []
        
        # Add data rows
        for level_key, results in self.results.items():
            if level_key.startswith("compliance_"):
                level = level_key.replace("compliance_", "")
                for op, metrics in results.items():
                    rows.append([
                        level,
                        op,
                        f"{metrics['time']:.4f}",
                        f"{metrics['memory']:.2f}",
                        f"{metrics['throughput']:.2f}"
                    ])
        
        # Write table
        file.write(tabulate(rows, headers=headers, tablefmt="pipe"))
        file.write("\n\n")
    
    def _write_modality_results(self, file):
        """Write modality benchmark results to file."""
        file.write("### Modality Performance\n\n")
        
        # Create table headers
        headers = ["Modality", "Operation", "Time (s)", "Memory (MB)", "Throughput (ops/s)"]
        rows = []
        
        # Add data rows
        for modality_key, results in self.results.items():
            if modality_key.startswith("modality_"):
                modality = modality_key.replace("modality_", "")
                for op, metrics in results.items():
                    rows.append([
                        modality,
                        op,
                        f"{metrics['time']:.4f}",
                        f"{metrics['memory']:.2f}",
                        f"{metrics['throughput']:.2f}"
                    ])
        
        # Write table
        file.write(tabulate(rows, headers=headers, tablefmt="pipe"))
        file.write("\n\n")
    
    def _generate_comparison_charts(self, output_path: Path):
        """Generate comparison charts."""
        # Implementation comparison
        self._generate_implementation_chart(output_path / "implementation_comparison.png")
        
        # Compliance comparison
        self._generate_compliance_chart(output_path / "compliance_comparison.png")
        
        # Modality comparison
        self._generate_modality_chart(output_path / "modality_comparison.png")
    
    def _generate_implementation_chart(self, output_file: Path):
        """Generate implementation comparison chart."""
        plt.figure(figsize=(12, 8))
        
        # Extract data for chart
        implementations = []
        write_times = []
        read_times = []
        
        for impl, results in self.results.items():
            if not impl.startswith("compliance_") and not impl.startswith("modality_"):
                implementations.append(impl)
                
                write_time = results.get("write", {}).get("time", 0)
                write_times.append(write_time)
                
                read_time = results.get("read", {}).get("time", 0)
                read_times.append(read_time)
        
        # Create bar chart
        x = np.arange(len(implementations))
        width = 0.35
        
        plt.bar(x - width/2, write_times, width, label='Write Time (s)')
        plt.bar(x + width/2, read_times, width, label='Read Time (s)')
        
        plt.xlabel('Implementation')
        plt.ylabel('Time (s)')
        plt.title('Implementation Performance Comparison')
        plt.xticks(x, implementations, rotation=45)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(output_file)
        plt.close()
    
    def _generate_compliance_chart(self, output_file: Path):
        """Generate compliance comparison chart."""
        plt.figure(figsize=(10, 6))
        
        # Extract data for chart
        levels = []
        write_times = []
        
        for level_key, results in self.results.items():
            if level_key.startswith("compliance_"):
                level = level_key.replace("compliance_", "")
                levels.append(level)
                
                write_time = results.get("write", {}).get("time", 0)
                write_times.append(write_time)
        
        # Create bar chart
        plt.bar(levels, write_times)
        
        plt.xlabel('Compliance Level')
        plt.ylabel('Write Time (s)')
        plt.title('Compliance Level Performance Comparison')
        plt.tight_layout()
        
        plt.savefig(output_file)
        plt.close()
    
    def _generate_modality_chart(self, output_file: Path):
        """Generate modality comparison chart."""
        plt.figure(figsize=(12, 8))
        
        # Extract data for chart
        modalities = []
        write_times = []
        
        for modality_key, results in self.results.items():
            if modality_key.startswith("modality_"):
                modality = modality_key.replace("modality_", "")
                modalities.append(modality)
                
                write_time = results.get("write", {}).get("time", 0)
                write_times.append(write_time)
        
        # Create bar chart
        plt.bar(modalities, write_times)
        
        plt.xlabel('Modality')
        plt.ylabel('Write Time (s)')
        plt.title('Modality Performance Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(output_file)
        plt.close()


def main():
    """Run the benchmark."""
    parser = argparse.ArgumentParser(description="MAIF Performance Benchmark")
    parser.add_argument("--output", type=str, help="Output directory for report")
    parser.add_argument("--num-blocks", type=int, default=1000, help="Number of blocks to benchmark")
    parser.add_argument("--block-size", type=int, default=1024, help="Block size in bytes")
    parser.add_argument("--num-runs", type=int, default=3, help="Number of runs for each benchmark")
    parser.add_argument("--implementations", type=str, nargs="+", help="Implementations to benchmark")
    parser.add_argument("--compliance-levels", type=str, nargs="+", help="Compliance levels to benchmark")
    parser.add_argument("--modalities", type=str, nargs="+", help="Modalities to benchmark")
    parser.add_argument("--operations", type=str, nargs="+", help="Operations to benchmark")
    
    args = parser.parse_args()
    
    # Create config from args
    config = DEFAULT_CONFIG.copy()
    
    if args.num_blocks:
        config["num_blocks"] = args.num_blocks
    
    if args.block_size:
        config["block_size"] = args.block_size
    
    if args.num_runs:
        config["num_runs"] = args.num_runs
    
    if args.implementations:
        config["implementations"] = args.implementations
    
    if args.compliance_levels:
        config["compliance_levels"] = args.compliance_levels
    
    if args.modalities:
        config["modalities"] = args.modalities
    
    if args.operations:
        config["operations"] = args.operations
    
    # Run benchmark
    benchmark = PerformanceBenchmark(config)
    results = benchmark.run_benchmarks()
    
    # Generate report
    report_path = benchmark.generate_report(args.output)
    
    print(f"Benchmark report generated at {report_path}")


if __name__ == "__main__":
    main()