"""
Streaming and high-performance I/O functionality for MAIF files.
"""

import os
import mmap
import threading
from typing import Iterator, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .core import MAIFDecoder, MAIFBlock

@dataclass
class StreamingConfig:
    """Configuration for streaming operations."""
    chunk_size: int = 4096  # 4KB default
    max_workers: int = 4
    buffer_size: int = 65536  # 64KB default
    use_memory_mapping: bool = True
    prefetch_blocks: int = 10
    enable_compression: bool = False
    compression_level: int = 6

class MAIFStreamReader:
    """High-performance streaming reader for MAIF files."""
    
    def __init__(self, maif_path: str, config: Optional[StreamingConfig] = None):
        self.maif_path = Path(maif_path)
        self.config = config or StreamingConfig()
        self.file_handle = None
        self.memory_map = None
        self.decoder = None
        self._blocks = None
        self._total_blocks_read = 0
        self._total_bytes_read = 0
        self._read_times = []
    
    @property
    def blocks(self):
        """Get the blocks from the decoder."""
        if not self.decoder:
            manifest_path = str(self.maif_path).replace('.maif', '_manifest.json')
            if os.path.exists(manifest_path):
                self.decoder = MAIFDecoder(str(self.maif_path), manifest_path)
            else:
                return []
        return self.decoder.blocks
        
    def __enter__(self):
        """Context manager entry."""
        self.file_handle = open(self.maif_path, 'rb')
        
        if self.config.use_memory_mapping:
            try:
                self.memory_map = mmap.mmap(
                    self.file_handle.fileno(), 
                    0, 
                    access=mmap.ACCESS_READ
                )
            except (OSError, ValueError):
                # Fallback to regular file I/O
                self.memory_map = None
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.memory_map:
            self.memory_map.close()
            self.memory_map = None
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
    
    def stream_blocks(self) -> Iterator[Tuple[str, bytes]]:
        """Stream blocks sequentially."""
        import time
        
        if not self.decoder:
            # Need manifest to decode - this is a simplified version
            manifest_path = str(self.maif_path).replace('.maif', '_manifest.json')
            if os.path.exists(manifest_path):
                self.decoder = MAIFDecoder(str(self.maif_path), manifest_path)
            else:
                raise ValueError("Manifest file not found for streaming")
        
        for block in self.decoder.blocks:
            start_time = time.time()
            data = self._read_block_data(block)
            read_time = time.time() - start_time
            
            self._total_blocks_read += 1
            self._total_bytes_read += len(data)
            self._read_times.append(read_time)
            
            yield block.block_type or "unknown", data
    
    def stream_blocks_parallel(self) -> Iterator[Tuple[str, bytes]]:
        """Stream blocks in parallel using multiple workers."""
        if not self.decoder:
            manifest_path = str(self.maif_path).replace('.maif', '_manifest.json')
            if os.path.exists(manifest_path):
                self.decoder = MAIFDecoder(str(self.maif_path), manifest_path)
            else:
                raise ValueError("Manifest file not found for streaming")
        
        # Process blocks in parallel
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all block reading tasks
            future_to_block = {
                executor.submit(self._read_block_data_safe, block): block 
                for block in self.decoder.blocks
            }
            
            # Yield results as they complete
            for future in as_completed(future_to_block):
                block = future_to_block[future]
                try:
                    data = future.result()
                    yield block.block_id or f"block_{block.offset}", data
                except Exception as e:
                    print(f"Error reading block {block.block_id}: {e}")
                    continue
    
    def _read_block_data(self, block: MAIFBlock) -> bytes:
        """Read data for a specific block."""
        header_size = 24 if hasattr(block, 'version') and block.version else 16
        data_size = block.size - header_size
        
        if self.memory_map:
            # Use memory mapping for faster access
            start_pos = block.offset + header_size
            return self.memory_map[start_pos:start_pos + data_size]
        else:
            # Use regular file I/O
            self.file_handle.seek(block.offset + header_size)
            return self.file_handle.read(data_size)
    
    def _read_block_data_safe(self, block: MAIFBlock) -> bytes:
        """Thread-safe version of block data reading."""
        # Each thread needs its own file handle for parallel access
        with open(self.maif_path, 'rb') as f:
            header_size = 24 if hasattr(block, 'version') and block.version else 16
            data_size = block.size - header_size
            f.seek(block.offset + header_size)
            return f.read(data_size)
    
    def get_block_by_id(self, block_id: str) -> Optional[bytes]:
        """Get a specific block by ID."""
        if not self.decoder:
            manifest_path = str(self.maif_path).replace('.maif', '_manifest.json')
            if os.path.exists(manifest_path):
                self.decoder = MAIFDecoder(str(self.maif_path), manifest_path)
            else:
                return None
        
        for block in self.decoder.blocks:
            if block.block_id == block_id:
                return self._read_block_data(block)
        
        return None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the current session."""
        file_size = os.path.getsize(self.maif_path)
        avg_read_time = sum(self._read_times) / len(self._read_times) if self._read_times else 0
        
        return {
            "total_blocks_read": self._total_blocks_read,
            "total_bytes_read": self._total_bytes_read,
            "average_read_time": avg_read_time,
            "file_size": file_size,
            "memory_mapped": self.memory_map is not None,
            "chunk_size": self.config.chunk_size,
            "max_workers": self.config.max_workers,
            "buffer_size": self.config.buffer_size
        }


class MAIFStreamWriter:
    """High-performance streaming writer for MAIF files."""
    
    def __init__(self, output_path: str, config: Optional[StreamingConfig] = None):
        self.output_path = Path(output_path)
        self.config = config or StreamingConfig()
        self.file_handle = None
        self.buffer = b""
        self.blocks_written = 0
        
    def __enter__(self):
        """Context manager entry."""
        self.file_handle = open(self.output_path, 'wb')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self._flush_buffer()
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
    
    def write_block_stream(self, block_type: str, data_stream: Iterator[bytes]) -> str:
        """Write a block from a data stream."""
        import hashlib
        import struct
        
        # Collect all data and calculate hash
        all_data = b""
        hasher = hashlib.sha256()
        
        for chunk in data_stream:
            all_data += chunk
            hasher.update(chunk)
        
        hash_value = hasher.hexdigest()
        
        # Write block header
        size = len(all_data)
        header = struct.pack('>I4sIII', size, block_type.encode('ascii')[:4].ljust(4, b'\0'),
                           1, 0, 0)  # version, flags, reserved
        
        self._write_to_buffer(header)
        self._write_to_buffer(all_data)
        
        self.blocks_written += 1
        return hash_value
    
    def _write_to_buffer(self, data: bytes):
        """Write data to internal buffer with automatic flushing."""
        self.buffer += data
        
        if len(self.buffer) >= self.config.buffer_size:
            self._flush_buffer()
    
    def _flush_buffer(self):
        """Flush internal buffer to file."""
        if self.buffer and self.file_handle:
            self.file_handle.write(self.buffer)
            self.file_handle.flush()
            self.buffer = b""


class PerformanceProfiler:
    """Profiles MAIF streaming performance."""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
        self.timings = {}
    
    def start_timing(self, operation: str):
        """Start timing an operation."""
        import time
        self.start_times[operation] = time.time()
    
    def end_timing(self, operation: str, bytes_processed: int = 0):
        """End timing an operation."""
        import time
        if operation in self.start_times:
            duration = time.time() - self.start_times[operation]
            
            timing_data = {
                "duration": duration,
                "bytes_processed": bytes_processed,
                "throughput_mbps": (bytes_processed / (1024 * 1024)) / duration if duration > 0 else 0
            }
            
            self.metrics[operation] = timing_data
            self.timings[operation] = timing_data
            
            del self.start_times[operation]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        return self.metrics.copy()
    
    def print_report(self):
        """Print a performance report."""
        print("\nMAIF Streaming Performance Report")
        print("=" * 40)
        
        for operation, metrics in self.metrics.items():
            print(f"\n{operation}:")
            print(f"  Duration: {metrics['duration']:.3f}s")
            print(f"  Bytes: {metrics['bytes_processed']:,}")
            print(f"  Throughput: {metrics['throughput_mbps']:.1f} MB/s")