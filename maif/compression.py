"""
Compression functionality for MAIF files.
"""

import zlib
import gzip
import bz2
from enum import Enum
from typing import Optional, Dict, Any
import struct
from dataclasses import dataclass

@dataclass
class CompressionMetadata:
    """Metadata for compressed data blocks."""
    algorithm: str
    original_size: int
    compressed_size: int
    compression_ratio: float
    compression_time: float = 0.0
    decompression_time: float = 0.0

try:
    import lzma
    LZMA_AVAILABLE = True
except ImportError:
    LZMA_AVAILABLE = False

try:
    import brotli
    BROTLI_AVAILABLE = True
except ImportError:
    BROTLI_AVAILABLE = False

try:
    import lz4.frame
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False

class CompressionAlgorithm(Enum):
    """Supported compression algorithms."""
    NONE = "none"
    ZLIB = "zlib"
    GZIP = "gzip"
    BZIP2 = "bzip2"
    LZMA = "lzma"
    BROTLI = "brotli"
    LZ4 = "lz4"
    ZSTD = "zstandard"
    HSC = "hierarchical_semantic"

class MAIFCompressor:
    """Handles compression and decompression for MAIF data."""
    
    def __init__(self):
        self.algorithms = {
            CompressionAlgorithm.NONE: (self._compress_none, self._decompress_none),
            CompressionAlgorithm.ZLIB: (self._compress_zlib, self._decompress_zlib),
            CompressionAlgorithm.GZIP: (self._compress_gzip, self._decompress_gzip),
            CompressionAlgorithm.BZIP2: (self._compress_bzip2, self._decompress_bzip2),
        }
        
        # Add optional algorithms if available
        if LZMA_AVAILABLE:
            self.algorithms[CompressionAlgorithm.LZMA] = (self._compress_lzma, self._decompress_lzma)
        
        if BROTLI_AVAILABLE:
            self.algorithms[CompressionAlgorithm.BROTLI] = (self._compress_brotli, self._decompress_brotli)
        
        if LZ4_AVAILABLE:
            self.algorithms[CompressionAlgorithm.LZ4] = (self._compress_lz4, self._decompress_lz4)
        
        if ZSTD_AVAILABLE:
            self.algorithms[CompressionAlgorithm.ZSTD] = (self._compress_zstd, self._decompress_zstd)
        
        # HSC algorithm available but not auto-registered due to complexity
        # Use compress_semantic() method for HSC compression
    
    def compress(self, data: bytes, algorithm: CompressionAlgorithm, level: Optional[int] = None) -> bytes:
        """Compress data using the specified algorithm."""
        if algorithm not in self.algorithms:
            raise ValueError(f"Compression algorithm {algorithm} not available")
        
        compress_func, _ = self.algorithms[algorithm]
        return compress_func(data, level)
    
    def decompress(self, data: bytes, algorithm: CompressionAlgorithm) -> bytes:
        """Decompress data using the specified algorithm."""
        if algorithm not in self.algorithms:
            raise ValueError(f"Compression algorithm {algorithm} not available")
        
        _, decompress_func = self.algorithms[algorithm]
        return decompress_func(data)
    
    def get_compression_ratio(self, original: bytes, compressed: bytes) -> float:
        """Calculate compression ratio."""
        if len(compressed) == 0:
            return 0.0
        return len(original) / len(compressed)
    
    def get_available_algorithms(self) -> list[CompressionAlgorithm]:
        """Get list of available compression algorithms."""
        return list(self.algorithms.keys())
    
    def benchmark_algorithms(self, data: bytes) -> Dict[str, Dict[str, Any]]:
        """Benchmark all available compression algorithms on given data."""
        import time
        
        results = {}
        original_size = len(data)
        
        for algorithm in self.algorithms:
            try:
                # Compression benchmark
                start_time = time.time()
                compressed = self.compress(data, algorithm)
                compress_time = time.time() - start_time
                
                # Decompression benchmark
                start_time = time.time()
                decompressed = self.decompress(compressed, algorithm)
                decompress_time = time.time() - start_time
                
                # Verify correctness
                is_correct = decompressed == data
                
                results[algorithm.value] = {
                    "original_size": original_size,
                    "compressed_size": len(compressed),
                    "compression_ratio": self.get_compression_ratio(data, compressed),
                    "reduction_percent": (1 - len(compressed)/original_size) * 100,
                    "compress_time": compress_time,
                    "decompress_time": decompress_time,
                    "compress_speed_mbps": (original_size / (1024*1024)) / compress_time if compress_time > 0 else 0,
                    "decompress_speed_mbps": (original_size / (1024*1024)) / decompress_time if decompress_time > 0 else 0,
                    "is_correct": is_correct
                }
                
            except Exception as e:
                results[algorithm.value] = {
                    "error": str(e),
                    "available": False
                }
        
        return results
    
    # Compression implementations
    def _compress_none(self, data: bytes, level: Optional[int] = None) -> bytes:
        """No compression."""
        return data
    
    def _decompress_none(self, data: bytes) -> bytes:
        """No decompression."""
        return data
    
    def _compress_zlib(self, data: bytes, level: Optional[int] = None) -> bytes:
        """Compress using zlib."""
        level = level or 6
        return zlib.compress(data, level)
    
    def _decompress_zlib(self, data: bytes) -> bytes:
        """Decompress using zlib."""
        return zlib.decompress(data)
    
    def _compress_gzip(self, data: bytes, level: Optional[int] = None) -> bytes:
        """Compress using gzip."""
        level = level or 6
        return gzip.compress(data, compresslevel=level)
    
    def _decompress_gzip(self, data: bytes) -> bytes:
        """Decompress using gzip."""
        return gzip.decompress(data)
    
    def _compress_bzip2(self, data: bytes, level: Optional[int] = None) -> bytes:
        """Compress using bzip2."""
        level = level or 9
        return bz2.compress(data, compresslevel=level)
    
    def _decompress_bzip2(self, data: bytes) -> bytes:
        """Decompress using bzip2."""
        return bz2.decompress(data)
    
    def _compress_lzma(self, data: bytes, level: Optional[int] = None) -> bytes:
        """Compress using LZMA."""
        level = level or 6
        return lzma.compress(data, preset=level)
    
    def _decompress_lzma(self, data: bytes) -> bytes:
        """Decompress using LZMA."""
        return lzma.decompress(data)
    
    def _compress_brotli(self, data: bytes, level: Optional[int] = None) -> bytes:
        """Compress using Brotli."""
        level = level or 6
        return brotli.compress(data, quality=level)
    
    def _decompress_brotli(self, data: bytes) -> bytes:
        """Decompress using Brotli."""
        return brotli.decompress(data)
    
    def _compress_lz4(self, data: bytes, level: Optional[int] = None) -> bytes:
        """Compress using LZ4."""
        return lz4.frame.compress(data)
    
    def _decompress_lz4(self, data: bytes) -> bytes:
        """Decompress using LZ4."""
        return lz4.frame.decompress(data)
    
    def _compress_zstd(self, data: bytes, level: Optional[int] = None) -> bytes:
        """Compress using Zstandard."""
        level = level or 3
        cctx = zstd.ZstdCompressor(level=level)
        return cctx.compress(data)
    
    def _decompress_zstd(self, data: bytes) -> bytes:
        """Decompress using Zstandard."""
        dctx = zstd.ZstdDecompressor()
        return dctx.decompress(data)


class SemanticAwareCompressor(MAIFCompressor):
    """Compression that preserves semantic relationships."""
    
    def __init__(self):
        super().__init__()
        self.semantic_preserving_algorithms = [
            CompressionAlgorithm.ZLIB,
            CompressionAlgorithm.BROTLI,
            CompressionAlgorithm.LZMA
        ]
    
    def compress_with_semantic_preservation(self, data: bytes, data_type: str, 
                                          algorithm: CompressionAlgorithm) -> bytes:
        """Compress data while attempting to preserve semantic structure."""
        
        # For text data, try to preserve word boundaries
        if data_type == "text" and algorithm in self.semantic_preserving_algorithms:
            return self._compress_text_semantic(data, algorithm)
        
        # For embeddings, use specialized compression
        elif data_type == "embeddings":
            return self._compress_embeddings_semantic(data, algorithm)
        
        # Default compression for other types
        else:
            return self.compress(data, algorithm)
    
    def _compress_text_semantic(self, data: bytes, algorithm: CompressionAlgorithm) -> bytes:
        """Compress text data with semantic awareness."""
        try:
            # Decode text and normalize whitespace for better compression
            text = data.decode('utf-8')
            
            # Normalize whitespace but preserve structure
            import re
            normalized = re.sub(r'\s+', ' ', text)
            normalized = normalized.strip()
            
            # Compress the normalized text
            normalized_bytes = normalized.encode('utf-8')
            return self.compress(normalized_bytes, algorithm)
            
        except UnicodeDecodeError:
            # Fallback to binary compression
            return self.compress(data, algorithm)
    
    def _compress_embeddings_semantic(self, data: bytes, algorithm: CompressionAlgorithm) -> bytes:
        """Compress embedding data with semantic preservation."""
        try:
            # Embeddings are typically float arrays
            # We can quantize them slightly for better compression while preserving semantics
            
            # Parse as float32 array
            import struct
            float_count = len(data) // 4
            floats = struct.unpack(f'{float_count}f', data)
            
            # Quantize to reduce precision slightly (preserves semantic meaning)
            quantized = [round(f, 6) for f in floats]  # 6 decimal places
            
            # Repack and compress
            quantized_bytes = struct.pack(f'{len(quantized)}f', *quantized)
            return self.compress(quantized_bytes, algorithm)
            
        except (struct.error, ValueError):
            # Fallback to regular compression
            return self.compress(data, algorithm)
    
    def _compress_hsc(self, data: bytes, level: Optional[int] = None) -> bytes:
        """Compress using Hierarchical Semantic Compression (HSC)."""
        try:
            # Import HSC from semantic module
            from .semantic import HierarchicalSemanticCompression
            import json
            import struct
            
            # Try to parse as embeddings first
            try:
                float_count = len(data) // 4
                if float_count > 0:
                    floats = struct.unpack(f'{float_count}f', data)
                    embeddings = [list(floats)]
                    
                    hsc = HierarchicalSemanticCompression()
                    compressed_result = hsc.compress_embeddings(embeddings)
                    
                    # Serialize the result
                    serialized = json.dumps(compressed_result).encode('utf-8')
                    
                    # Apply additional compression
                    return self.compress(serialized, CompressionAlgorithm.ZLIB)
            except (struct.error, ValueError):
                pass
            
            # Try to parse as text
            try:
                text = data.decode('utf-8')
                # For text, apply semantic-aware text compression
                return self._compress_text_semantic(data, CompressionAlgorithm.BROTLI)
            except UnicodeDecodeError:
                pass
            
            # Fallback to regular compression
            return self.compress(data, CompressionAlgorithm.ZLIB)
            
        except ImportError:
            # Fallback if semantic module not available
            return self.compress(data, CompressionAlgorithm.ZLIB)
    
    def _decompress_hsc(self, data: bytes) -> bytes:
        """Decompress using Hierarchical Semantic Compression (HSC)."""
        try:
            # Import HSC from semantic module
            from .semantic import HierarchicalSemanticCompression
            import json
            import struct
            
            # First decompress the outer layer
            decompressed = self.decompress(data, CompressionAlgorithm.ZLIB)
            
            try:
                # Try to parse as HSC-compressed embeddings
                compressed_result = json.loads(decompressed.decode('utf-8'))
                
                if isinstance(compressed_result, dict) and 'compressed_data' in compressed_result:
                    hsc = HierarchicalSemanticCompression()
                    embeddings = hsc.decompress_embeddings(compressed_result)
                    
                    # Convert back to bytes
                    if embeddings and len(embeddings) > 0:
                        floats = embeddings[0]  # Take first embedding
                        return struct.pack(f'{len(floats)}f', *floats)
            except (json.JSONDecodeError, KeyError, ValueError):
                pass
            
            # If not HSC format, return the decompressed data as-is
            return decompressed
            
        except ImportError:
            # Fallback if semantic module not available
            return self.decompress(data, CompressionAlgorithm.ZLIB)