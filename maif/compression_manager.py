"""
Compression Manager for MAIF
============================

Provides a unified interface for compression operations across the MAIF system.
This module serves as a facade for the underlying compression implementations,
making it easier to use compression functionality throughout the codebase.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
import os
import json
from enum import Enum
from dataclasses import dataclass

from .compression import (
    MAIFCompressor, 
    CompressionAlgorithm, 
    CompressionConfig, 
    CompressionResult
)

class CompressionManager:
    """
    Manages compression operations across the MAIF system.
    Provides a unified interface for compression and decompression.
    """
    
    def __init__(self, config: Optional[CompressionConfig] = None):
        """Initialize the compression manager with optional configuration."""
        self.compressor = MAIFCompressor(config)
        self.config = config or CompressionConfig()
        self.compression_stats = {}
        
    def compress(self, data: bytes, algorithm: Union[str, CompressionAlgorithm] = None, 
                level: int = None) -> bytes:
        """
        Compress data using the specified algorithm and level.
        
        Args:
            data: The data to compress
            algorithm: The compression algorithm to use (defaults to config)
            level: The compression level to use (defaults to config)
            
        Returns:
            Compressed bytes
        """
        if algorithm is None:
            algorithm = self.config.algorithm
            
        return self.compressor.compress(data, algorithm, level)
    
    def decompress(self, data: bytes, algorithm: Union[str, CompressionAlgorithm]) -> bytes:
        """
        Decompress data using the specified algorithm.
        
        Args:
            data: The compressed data
            algorithm: The algorithm used for compression
            
        Returns:
            Decompressed bytes
        """
        return self.compressor.decompress(data, algorithm)
    
    def compress_with_metadata(self, data: bytes, data_type: str = "binary",
                              semantic_context: Optional[Dict] = None) -> CompressionResult:
        """
        Compress data with full metadata and semantic awareness.
        
        Args:
            data: The data to compress
            data_type: Type of data (binary, text, json, embeddings, etc.)
            semantic_context: Optional context for semantic compression
            
        Returns:
            CompressionResult object with compressed data and metadata
        """
        return self.compressor.compress_data(data, data_type, semantic_context)
    
    def decompress_with_metadata(self, result: CompressionResult) -> bytes:
        """
        Decompress data using metadata from CompressionResult.
        
        Args:
            result: CompressionResult from previous compression
            
        Returns:
            Decompressed bytes
        """
        return self.compressor.decompress_data(result.compressed_data, result.metadata)
    
    def get_compression_ratio(self, original: bytes, compressed: bytes) -> float:
        """Calculate compression ratio between original and compressed data."""
        return self.compressor.get_compression_ratio(original, compressed)
    
    def benchmark_algorithms(self, data: bytes, data_type: str = "binary") -> Dict[str, Any]:
        """
        Benchmark all available compression algorithms on the given data.
        
        Args:
            data: Sample data to benchmark
            data_type: Type of data for semantic algorithms
            
        Returns:
            Dictionary of algorithm names to benchmark results
        """
        return self.compressor.benchmark_algorithms(data)
    
    def get_optimal_algorithm(self, data: bytes, data_type: str = "binary") -> CompressionAlgorithm:
        """
        Determine the optimal compression algorithm for the given data.
        
        Args:
            data: Sample data to analyze
            data_type: Type of data (binary, text, json, embeddings, etc.)
            
        Returns:
            The recommended CompressionAlgorithm
        """
        return self.compressor._select_optimal_algorithm(data, data_type, None)
    
    def get_supported_algorithms(self) -> List[str]:
        """Get list of supported compression algorithms."""
        return [algo.value for algo in self.compressor.supported_algorithms]