"""
MAIF SDK Client - High-performance native interface for MAIF operations.

This client provides the "hot path" for latency-sensitive operations with direct
memory-mapped I/O and optimized block handling as recommended in the decision memo.
"""

import os
import mmap
import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Union, BinaryIO, Any, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, asdict

from ..maif.core import MAIFFile, MAIFBlock
from ..maif.security import SecurityEngine
from ..maif.compression import CompressionEngine
from ..maif.privacy import PrivacyEngine, PrivacyPolicy
from .types import ContentType, SecurityLevel, CompressionLevel, ContentMetadata, SecurityOptions, ProcessingOptions
from .artifact import Artifact


@dataclass
class ClientConfig:
    """Configuration for MAIF client."""
    agent_id: str = "default_agent"
    enable_mmap: bool = True
    buffer_size: int = 64 * 1024  # 64KB buffer for write combining
    max_concurrent_writers: int = 4
    enable_compression: bool = True
    default_security_level: SecurityLevel = SecurityLevel.PUBLIC
    cache_embeddings: bool = True
    validate_blocks: bool = True


class WriteBuffer:
    """Write-combining buffer to coalesce multiple writes into single MAIF blocks."""
    
    def __init__(self, max_size: int = 64 * 1024):
        self.max_size = max_size
        self.buffer = []
        self.current_size = 0
        self.lock = threading.Lock()
    
    def add(self, data: bytes, metadata: Optional[Dict] = None) -> bool:
        """Add data to buffer. Returns True if buffer should be flushed."""
        with self.lock:
            entry_size = len(data)
            if self.current_size + entry_size > self.max_size and self.buffer:
                return True  # Signal flush needed
            
            self.buffer.append({
                'data': data,
                'metadata': metadata or {},
                'timestamp': time.time()
            })
            self.current_size += entry_size
            return False
    
    def flush(self) -> List[Dict]:
        """Flush buffer and return all entries."""
        with self.lock:
            entries = self.buffer.copy()
            self.buffer.clear()
            self.current_size = 0
            return entries


class MAIFClient:
    """
    High-performance MAIF client with direct memory-mapped I/O.
    
    This is the recommended "hot path" for latency-sensitive operations,
    providing native SDK performance without FUSE overhead.
    """
    
    def __init__(self, agent_id: str = "default_agent", **kwargs):
        self.config = ClientConfig(agent_id=agent_id, **kwargs)
        self.security_engine = SecurityEngine()
        self.compression_engine = CompressionEngine()
        self.privacy_engine = PrivacyEngine()
        self.write_buffer = WriteBuffer(self.config.buffer_size)
        self._open_files: Dict[str, MAIFFile] = {}
        self._mmaps: Dict[str, mmap.mmap] = {}
        self._lock = threading.RLock()
    
    def create_artifact(self, name: str, **kwargs) -> Artifact:
        """Create a new artifact with this client's configuration."""
        return Artifact(
            name=name,
            client=self,
            security_level=kwargs.get('security_level', self.config.default_security_level),
            **kwargs
        )
    
    @contextmanager
    def open_file(self, filepath: Union[str, Path], mode: str = 'r'):
        """
        Open a MAIF file with memory mapping for high performance.
        
        Args:
            filepath: Path to MAIF file
            mode: File mode ('r', 'w', 'a')
            
        Yields:
            MAIFFile instance with optional memory mapping
        """
        filepath = str(filepath)
        
        try:
            with self._lock:
                if filepath in self._open_files:
                    yield self._open_files[filepath]
                    return
                
                maif_file = MAIFFile(filepath)
                
                if mode in ('r', 'a') and os.path.exists(filepath):
                    maif_file.load()
                    
                    # Enable memory mapping for read operations if configured
                    if self.config.enable_mmap and mode == 'r':
                        try:
                            with open(filepath, 'rb') as f:
                                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                                self._mmaps[filepath] = mm
                                maif_file._mmap = mm
                        except (OSError, ValueError):
                            # Fallback to regular file I/O
                            pass
                
                self._open_files[filepath] = maif_file
                
            yield maif_file
            
        finally:
            # Keep file open for potential reuse, cleanup happens in close()
            pass
    
    def write_content(self, filepath: Union[str, Path], content: bytes, 
                     content_type: ContentType = ContentType.DATA,
                     metadata: Optional[ContentMetadata] = None,
                     security_options: Optional[SecurityOptions] = None,
                     processing_options: Optional[ProcessingOptions] = None,
                     flush_immediately: bool = False) -> str:
        """
        Write content to MAIF file with write buffering for performance.
        
        Args:
            filepath: Target MAIF file path
            content: Raw content bytes
            content_type: Type of content
            metadata: Content metadata
            security_options: Security configuration
            processing_options: Processing options
            flush_immediately: Skip buffering and write immediately
            
        Returns:
            Block ID of written content
        """
        filepath = str(filepath)
        
        # Prepare metadata
        meta_dict = {}
        if metadata:
            meta_dict.update(asdict(metadata))
        meta_dict['content_type'] = content_type.value
        meta_dict['agent_id'] = self.config.agent_id
        meta_dict['timestamp'] = time.time()
        
        # Apply security options
        if security_options:
            if security_options.encrypt:
                content = self.security_engine.encrypt_data(content)
                meta_dict['encrypted'] = True
            
            if security_options.sign:
                signature = self.security_engine.sign_data(content)
                meta_dict['signature'] = signature
        
        # Apply processing options
        if processing_options:
            if processing_options.compression != CompressionLevel.NONE:
                content = self.compression_engine.compress(
                    content, 
                    level=processing_options.compression.value
                )
                meta_dict['compressed'] = True
                meta_dict['compression_level'] = processing_options.compression.value
        
        # Use write buffering unless immediate flush requested
        if not flush_immediately:
            should_flush = self.write_buffer.add(content, meta_dict)
            if should_flush:
                self._flush_buffer_to_file(filepath)
        
        # Write immediately
        with self.open_file(filepath, 'a') as maif_file:
            block = MAIFBlock(
                block_type=content_type.value,
                data=content,
                metadata=meta_dict
            )
            maif_file.add_block(block)
            maif_file.save()
            return block.block_id
    
    def _flush_buffer_to_file(self, filepath: str):
        """Flush write buffer to file as a single operation."""
        entries = self.write_buffer.flush()
        if not entries:
            return
        
        with self.open_file(filepath, 'a') as maif_file:
            for entry in entries:
                block = MAIFBlock(
                    block_type=entry['metadata'].get('content_type', 'data'),
                    data=entry['data'],
                    metadata=entry['metadata']
                )
                maif_file.add_block(block)
            maif_file.save()
    
    def read_content(self, filepath: Union[str, Path], 
                    block_id: Optional[str] = None,
                    content_type: Optional[ContentType] = None) -> Iterator[Dict]:
        """
        Read content from MAIF file with memory-mapped access for performance.
        
        Args:
            filepath: MAIF file path
            block_id: Specific block ID to read (optional)
            content_type: Filter by content type (optional)
            
        Yields:
            Dictionary with block data and metadata
        """
        with self.open_file(filepath, 'r') as maif_file:
            for block in maif_file.blocks:
                # Apply filters
                if block_id and block.block_id != block_id:
                    continue
                if content_type and block.block_type != content_type.value:
                    continue
                
                # Decrypt if needed
                data = block.data
                if block.metadata and block.metadata.get('encrypted'):
                    data = self.security_engine.decrypt_data(data)
                
                # Decompress if needed
                if block.metadata and block.metadata.get('compressed'):
                    data = self.compression_engine.decompress(data)
                
                yield {
                    'block_id': block.block_id,
                    'content_type': block.block_type,
                    'data': data,
                    'metadata': block.metadata or {},
                    'size': len(data),
                    'hash': block.hash_value
                }
    
    def get_file_info(self, filepath: Union[str, Path]) -> Dict:
        """Get information about a MAIF file."""
        with self.open_file(filepath, 'r') as maif_file:
            return {
                'filepath': str(filepath),
                'total_blocks': len(maif_file.blocks),
                'file_size': os.path.getsize(filepath) if os.path.exists(filepath) else 0,
                'content_types': list(set(block.block_type for block in maif_file.blocks)),
                'agents': list(set(
                    block.metadata.get('agent_id', 'unknown') 
                    for block in maif_file.blocks 
                    if block.metadata
                )),
                'created': min(
                    (block.metadata.get('timestamp', 0) for block in maif_file.blocks if block.metadata),
                    default=0
                ),
                'modified': max(
                    (block.metadata.get('timestamp', 0) for block in maif_file.blocks if block.metadata),
                    default=0
                )
            }
    
    def flush_all_buffers(self):
        """Flush all pending write buffers."""
        # In a real implementation, we'd track which files have pending buffers
        # For now, this is a placeholder for the interface
        pass
    
    def close(self):
        """Close all open files and clean up resources."""
        with self._lock:
            # Close memory maps
            for mm in self._mmaps.values():
                try:
                    mm.close()
                except:
                    pass
            self._mmaps.clear()
            
            # Close MAIF files
            for maif_file in self._open_files.values():
                try:
                    if hasattr(maif_file, 'close'):
                        maif_file.close()
                except:
                    pass
            self._open_files.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Convenience functions for quick operations
def quick_write(filepath: Union[str, Path], content: bytes, 
               content_type: ContentType = ContentType.DATA, **kwargs) -> str:
    """Quick write operation using default client."""
    with MAIFClient() as client:
        return client.write_content(filepath, content, content_type, **kwargs)


def quick_read(filepath: Union[str, Path], **kwargs) -> List[Dict]:
    """Quick read operation using default client."""
    with MAIFClient() as client:
        return list(client.read_content(filepath, **kwargs))