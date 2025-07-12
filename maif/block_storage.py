"""
MAIF Block Storage System
Implements hierarchical block structure with efficient parsing and validation.
"""

import struct
import hashlib
import uuid
import time
import logging
import threading
import os
from typing import Dict, List, Optional, Union, BinaryIO, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import io
from .signature_verification import (
    SignatureVerifier, create_default_verifier, sign_block_data,
    verify_block_signature, SignatureInfo
)

logger = logging.getLogger(__name__)

class BlockType(Enum):
    """MAIF Block Types as defined in the paper."""
    HEADER = "HDER"
    TEXT_DATA = "TEXT"
    EMBEDDING = "EMBD"
    KNOWLEDGE_GRAPH = "KGRF"
    SECURITY = "SECU"
    BINARY_DATA = "BDAT"
    IMAGE_DATA = "IMAG"
    AUDIO_DATA = "AUDI"
    VIDEO_DATA = "VIDE"
    AI_MODEL = "AIMD"
    PROVENANCE = "PROV"
    ACCESS_CONTROL = "ACCS"
    LIFECYCLE = "LIFE"
    CROSS_MODAL = "XMOD"
    SEMANTIC_BINDING = "SBND"
    COMPRESSED_EMBEDDINGS = "CEMB"

@dataclass
class BlockHeader:
    """MAIF Block Header Structure."""
    size: int
    block_type: str  # FourCC identifier
    version: int
    uuid: str
    timestamp: float
    previous_hash: Optional[str] = None
    flags: int = 0
    
    def to_bytes(self) -> bytes:
        """Serialize header to bytes."""
        uuid_bytes = uuid.UUID(self.uuid).bytes
        type_bytes = self.block_type.encode('ascii')[:4].ljust(4, b'\x00')
        prev_hash_bytes = (self.previous_hash or '').encode('utf-8')[:32].ljust(32, b'\x00')
        
        return struct.pack(
            '<I4sIQ16s32sI',
            self.size,
            type_bytes,
            self.version,
            int(self.timestamp * 1000000),  # microseconds
            uuid_bytes,
            prev_hash_bytes,
            self.flags
        )
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'BlockHeader':
        """Deserialize header from bytes."""
        size, type_bytes, version, timestamp_us, uuid_bytes, prev_hash_bytes, flags = struct.unpack(
            '<I4sIQ16s32sI', data[:72]
        )
        
        block_type = type_bytes.decode('ascii').rstrip('\x00')
        uuid_str = str(uuid.UUID(bytes=uuid_bytes))
        timestamp = timestamp_us / 1000000.0
        prev_hash = prev_hash_bytes.decode('utf-8').rstrip('\x00') or None
        
        return cls(
            size=size,
            block_type=block_type,
            version=version,
            uuid=uuid_str,
            timestamp=timestamp,
            previous_hash=prev_hash,
            flags=flags
        )

class BlockStorage:
    """High-performance block storage with memory-mapped access."""
    
    HEADER_SIZE = 72  # Fixed header size
    
    def __init__(self, file_path: Optional[str] = None, verify_signatures: bool = True):
        self.file_path = file_path
        self.blocks: List[Tuple[BlockHeader, int]] = []  # (header, data_offset)
        self.block_index: Dict[str, int] = {}  # uuid -> block_index
        self.file_handle: Optional[BinaryIO] = None
        self.memory_mapped = False
        self._lock = threading.RLock()  # Thread safety
        self._closed = False
        self._in_memory_data: Dict[str, bytes] = {}  # For in-memory storage
        
        # Signature verification
        self.verify_signatures = verify_signatures
        self.signature_verifier = create_default_verifier() if verify_signatures else None
        self.block_signatures: Dict[str, Dict[str, Any]] = {}  # block_uuid -> signature_metadata
        
    def __enter__(self):
        with self._lock:
            if self.file_path:
                try:
                    # Try to open existing file for read/write
                    if os.path.exists(self.file_path):
                        self.file_handle = open(self.file_path, 'rb+')
                    else:
                        # Create new file
                        self.file_handle = open(self.file_path, 'wb+')
                except Exception as e:
                    raise RuntimeError(f"Failed to open block storage file: {e}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def close(self):
        """Close the block storage file."""
        with self._lock:
            self._closed = True
            if self.file_handle and not self.file_handle.closed:
                try:
                    self.file_handle.flush()
                    self.file_handle.close()
                except:
                    pass
                self.file_handle = None
    
    def add_block(self, block_type: str, data: bytes, metadata: Optional[Dict] = None) -> str:
        """Add a new block to storage with signature."""
        # Validate inputs
        if not isinstance(data, bytes):
            raise TypeError("Block data must be bytes")
        if not isinstance(block_type, str) or len(block_type) > 4:
            raise ValueError("Block type must be a string of max 4 characters")
            
        with self._lock:
            if self._closed:
                raise RuntimeError("BlockStorage is closed")
                
            block_uuid = str(uuid.uuid4())
            timestamp = time.time()
            
            # Ensure metadata is a dictionary
            if metadata is None:
                metadata = {}
            
            # Calculate previous hash for chain integrity
            previous_hash = None
            if self.blocks:
                last_header = self.blocks[-1][0]
                # Get the actual data for proper hash calculation
                if self.file_handle:
                    last_data_offset = self.blocks[-1][1]
                    self.file_handle.seek(last_data_offset + self.HEADER_SIZE)
                    last_data = self.file_handle.read(last_header.size)
                else:
                    last_data = self._in_memory_data.get(last_header.uuid, b"")
                previous_hash = self._calculate_block_hash(last_header, last_data)
            
            # Create header
            header = BlockHeader(
                size=len(data),
                block_type=block_type,
                version=1,
                uuid=block_uuid,
                timestamp=timestamp,
                previous_hash=previous_hash
            )
            
            # Sign the block if signature verification is enabled
            if self.verify_signatures and self.signature_verifier:
                try:
                    # Sign the block data
                    signature_metadata = sign_block_data(
                        self.signature_verifier,
                        data,
                        key_id="default"  # Use default key for now
                    )
                    
                    # Store signature metadata
                    self.block_signatures[block_uuid] = signature_metadata
                    
                    # Add signature info to block metadata
                    metadata["signature"] = signature_metadata
                except Exception as e:
                    logger.warning(f"Failed to sign block {block_uuid}: {e}")
            
            # Store block
            if self.file_handle:
                # Write to file
                data_offset = self.file_handle.tell()
                self.file_handle.write(header.to_bytes())
                self.file_handle.write(data)
                self.file_handle.flush()
            else:
                # In-memory storage
                data_offset = len(self.blocks) * (self.HEADER_SIZE + 1000)  # Simulated offset
                self._in_memory_data[block_uuid] = data
            
            # Update index
            block_index = len(self.blocks)
            self.blocks.append((header, data_offset))
            self.block_index[block_uuid] = block_index
            
            return block_uuid
    
    def get_block(self, block_uuid: str) -> Optional[Tuple[BlockHeader, bytes]]:
        """Retrieve a block by UUID with signature verification."""
        with self._lock:
            if self._closed:
                raise RuntimeError("BlockStorage is closed")
                
            if block_uuid not in self.block_index:
                return None
            
            block_index = self.block_index[block_uuid]
            header, data_offset = self.blocks[block_index]
            
            if self.file_handle:
                # Read from file
                self.file_handle.seek(data_offset + self.HEADER_SIZE)
                data = self.file_handle.read(header.size)
            else:
                # Get from in-memory storage
                data = self._in_memory_data.get(block_uuid, b"")
            
            # Verify signature if enabled
            if self.verify_signatures and self.signature_verifier:
                signature_metadata = self.block_signatures.get(block_uuid)
                
                if signature_metadata:
                    # Verify signature
                    is_valid = verify_block_signature(
                        self.signature_verifier,
                        data,
                        signature_metadata
                    )
                    
                    # Log warning if signature is invalid
                    if not is_valid:
                        logger.warning(f"Invalid signature for block {block_uuid}")
                else:
                    logger.warning(f"No signature found for block {block_uuid}")
            
            return header, data
    
    def get_block_with_metadata(self, block_uuid: str) -> Optional[Tuple[BlockHeader, bytes, Dict]]:
        """Retrieve a block by UUID with signature verification and metadata."""
        result = self.get_block(block_uuid)
        if result is None:
            return None
            
        header, data = result
        
        # Prepare metadata
        metadata = {}
        
        # Add signature info if available
        if self.verify_signatures and block_uuid in self.block_signatures:
            signature_metadata = self.block_signatures[block_uuid]
            metadata["signature"] = signature_metadata
            
            # Verify signature
            if self.signature_verifier:
                is_valid = verify_block_signature(
                    self.signature_verifier,
                    data,
                    signature_metadata
                )
                metadata["signature_verified"] = is_valid
        
        return header, data, metadata
    
    def list_blocks(self) -> List[BlockHeader]:
        """List all block headers."""
        return [header for header, _ in self.blocks]
    
    def _calculate_block_hash(self, header: BlockHeader, data: bytes) -> str:
        """Calculate SHA-256 hash of block."""
        hasher = hashlib.sha256()
        hasher.update(header.to_bytes())
        hasher.update(data)
        return hasher.hexdigest()
    
    def validate_integrity(self) -> bool:
        """Validate block chain integrity."""
        with self._lock:
            for i, (header, data_offset) in enumerate(self.blocks):
                if i == 0:
                    continue  # First block has no previous
                
                prev_header = self.blocks[i-1][0]
                
                # Get the actual data for proper hash calculation
                if self.file_handle:
                    prev_data_offset = self.blocks[i-1][1]
                    self.file_handle.seek(prev_data_offset + self.HEADER_SIZE)
                    prev_data = self.file_handle.read(prev_header.size)
                else:
                    prev_data = self._in_memory_data.get(prev_header.uuid, b"")
                
                expected_hash = self._calculate_block_hash(prev_header, prev_data)
                
                if header.previous_hash != expected_hash:
                    logger.error(f"Chain integrity broken at block {i}: expected {expected_hash}, got {header.previous_hash}")
                    return False
            
            return True
    
    def validate_all_signatures(self) -> Dict[str, Any]:
        """Validate signatures for all blocks."""
        with self._lock:
            if not self.verify_signatures or not self.signature_verifier:
                return {"enabled": False, "message": "Signature verification not enabled"}
            
            results = {
                "total_blocks": len(self.blocks),
                "signed_blocks": 0,
                "valid_signatures": 0,
                "invalid_signatures": 0,
                "missing_signatures": 0,
                "blocks_with_issues": []
            }
            
            for header, data_offset in self.blocks:
                block_uuid = header.uuid
                
                # Check if block has signature
                if block_uuid not in self.block_signatures:
                    results["missing_signatures"] += 1
                    results["blocks_with_issues"].append({
                        "block_uuid": block_uuid,
                        "issue": "missing_signature"
                    })
                    continue
                
                results["signed_blocks"] += 1
                
                # Read block data
                try:
                    if self.file_handle:
                        self.file_handle.seek(data_offset + self.HEADER_SIZE)
                        data = self.file_handle.read(header.size)
                    else:
                        data = self._in_memory_data.get(block_uuid, b"")
                        if not data:
                            results["blocks_with_issues"].append({
                                "block_uuid": block_uuid,
                                "issue": "data_not_found"
                            })
                            continue
                    
                    # Verify signature
                    signature_metadata = self.block_signatures[block_uuid]
                    is_valid = verify_block_signature(
                        self.signature_verifier,
                        data,
                        signature_metadata
                    )
                    
                    if is_valid:
                        results["valid_signatures"] += 1
                    else:
                        results["invalid_signatures"] += 1
                        results["blocks_with_issues"].append({
                            "block_uuid": block_uuid,
                            "issue": "invalid_signature"
                        })
                except Exception as e:
                    logger.error(f"Error validating signature for block {block_uuid}: {e}")
                    results["blocks_with_issues"].append({
                        "block_uuid": block_uuid,
                        "issue": f"validation_error: {str(e)}"
                    })
            
            return results

class HighPerformanceBlockParser:
    """Optimized block parser for streaming operations."""
    
    def __init__(self, chunk_size: int = 64 * 1024):  # 64KB chunks
        self.chunk_size = max(1024, chunk_size)  # Minimum 1KB chunks
        self.buffer = bytearray()
        self.parsed_blocks = 0
        self._max_buffer_size = 10 * 1024 * 1024  # 10MB max buffer to prevent OOM
        
    def parse_stream(self, stream: BinaryIO) -> List[BlockHeader]:
        """Parse blocks from stream with high performance."""
        headers = []
        
        while True:
            # Check buffer size to prevent unbounded growth
            if len(self.buffer) > self._max_buffer_size:
                logger.warning(f"Parser buffer exceeded max size ({self._max_buffer_size} bytes)")
                # Try to recover by skipping to next valid header
                self._recover_from_corrupt_data()
                if len(self.buffer) > self._max_buffer_size:
                    raise RuntimeError("Unable to recover from corrupt stream data")
            
            # Read chunk
            try:
                chunk = stream.read(self.chunk_size)
                if not chunk:
                    break
            except Exception as e:
                logger.error(f"Error reading from stream: {e}")
                break
            
            self.buffer.extend(chunk)
            
            # Parse complete blocks from buffer
            while len(self.buffer) >= BlockStorage.HEADER_SIZE:
                try:
                    header = BlockHeader.from_bytes(self.buffer[:BlockStorage.HEADER_SIZE])
                    
                    # Validate header size is reasonable
                    if header.size < 0 or header.size > 100 * 1024 * 1024:  # Max 100MB per block
                        logger.warning(f"Invalid block size: {header.size}")
                        self._skip_invalid_header()
                        continue
                    
                    # Check if we have complete block
                    total_block_size = BlockStorage.HEADER_SIZE + header.size
                    if len(self.buffer) >= total_block_size:
                        headers.append(header)
                        # Remove processed block from buffer
                        self.buffer = self.buffer[total_block_size:]
                        self.parsed_blocks += 1
                    else:
                        break  # Need more data
                        
                except (struct.error, Exception) as e:
                    logger.debug(f"Error parsing header: {e}")
                    # Try to recover by finding next valid header
                    if not self._skip_invalid_header():
                        break  # No more valid headers found
        
        return headers
    
    def _skip_invalid_header(self) -> bool:
        """Skip invalid header and find next valid one."""
        # Skip one byte at a time looking for valid header
        self.buffer = self.buffer[1:]
        return len(self.buffer) >= BlockStorage.HEADER_SIZE
    
    def _recover_from_corrupt_data(self):
        """Try to recover from corrupt data by finding next valid block."""
        # Look for MAIF block signatures or patterns
        # This is a simplified recovery - in production would be more sophisticated
        while len(self.buffer) > BlockStorage.HEADER_SIZE:
            try:
                # Try to parse header
                BlockHeader.from_bytes(self.buffer[:BlockStorage.HEADER_SIZE])
                # If successful, we found a valid header
                return
            except:
                # Skip one byte and try again
                self.buffer = self.buffer[1:]
        
        # If we get here, clear the buffer as we couldn't find valid data
        self.buffer.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get parsing statistics."""
        return {
            "parsed_blocks": self.parsed_blocks,
            "buffer_size": len(self.buffer),
            "chunk_size": self.chunk_size
        }
    
    def reset(self):
        """Reset parser state."""
        self.buffer.clear()
        self.parsed_blocks = 0
    
    def __del__(self):
        """Clean up resources."""
        # Clear potentially large buffer
        self.buffer.clear()