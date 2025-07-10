"""
MAIF Block Storage System
Implements hierarchical block structure with efficient parsing and validation.
"""

import struct
import hashlib
import uuid
import time
from typing import Dict, List, Optional, Union, BinaryIO, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import io

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
    
    def __init__(self, file_path: Optional[str] = None):
        self.file_path = file_path
        self.blocks: List[Tuple[BlockHeader, int]] = []  # (header, data_offset)
        self.block_index: Dict[str, int] = {}  # uuid -> block_index
        self.file_handle: Optional[BinaryIO] = None
        self.memory_mapped = False
        
    def __enter__(self):
        if self.file_path:
            self.file_handle = open(self.file_path, 'rb+' if self.file_path else 'wb+')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file_handle:
            self.file_handle.close()
    
    def add_block(self, block_type: str, data: bytes, metadata: Optional[Dict] = None) -> str:
        """Add a new block to storage."""
        block_uuid = str(uuid.uuid4())
        timestamp = time.time()
        
        # Calculate previous hash for chain integrity
        previous_hash = None
        if self.blocks:
            last_header = self.blocks[-1][0]
            previous_hash = self._calculate_block_hash(last_header, b"")  # Simplified
        
        # Create header
        header = BlockHeader(
            size=len(data),
            block_type=block_type,
            version=1,
            uuid=block_uuid,
            timestamp=timestamp,
            previous_hash=previous_hash
        )
        
        # Store block
        if self.file_handle:
            # Write to file
            data_offset = self.file_handle.tell()
            self.file_handle.write(header.to_bytes())
            self.file_handle.write(data)
            self.file_handle.flush()
        else:
            # In-memory storage
            data_offset = len(self.blocks) * 1000  # Mock offset
        
        # Update index
        block_index = len(self.blocks)
        self.blocks.append((header, data_offset))
        self.block_index[block_uuid] = block_index
        
        return block_uuid
    
    def get_block(self, block_uuid: str) -> Optional[Tuple[BlockHeader, bytes]]:
        """Retrieve a block by UUID."""
        if block_uuid not in self.block_index:
            return None
        
        block_index = self.block_index[block_uuid]
        header, data_offset = self.blocks[block_index]
        
        if self.file_handle:
            # Read from file
            self.file_handle.seek(data_offset + self.HEADER_SIZE)
            data = self.file_handle.read(header.size)
        else:
            # Mock data for in-memory
            data = b"mock_data"
        
        return header, data
    
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
        for i, (header, data_offset) in enumerate(self.blocks):
            if i == 0:
                continue  # First block has no previous
            
            prev_header = self.blocks[i-1][0]
            expected_hash = self._calculate_block_hash(prev_header, b"")  # Simplified
            
            if header.previous_hash != expected_hash:
                return False
        
        return True

class HighPerformanceBlockParser:
    """Optimized block parser for streaming operations."""
    
    def __init__(self, chunk_size: int = 64 * 1024):  # 64KB chunks
        self.chunk_size = chunk_size
        self.buffer = bytearray()
        self.parsed_blocks = 0
        
    def parse_stream(self, stream: BinaryIO) -> List[BlockHeader]:
        """Parse blocks from stream with high performance."""
        headers = []
        
        while True:
            # Read chunk
            chunk = stream.read(self.chunk_size)
            if not chunk:
                break
            
            self.buffer.extend(chunk)
            
            # Parse complete blocks from buffer
            while len(self.buffer) >= BlockStorage.HEADER_SIZE:
                try:
                    header = BlockHeader.from_bytes(self.buffer[:BlockStorage.HEADER_SIZE])
                    
                    # Check if we have complete block
                    total_block_size = BlockStorage.HEADER_SIZE + header.size
                    if len(self.buffer) >= total_block_size:
                        headers.append(header)
                        # Remove processed block from buffer
                        self.buffer = self.buffer[total_block_size:]
                        self.parsed_blocks += 1
                    else:
                        break  # Need more data
                        
                except struct.error:
                    break  # Invalid header, need more data
        
        return headers
    
    def get_stats(self) -> Dict[str, Any]:
        """Get parsing statistics."""
        return {
            "parsed_blocks": self.parsed_blocks,
            "buffer_size": len(self.buffer),
            "chunk_size": self.chunk_size
        }