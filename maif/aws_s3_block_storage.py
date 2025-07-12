"""
AWS S3 Block Storage Integration for MAIF
=======================================

Extends MAIF's block storage system to use AWS S3 for scalable, durable storage.
"""

import io
import json
import time
import logging
import hashlib
import uuid
from typing import Dict, List, Optional, Union, BinaryIO, Any, Tuple, Set
from pathlib import Path
import threading
import os

from .block_storage import BlockStorage, BlockHeader
from .block_types import BlockType, BlockValidator, BlockFactory
from .signature_verification import (
    SignatureVerifier, create_default_verifier, sign_block_data,
    verify_block_signature, SignatureInfo
)
from .aws_s3_integration import S3Client

# Configure logger
logger = logging.getLogger(__name__)


class S3BlockStorageError(Exception):
    """Base exception for S3 block storage errors."""
    pass


class S3BlockStorage(BlockStorage):
    """MAIF block storage implementation using AWS S3."""
    
    def __init__(self, bucket_name: str, prefix: str = "blocks/", 
                region_name: str = "us-east-1", verify_signatures: bool = True,
                local_cache_dir: Optional[str] = None, max_cache_size_mb: int = 100):
        """
        Initialize S3 block storage.
        
        Args:
            bucket_name: S3 bucket name
            prefix: Prefix for S3 keys
            region_name: AWS region name
            verify_signatures: Whether to verify block signatures
            local_cache_dir: Directory for local block cache (optional)
            max_cache_size_mb: Maximum size of local cache in MB
        """
        super().__init__(file_path=None, verify_signatures=verify_signatures)
        
        self.bucket_name = bucket_name
        self.prefix = prefix.rstrip('/') + '/'
        self.region_name = region_name
        
        # Initialize S3 client
        self.s3_client = S3Client(region_name=region_name)
        
        # Local cache settings
        self.local_cache_dir = local_cache_dir
        if local_cache_dir:
            os.makedirs(local_cache_dir, exist_ok=True)
        self.max_cache_size_bytes = max_cache_size_mb * 1024 * 1024
        self.current_cache_size = 0
        self.cache_lock = threading.Lock()
        
        # Block index
        self.block_index = {}  # uuid -> metadata
        self.blocks = []  # (header, s3_key)
        
        # Load block index from S3
        self._load_block_index()
    
    def _load_block_index(self):
        """Load block index from S3."""
        try:
            # Check if index file exists
            index_key = f"{self.prefix}block_index.json"
            
            try:
                response = self.s3_client.get_object(
                    bucket_name=self.bucket_name,
                    key=index_key
                )
                
                if response and 'Body' in response:
                    index_data = json.loads(response['Body'].read().decode('utf-8'))
                    
                    # Restore block index
                    self.block_index = index_data.get('block_index', {})
                    
                    # Restore blocks list
                    self.blocks = []
                    for block_info in index_data.get('blocks', []):
                        if 'header' in block_info and 's3_key' in block_info:
                            header_dict = block_info['header']
                            header = BlockHeader(
                                size=header_dict['size'],
                                block_type=header_dict['block_type'],
                                version=header_dict['version'],
                                uuid=header_dict['uuid'],
                                timestamp=header_dict['timestamp'],
                                previous_hash=header_dict.get('previous_hash'),
                                flags=header_dict.get('flags', 0)
                            )
                            self.blocks.append((header, block_info['s3_key']))
                    
                    # Restore block signatures
                    self.block_signatures = index_data.get('block_signatures', {})
                    
                    logger.info(f"Loaded block index from S3: {len(self.blocks)} blocks")
                
            except Exception as e:
                logger.warning(f"Block index not found in S3 or error loading it: {e}")
                
                # If index not found, scan bucket for blocks
                self._scan_bucket_for_blocks()
        
        except Exception as e:
            logger.error(f"Error loading block index from S3: {e}")
            raise S3BlockStorageError(f"Failed to load block index: {e}")
    
    def _scan_bucket_for_blocks(self):
        """Scan S3 bucket for blocks and rebuild index."""
        try:
            logger.info(f"Scanning S3 bucket {self.bucket_name} for blocks...")
            
            # List objects with block prefix
            block_prefix = f"{self.prefix}data/"
            
            response = self.s3_client.list_objects(
                bucket_name=self.bucket_name,
                prefix=block_prefix
            )
            
            if 'Contents' in response:
                for obj in response['Contents']:
                    key = obj['Key']
                    
                    # Extract block UUID from key
                    block_uuid = key.split('/')[-1].split('.')[0]
                    
                    try:
                        # Get block metadata
                        obj_response = self.s3_client.head_object(
                            bucket_name=self.bucket_name,
                            key=key
                        )
                        
                        metadata = obj_response.get('Metadata', {})
                        
                        if 'block_type' in metadata and 'size' in metadata:
                            # Create header
                            header = BlockHeader(
                                size=int(metadata.get('size', 0)),
                                block_type=metadata.get('block_type', 'BDAT'),
                                version=int(metadata.get('version', 1)),
                                uuid=block_uuid,
                                timestamp=float(metadata.get('timestamp', time.time())),
                                previous_hash=metadata.get('previous_hash'),
                                flags=int(metadata.get('flags', 0))
                            )
                            
                            # Add to index
                            self.blocks.append((header, key))
                            self.block_index[block_uuid] = len(self.blocks) - 1
                            
                            # Check for signature
                            if 'signature' in metadata:
                                try:
                                    signature_data = json.loads(metadata['signature'])
                                    self.block_signatures[block_uuid] = signature_data
                                except:
                                    pass
                    
                    except Exception as e:
                        logger.warning(f"Error processing block {block_uuid}: {e}")
            
            logger.info(f"Rebuilt block index from S3 scan: {len(self.blocks)} blocks")
            
            # Save rebuilt index
            self._save_block_index()
            
        except Exception as e:
            logger.error(f"Error scanning S3 bucket for blocks: {e}")
    
    def _save_block_index(self):
        """Save block index to S3."""
        try:
            # Prepare index data
            index_data = {
                'block_index': self.block_index,
                'blocks': [
                    {
                        'header': {
                            'size': header.size,
                            'block_type': header.block_type,
                            'version': header.version,
                            'uuid': header.uuid,
                            'timestamp': header.timestamp,
                            'previous_hash': header.previous_hash,
                            'flags': header.flags
                        },
                        's3_key': s3_key
                    }
                    for header, s3_key in self.blocks
                ],
                'block_signatures': self.block_signatures,
                'updated_at': time.time()
            }
            
            # Save to S3
            index_key = f"{self.prefix}block_index.json"
            
            self.s3_client.put_object(
                bucket_name=self.bucket_name,
                key=index_key,
                data=json.dumps(index_data).encode('utf-8'),
                metadata={
                    'content_type': 'application/json',
                    'updated_at': str(time.time())
                }
            )
            
            logger.info(f"Saved block index to S3: {len(self.blocks)} blocks")
            
        except Exception as e:
            logger.error(f"Error saving block index to S3: {e}")
    
    def add_block(self, block_type: str, data: bytes, metadata: Optional[Dict] = None) -> str:
        """
        Add a new block to S3 storage with signature.
        
        Args:
            block_type: Block type (FourCC)
            data: Block data
            metadata: Block metadata
            
        Returns:
            Block UUID
        """
        block_uuid = str(uuid.uuid4())
        timestamp = time.time()
        
        # Ensure metadata is a dictionary
        if metadata is None:
            metadata = {}
        
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
        
        # Prepare S3 metadata
        s3_metadata = {
            'block_type': block_type,
            'size': str(len(data)),
            'version': str(header.version),
            'timestamp': str(timestamp),
            'content_type': 'application/octet-stream'
        }
        
        if previous_hash:
            s3_metadata['previous_hash'] = previous_hash
        
        if self.verify_signatures and block_uuid in self.block_signatures:
            s3_metadata['signature'] = json.dumps(self.block_signatures[block_uuid])
        
        # Add user metadata
        for key, value in metadata.items():
            if key != 'signature' and isinstance(value, (str, int, float, bool)):
                s3_metadata[f'user_{key}'] = str(value)
        
        # Store in S3
        s3_key = f"{self.prefix}data/{block_uuid}"
        
        try:
            self.s3_client.put_object(
                bucket_name=self.bucket_name,
                key=s3_key,
                data=data,
                metadata=s3_metadata
            )
            
            logger.info(f"Stored block {block_uuid} in S3")
            
            # Update local cache if enabled
            if self.local_cache_dir:
                self._cache_block(block_uuid, data)
            
            # Update index
            block_index = len(self.blocks)
            self.blocks.append((header, s3_key))
            self.block_index[block_uuid] = block_index
            
            # Save updated index
            self._save_block_index()
            
            return block_uuid
            
        except Exception as e:
            logger.error(f"Error storing block {block_uuid} in S3: {e}")
            raise S3BlockStorageError(f"Failed to store block: {e}")
    
    def get_block(self, block_uuid: str) -> Optional[Tuple[BlockHeader, bytes, Dict]]:
        """
        Retrieve a block by UUID with signature verification.
        
        Args:
            block_uuid: Block UUID
            
        Returns:
            Tuple of (header, data, metadata) or None if not found
        """
        if block_uuid not in self.block_index:
            logger.warning(f"Block {block_uuid} not found in index")
            return None
        
        # Check local cache first
        if self.local_cache_dir:
            cached_data = self._get_cached_block(block_uuid)
            if cached_data:
                logger.debug(f"Retrieved block {block_uuid} from local cache")
                block_index = self.block_index[block_uuid]
                header, _ = self.blocks[block_index]
                
                # Prepare metadata
                metadata = {}
                
                # Add signature info if available
                if self.verify_signatures and block_uuid in self.block_signatures:
                    metadata["signature"] = self.block_signatures[block_uuid]
                    
                    # Verify signature
                    is_valid = verify_block_signature(
                        self.signature_verifier,
                        cached_data,
                        metadata["signature"]
                    )
                    
                    metadata["signature_verified"] = is_valid
                    
                    if not is_valid:
                        logger.warning(f"Invalid signature for cached block {block_uuid}")
                
                return header, cached_data, metadata
        
        # Retrieve from S3
        try:
            block_index = self.block_index[block_uuid]
            header, s3_key = self.blocks[block_index]
            
            response = self.s3_client.get_object(
                bucket_name=self.bucket_name,
                key=s3_key
            )
            
            if not response or 'Body' not in response:
                logger.warning(f"Block {block_uuid} not found in S3")
                return None
            
            data = response['Body'].read()
            
            # Extract metadata from S3 object
            s3_metadata = response.get('Metadata', {})
            metadata = {}
            
            # Extract user metadata
            for key, value in s3_metadata.items():
                if key.startswith('user_'):
                    metadata[key[5:]] = value
            
            # Add signature info if available
            if self.verify_signatures:
                if 'signature' in s3_metadata:
                    try:
                        signature_data = json.loads(s3_metadata['signature'])
                        metadata["signature"] = signature_data
                        self.block_signatures[block_uuid] = signature_data
                        
                        # Verify signature
                        is_valid = verify_block_signature(
                            self.signature_verifier,
                            data,
                            signature_data
                        )
                        
                        metadata["signature_verified"] = is_valid
                        
                        if not is_valid:
                            logger.warning(f"Invalid signature for block {block_uuid}")
                    except Exception as e:
                        logger.warning(f"Error verifying signature for block {block_uuid}: {e}")
                        metadata["signature_verified"] = False
                else:
                    metadata["signature_verified"] = False
                    logger.warning(f"No signature found for block {block_uuid}")
            
            # Update local cache if enabled
            if self.local_cache_dir:
                self._cache_block(block_uuid, data)
            
            logger.info(f"Retrieved block {block_uuid} from S3")
            return header, data, metadata
            
        except Exception as e:
            logger.error(f"Error retrieving block {block_uuid} from S3: {e}")
            return None
    
    def list_blocks(self) -> List[BlockHeader]:
        """List all block headers."""
        return [header for header, _ in self.blocks]
    
    def _cache_block(self, block_uuid: str, data: bytes):
        """Cache block data locally."""
        if not self.local_cache_dir:
            return
        
        with self.cache_lock:
            # Check if we need to make room in the cache
            if self.current_cache_size + len(data) > self.max_cache_size_bytes:
                self._clean_cache(needed_bytes=len(data))
            
            # Write to cache
            cache_path = os.path.join(self.local_cache_dir, block_uuid)
            try:
                with open(cache_path, 'wb') as f:
                    f.write(data)
                
                self.current_cache_size += len(data)
                logger.debug(f"Cached block {block_uuid} locally ({len(data)} bytes)")
            except Exception as e:
                logger.warning(f"Error caching block {block_uuid}: {e}")
    
    def _get_cached_block(self, block_uuid: str) -> Optional[bytes]:
        """Get block data from local cache."""
        if not self.local_cache_dir:
            return None
        
        cache_path = os.path.join(self.local_cache_dir, block_uuid)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return f.read()
            except Exception as e:
                logger.warning(f"Error reading cached block {block_uuid}: {e}")
        
        return None
    
    def _clean_cache(self, needed_bytes: int):
        """Clean local cache to make room for new data."""
        if not self.local_cache_dir:
            return
        
        with self.cache_lock:
            # Get list of cached files with their access times
            cache_files = []
            for filename in os.listdir(self.local_cache_dir):
                file_path = os.path.join(self.local_cache_dir, filename)
                if os.path.isfile(file_path):
                    cache_files.append((
                        filename,
                        os.path.getsize(file_path),
                        os.path.getatime(file_path)
                    ))
            
            # Sort by access time (oldest first)
            cache_files.sort(key=lambda x: x[2])
            
            # Remove files until we have enough space
            bytes_freed = 0
            for filename, size, _ in cache_files:
                if self.current_cache_size - bytes_freed + needed_bytes <= self.max_cache_size_bytes:
                    break
                
                file_path = os.path.join(self.local_cache_dir, filename)
                try:
                    os.remove(file_path)
                    bytes_freed += size
                    logger.debug(f"Removed {filename} from cache ({size} bytes)")
                except Exception as e:
                    logger.warning(f"Error removing {filename} from cache: {e}")
            
            self.current_cache_size -= bytes_freed
            logger.info(f"Cleaned cache: freed {bytes_freed} bytes")
    
    def delete_block(self, block_uuid: str) -> bool:
        """
        Delete a block from S3 storage.
        
        Args:
            block_uuid: Block UUID
            
        Returns:
            True if deleted, False otherwise
        """
        if block_uuid not in self.block_index:
            logger.warning(f"Block {block_uuid} not found in index")
            return False
        
        try:
            block_index = self.block_index[block_uuid]
            _, s3_key = self.blocks[block_index]
            
            # Delete from S3
            self.s3_client.delete_object(
                bucket_name=self.bucket_name,
                key=s3_key
            )
            
            # Remove from index
            del self.block_index[block_uuid]
            self.blocks.pop(block_index)
            
            # Update indices
            for uuid, idx in self.block_index.items():
                if idx > block_index:
                    self.block_index[uuid] = idx - 1
            
            # Remove from signatures
            if block_uuid in self.block_signatures:
                del self.block_signatures[block_uuid]
            
            # Remove from local cache
            if self.local_cache_dir:
                cache_path = os.path.join(self.local_cache_dir, block_uuid)
                if os.path.exists(cache_path):
                    try:
                        size = os.path.getsize(cache_path)
                        os.remove(cache_path)
                        self.current_cache_size -= size
                    except Exception as e:
                        logger.warning(f"Error removing {block_uuid} from cache: {e}")
            
            # Save updated index
            self._save_block_index()
            
            logger.info(f"Deleted block {block_uuid} from S3")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting block {block_uuid} from S3: {e}")
            return False
    
    def validate_integrity(self) -> bool:
        """Validate block chain integrity."""
        for i, (header, _) in enumerate(self.blocks):
            if i == 0:
                continue  # First block has no previous
            
            prev_header = self.blocks[i-1][0]
            expected_hash = self._calculate_block_hash(prev_header, b"")  # Simplified
            
            if header.previous_hash != expected_hash:
                logger.warning(f"Integrity validation failed at block {header.uuid}")
                return False
        
        logger.info("Block chain integrity validated successfully")
        return True
    
    def validate_all_signatures(self) -> Dict[str, Any]:
        """Validate signatures for all blocks."""
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
        
        for header, s3_key in self.blocks:
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
            
            # Get block data
            try:
                response = self.s3_client.get_object(
                    bucket_name=self.bucket_name,
                    key=s3_key
                )
                
                if response and 'Body' in response:
                    data = response['Body'].read()
                    
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
                else:
                    results["blocks_with_issues"].append({
                        "block_uuid": block_uuid,
                        "issue": "block_not_found"
                    })
            except Exception as e:
                logger.warning(f"Error verifying signature for block {block_uuid}: {e}")
                results["blocks_with_issues"].append({
                    "block_uuid": block_uuid,
                    "issue": f"verification_error: {str(e)}"
                })
        
        return results
    
    def create_bucket_if_not_exists(self):
        """Create S3 bucket if it doesn't exist."""
        try:
            # Check if bucket exists
            buckets = self.s3_client.list_buckets()
            
            if self.bucket_name not in [b['Name'] for b in buckets.get('Buckets', [])]:
                # Create bucket
                self.s3_client.create_bucket(
                    bucket_name=self.bucket_name,
                    region=self.region_name
                )
                
                logger.info(f"Created S3 bucket: {self.bucket_name}")
            else:
                logger.info(f"S3 bucket already exists: {self.bucket_name}")
                
        except Exception as e:
            logger.error(f"Error creating S3 bucket {self.bucket_name}: {e}")
            raise S3BlockStorageError(f"Failed to create bucket: {e}")


class S3StreamingBlockParser:
    """Optimized block parser for streaming operations from S3."""
    
    def __init__(self, s3_client: S3Client, bucket_name: str, prefix: str = "blocks/data/",
                chunk_size: int = 64 * 1024):  # 64KB chunks
        """
        Initialize S3 streaming block parser.
        
        Args:
            s3_client: S3 client
            bucket_name: S3 bucket name
            prefix: Prefix for S3 keys
            chunk_size: Chunk size for streaming
        """
        self.s3_client = s3_client
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.chunk_size = chunk_size
        self.buffer = bytearray()
        self.parsed_blocks = 0
    
    def parse_object(self, key: str) -> List[BlockHeader]:
        """
        Parse blocks from S3 object with high performance.
        
        Args:
            key: S3 object key
            
        Returns:
            List of block headers
        """
        headers = []
        
        try:
            # Get object
            response = self.s3_client.get_object(
                bucket_name=self.bucket_name,
                key=key
            )
            
            if not response or 'Body' not in response:
                logger.warning(f"Object {key} not found in S3")
                return headers
            
            # Stream object body
            body = response['Body']
            
            while True:
                # Read chunk
                chunk = body.read(self.chunk_size)
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
                            
                    except Exception:
                        break  # Invalid header, need more data
            
            return headers
            
        except Exception as e:
            logger.error(f"Error parsing S3 object {key}: {e}")
            return headers
    
    def parse_prefix(self, max_objects: int = 100) -> List[BlockHeader]:
        """
        Parse blocks from all objects with prefix.
        
        Args:
            max_objects: Maximum number of objects to parse
            
        Returns:
            List of block headers
        """
        headers = []
        
        try:
            # List objects with prefix
            response = self.s3_client.list_objects(
                bucket_name=self.bucket_name,
                prefix=self.prefix
            )
            
            if 'Contents' in response:
                objects = response['Contents'][:max_objects]
                
                for obj in objects:
                    key = obj['Key']
                    obj_headers = self.parse_object(key)
                    headers.extend(obj_headers)
            
            return headers
            
        except Exception as e:
            logger.error(f"Error parsing S3 prefix {self.prefix}: {e}")
            return headers
    
    def get_stats(self) -> Dict[str, Any]:
        """Get parsing statistics."""
        return {
            "parsed_blocks": self.parsed_blocks,
            "buffer_size": len(self.buffer),
            "chunk_size": self.chunk_size
        }


# Export classes
__all__ = ['S3BlockStorage', 'S3StreamingBlockParser', 'S3BlockStorageError']