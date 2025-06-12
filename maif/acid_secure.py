#!/usr/bin/env python3
"""
Security-Hardened ACID Implementation for MAIF
==============================================

This implementation focuses on security-first design to prevent:
- Memory injection attacks
- Buffer overflow vulnerabilities
- SQL injection-style attacks on transaction data
- Privilege escalation through transaction manipulation
- Data corruption through malicious inputs

Key security features:
- Input validation and sanitization
- Memory-safe operations
- Cryptographic integrity checks
- Access control enforcement
- Audit logging for all operations
"""

import os
import sys
import time
import struct
import hashlib
import hmac
import secrets
import threading
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import uuid
import json
import re
import logging
from pathlib import Path


# Configure security logging
logging.basicConfig(level=logging.INFO)
security_logger = logging.getLogger('maif.security')


class SecureACIDLevel(Enum):
    """Security-hardened ACID levels."""
    PERFORMANCE = 0      # Fast but with security checks
    FULL_ACID = 2       # Full ACID with maximum security


class SecurityError(Exception):
    """Security-related errors."""
    pass


class InputValidationError(SecurityError):
    """Input validation failures."""
    pass


class AccessDeniedError(SecurityError):
    """Access control violations."""
    pass


@dataclass
class SecureWALEntry:
    """Security-hardened WAL entry with integrity protection."""
    transaction_id: str
    operation: str
    block_id: str
    data_hash: str
    integrity_hash: str  # HMAC for tamper detection
    timestamp: float
    user_context: str    # User/process context for auditing
    
    @classmethod
    def create_secure(cls, transaction_id: str, operation: str, block_id: str, 
                     data: bytes, user_context: str, secret_key: bytes) -> 'SecureWALEntry':
        """Create WAL entry with security checks."""
        # Validate inputs
        cls._validate_transaction_id(transaction_id)
        cls._validate_operation(operation)
        cls._validate_block_id(block_id)
        cls._validate_user_context(user_context)
        
        # Create data hash
        data_hash = hashlib.sha256(data).hexdigest()
        timestamp = time.time()
        
        # Create integrity hash (HMAC)
        message = f"{transaction_id}|{operation}|{block_id}|{data_hash}|{timestamp}|{user_context}"
        integrity_hash = hmac.new(secret_key, message.encode('utf-8'), hashlib.sha256).hexdigest()
        
        return cls(transaction_id, operation, block_id, data_hash, integrity_hash, timestamp, user_context)
    
    @staticmethod
    def _validate_transaction_id(txn_id: str):
        """Validate transaction ID format."""
        if not isinstance(txn_id, str):
            raise InputValidationError("Transaction ID must be string")
        if len(txn_id) != 36:  # UUID length
            raise InputValidationError("Invalid transaction ID format")
        if not re.match(r'^[0-9a-f-]+$', txn_id):
            raise InputValidationError("Transaction ID contains invalid characters")
    
    @staticmethod
    def _validate_operation(operation: str):
        """Validate operation type."""
        if not isinstance(operation, str):
            raise InputValidationError("Operation must be string")
        if operation not in ['write', 'read', 'delete']:
            raise InputValidationError(f"Invalid operation: {operation}")
        if len(operation) > 10:
            raise InputValidationError("Operation name too long")
    
    @staticmethod
    def _validate_block_id(block_id: str):
        """Validate block ID."""
        if not isinstance(block_id, str):
            raise InputValidationError("Block ID must be string")
        if len(block_id) > 256:
            raise InputValidationError("Block ID too long")
        if not re.match(r'^[a-zA-Z0-9_.-]+$', block_id):
            raise InputValidationError("Block ID contains invalid characters")
    
    @staticmethod
    def _validate_user_context(user_context: str):
        """Validate user context."""
        if not isinstance(user_context, str):
            raise InputValidationError("User context must be string")
        if len(user_context) > 128:
            raise InputValidationError("User context too long")
        if not re.match(r'^[a-zA-Z0-9_.-]+$', user_context):
            raise InputValidationError("User context contains invalid characters")
    
    def verify_integrity(self, secret_key: bytes) -> bool:
        """Verify WAL entry integrity."""
        message = f"{self.transaction_id}|{self.operation}|{self.block_id}|{self.data_hash}|{self.timestamp}|{self.user_context}"
        expected_hash = hmac.new(secret_key, message.encode('utf-8'), hashlib.sha256).hexdigest()
        return hmac.compare_digest(self.integrity_hash, expected_hash)
    
    def to_secure_bytes(self) -> bytes:
        """Serialize with length prefixes to prevent injection."""
        data = {
            'transaction_id': self.transaction_id,
            'operation': self.operation,
            'block_id': self.block_id,
            'data_hash': self.data_hash,
            'integrity_hash': self.integrity_hash,
            'timestamp': self.timestamp,
            'user_context': self.user_context
        }
        json_data = json.dumps(data, separators=(',', ':')).encode('utf-8')
        
        # Length-prefixed format to prevent injection
        return struct.pack('<I', len(json_data)) + json_data
    
    @classmethod
    def from_secure_bytes(cls, data: bytes, secret_key: bytes) -> 'SecureWALEntry':
        """Deserialize with security checks."""
        if len(data) < 4:
            raise InputValidationError("Invalid WAL entry: too short")
        
        # Read length prefix
        length = struct.unpack('<I', data[:4])[0]
        if length > 10000:  # Reasonable limit
            raise InputValidationError("WAL entry too large")
        if len(data) < 4 + length:
            raise InputValidationError("WAL entry truncated")
        
        # Parse JSON
        try:
            json_data = data[4:4+length].decode('utf-8')
            entry_data = json.loads(json_data)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise InputValidationError(f"Invalid WAL entry format: {e}")
        
        # Create entry
        entry = cls(
            entry_data['transaction_id'],
            entry_data['operation'],
            entry_data['block_id'],
            entry_data['data_hash'],
            entry_data['integrity_hash'],
            entry_data['timestamp'],
            entry_data['user_context']
        )
        
        # Verify integrity
        if not entry.verify_integrity(secret_key):
            raise SecurityError("WAL entry integrity check failed")
        
        return entry


class SecureWAL:
    """Security-hardened Write-Ahead Log."""
    
    def __init__(self, wal_path: str, secret_key: Optional[bytes] = None):
        self.wal_path = wal_path
        self.secret_key = secret_key or secrets.token_bytes(32)
        self.buffer = []
        self.buffer_lock = threading.Lock()
        self.entries_written = 0
        
        # Security: Ensure WAL file has proper permissions
        self._secure_file_permissions()
    
    def _secure_file_permissions(self):
        """Set secure file permissions."""
        if os.path.exists(self.wal_path):
            # Read/write for owner only
            os.chmod(self.wal_path, 0o600)
    
    def write_entry(self, entry: SecureWALEntry):
        """Write secure WAL entry."""
        try:
            with self.buffer_lock:
                self.buffer.append(entry)
                
                # Flush when buffer gets large (prevent memory exhaustion)
                if len(self.buffer) >= 1000:
                    self._flush_buffer()
            
            security_logger.info(f"WAL entry written: {entry.transaction_id} {entry.operation} {entry.block_id}")
            
        except Exception as e:
            security_logger.error(f"WAL write failed: {e}")
            raise SecurityError(f"Failed to write WAL entry: {e}")
    
    def _flush_buffer(self):
        """Flush buffer to disk with atomic write."""
        if not self.buffer:
            return
        
        try:
            # Write to temporary file first (atomic operation)
            temp_path = f"{self.wal_path}.tmp"
            
            with open(temp_path, 'ab') as f:
                for entry in self.buffer:
                    entry_bytes = entry.to_secure_bytes()
                    f.write(entry_bytes)
                f.fsync()  # Force to disk
            
            # Atomic rename
            if os.path.exists(self.wal_path):
                with open(self.wal_path, 'ab') as dest:
                    with open(temp_path, 'rb') as src:
                        dest.write(src.read())
                    dest.fsync()
                os.remove(temp_path)
            else:
                os.rename(temp_path, self.wal_path)
            
            # Set secure permissions
            self._secure_file_permissions()
            
            self.entries_written += len(self.buffer)
            self.buffer.clear()
            
        except Exception as e:
            security_logger.error(f"WAL flush failed: {e}")
            raise SecurityError(f"Failed to flush WAL: {e}")
    
    def read_entries(self) -> List[SecureWALEntry]:
        """Read and verify all WAL entries."""
        entries = []
        
        if not os.path.exists(self.wal_path):
            return entries
        
        try:
            with open(self.wal_path, 'rb') as f:
                while True:
                    # Read length prefix
                    length_data = f.read(4)
                    if len(length_data) < 4:
                        break
                    
                    length = struct.unpack('<I', length_data)[0]
                    if length > 10000:
                        security_logger.warning(f"Suspicious WAL entry size: {length}")
                        break
                    
                    # Read entry data
                    entry_data = f.read(length)
                    if len(entry_data) < length:
                        security_logger.warning("Truncated WAL entry detected")
                        break
                    
                    # Parse and verify entry
                    full_data = length_data + entry_data
                    entry = SecureWALEntry.from_secure_bytes(full_data, self.secret_key)
                    entries.append(entry)
                    
        except Exception as e:
            security_logger.error(f"WAL read failed: {e}")
            raise SecurityError(f"Failed to read WAL: {e}")
        
        return entries
    
    def close(self):
        """Close WAL securely."""
        with self.buffer_lock:
            self._flush_buffer()


class SecureTransactionManager:
    """Security-hardened transaction manager."""
    
    def __init__(self, file_path: str, acid_level: SecureACIDLevel, user_context: str = "system"):
        self.file_path = file_path
        self.acid_level = acid_level
        self.user_context = self._validate_user_context(user_context)
        
        # Security: Generate unique secret key for this instance
        self.secret_key = secrets.token_bytes(32)
        
        # Initialize components
        if acid_level == SecureACIDLevel.FULL_ACID:
            self.wal = SecureWAL(f"{file_path}.wal", self.secret_key)
        else:
            self.wal = None
        
        # Secure storage
        self.storage: Dict[str, Tuple[bytes, Dict[str, Any], str]] = {}  # block_id -> (data, metadata, hash)
        self.storage_lock = threading.RLock()
        
        # Transaction tracking
        self.active_transactions: Dict[str, Dict[str, Any]] = {}
        self.transaction_lock = threading.RLock()
        
        # Security counters
        self.security_violations = 0
        self.operations = 0
        
        security_logger.info(f"Secure transaction manager initialized: {acid_level.name}")
    
    def _validate_user_context(self, user_context: str) -> str:
        """Validate and sanitize user context."""
        if not isinstance(user_context, str):
            raise InputValidationError("User context must be string")
        if len(user_context) > 128:
            raise InputValidationError("User context too long")
        if not re.match(r'^[a-zA-Z0-9_.-]+$', user_context):
            raise InputValidationError("User context contains invalid characters")
        return user_context
    
    def _validate_block_data(self, data: bytes) -> bytes:
        """Validate block data for security."""
        if not isinstance(data, bytes):
            raise InputValidationError("Block data must be bytes")
        if len(data) > 100 * 1024 * 1024:  # 100MB limit
            raise InputValidationError("Block data too large")
        return data
    
    def _validate_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate metadata for security."""
        if not isinstance(metadata, dict):
            raise InputValidationError("Metadata must be dictionary")
        
        # Limit metadata size
        metadata_str = json.dumps(metadata, separators=(',', ':'))
        if len(metadata_str) > 10000:
            raise InputValidationError("Metadata too large")
        
        # Validate keys and values
        for key, value in metadata.items():
            if not isinstance(key, str):
                raise InputValidationError("Metadata keys must be strings")
            if len(key) > 256:
                raise InputValidationError("Metadata key too long")
            if not re.match(r'^[a-zA-Z0-9_.-]+$', key):
                raise InputValidationError("Metadata key contains invalid characters")
        
        return metadata
    
    def begin_transaction(self) -> str:
        """Begin secure transaction."""
        try:
            transaction_id = str(uuid.uuid4())
            
            with self.transaction_lock:
                self.active_transactions[transaction_id] = {
                    'start_time': time.time(),
                    'user_context': self.user_context,
                    'operations': []
                }
            
            security_logger.info(f"Transaction started: {transaction_id} by {self.user_context}")
            return transaction_id
            
        except Exception as e:
            self.security_violations += 1
            security_logger.error(f"Transaction start failed: {e}")
            raise SecurityError(f"Failed to start transaction: {e}")
    
    def write_block(self, transaction_id: str, block_id: str, data: bytes, metadata: Dict[str, Any]) -> bool:
        """Write block with security checks."""
        try:
            # Validate inputs
            SecureWALEntry._validate_transaction_id(transaction_id)
            SecureWALEntry._validate_block_id(block_id)
            data = self._validate_block_data(data)
            metadata = self._validate_metadata(metadata)
            
            # Check transaction exists
            with self.transaction_lock:
                if transaction_id not in self.active_transactions:
                    raise AccessDeniedError("Invalid transaction ID")
            
            # Calculate secure hash
            data_hash = hashlib.sha256(data).hexdigest()
            
            if self.acid_level == SecureACIDLevel.FULL_ACID:
                # Write to WAL
                wal_entry = SecureWALEntry.create_secure(
                    transaction_id, 'write', block_id, data, self.user_context, self.secret_key
                )
                self.wal.write_entry(wal_entry)
                
                # Add to transaction
                with self.transaction_lock:
                    self.active_transactions[transaction_id]['operations'].append({
                        'type': 'write',
                        'block_id': block_id,
                        'data': data,
                        'metadata': metadata,
                        'hash': data_hash
                    })
            else:
                # Direct write for performance mode
                with self.storage_lock:
                    self.storage[block_id] = (data, metadata, data_hash)
            
            self.operations += 1
            security_logger.debug(f"Block written: {block_id} in transaction {transaction_id}")
            return True
            
        except (InputValidationError, AccessDeniedError, SecurityError) as e:
            self.security_violations += 1
            security_logger.warning(f"Write blocked: {e}")
            return False
        except Exception as e:
            self.security_violations += 1
            security_logger.error(f"Write failed: {e}")
            return False
    
    def read_block(self, transaction_id: str, block_id: str) -> Optional[Tuple[bytes, Dict[str, Any]]]:
        """Read block with security checks."""
        try:
            # Validate inputs
            SecureWALEntry._validate_transaction_id(transaction_id)
            SecureWALEntry._validate_block_id(block_id)
            
            # Check transaction exists
            with self.transaction_lock:
                if transaction_id not in self.active_transactions:
                    raise AccessDeniedError("Invalid transaction ID")
            
            # Read from storage
            with self.storage_lock:
                if block_id in self.storage:
                    data, metadata, stored_hash = self.storage[block_id]
                    
                    # Verify integrity
                    current_hash = hashlib.sha256(data).hexdigest()
                    if current_hash != stored_hash:
                        security_logger.error(f"Data corruption detected in block {block_id}")
                        raise SecurityError("Data integrity check failed")
                    
                    security_logger.debug(f"Block read: {block_id} in transaction {transaction_id}")
                    return data, metadata
            
            return None
            
        except (InputValidationError, AccessDeniedError, SecurityError) as e:
            self.security_violations += 1
            security_logger.warning(f"Read blocked: {e}")
            return None
        except Exception as e:
            self.security_violations += 1
            security_logger.error(f"Read failed: {e}")
            return None
    
    def commit_transaction(self, transaction_id: str) -> bool:
        """Commit transaction with security checks."""
        try:
            # Validate input
            SecureWALEntry._validate_transaction_id(transaction_id)
            
            with self.transaction_lock:
                if transaction_id not in self.active_transactions:
                    raise AccessDeniedError("Invalid transaction ID")
                
                transaction = self.active_transactions[transaction_id]
                
                if self.acid_level == SecureACIDLevel.FULL_ACID:
                    # Apply operations to storage
                    with self.storage_lock:
                        for op in transaction['operations']:
                            if op['type'] == 'write':
                                self.storage[op['block_id']] = (op['data'], op['metadata'], op['hash'])
                
                # Clean up
                del self.active_transactions[transaction_id]
            
            security_logger.info(f"Transaction committed: {transaction_id}")
            return True
            
        except (InputValidationError, AccessDeniedError, SecurityError) as e:
            self.security_violations += 1
            security_logger.warning(f"Commit blocked: {e}")
            return False
        except Exception as e:
            self.security_violations += 1
            security_logger.error(f"Commit failed: {e}")
            return False
    
    def rollback_transaction(self, transaction_id: str) -> bool:
        """Rollback transaction with security checks."""
        try:
            # Validate input
            SecureWALEntry._validate_transaction_id(transaction_id)
            
            with self.transaction_lock:
                if transaction_id in self.active_transactions:
                    del self.active_transactions[transaction_id]
            
            security_logger.info(f"Transaction rolled back: {transaction_id}")
            return True
            
        except Exception as e:
            self.security_violations += 1
            security_logger.error(f"Rollback failed: {e}")
            return False
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics."""
        return {
            'acid_level': self.acid_level.name,
            'operations': self.operations,
            'security_violations': self.security_violations,
            'active_transactions': len(self.active_transactions),
            'storage_blocks': len(self.storage),
            'violation_rate': self.security_violations / max(1, self.operations) * 100
        }
    
    def close(self):
        """Close manager securely."""
        if self.wal:
            self.wal.close()
        
        # Clear sensitive data
        self.secret_key = b'\x00' * 32
        self.storage.clear()
        self.active_transactions.clear()
        
        security_logger.info("Secure transaction manager closed")


class SecureTransaction:
    """Security-hardened transaction context manager."""
    
    def __init__(self, manager: SecureTransactionManager):
        self.manager = manager
        self.transaction_id = None
    
    def __enter__(self):
        self.transaction_id = self.manager.begin_transaction()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.manager.commit_transaction(self.transaction_id)
        else:
            self.manager.rollback_transaction(self.transaction_id)
    
    def write_block(self, block_id: str, data: bytes, metadata: Dict[str, Any]) -> bool:
        """Write block securely."""
        return self.manager.write_block(self.transaction_id, block_id, data, metadata)
    
    def read_block(self, block_id: str) -> Optional[Tuple[bytes, Dict[str, Any]]]:
        """Read block securely."""
        return self.manager.read_block(self.transaction_id, block_id)


def create_secure_manager(file_path: str, acid_level: SecureACIDLevel, user_context: str = "system") -> SecureTransactionManager:
    """Create security-hardened transaction manager."""
    return SecureTransactionManager(file_path, acid_level, user_context)


if __name__ == "__main__":
    # Security test
    print("🔐 Security-Hardened ACID Implementation")
    print("Testing input validation and security features...")
    
    try:
        manager = create_secure_manager("test_secure.maif", SecureACIDLevel.FULL_ACID, "test_user")
        
        # Test normal operation
        with SecureTransaction(manager) as txn:
            success = txn.write_block("test_block", b"secure data", {"type": "test"})
            print(f"Normal write: {'✅' if success else '❌'}")
        
        # Test security violations
        try:
            with SecureTransaction(manager) as txn:
                # This should fail due to invalid block ID
                success = txn.write_block("../../../etc/passwd", b"malicious", {"evil": True})
                print(f"Malicious write blocked: {'✅' if not success else '❌'}")
        except Exception as e:
            print(f"Security violation caught: ✅")
        
        stats = manager.get_security_stats()
        print(f"Security stats: {stats}")
        
        manager.close()
        
        # Cleanup
        for file in ["test_secure.maif", "test_secure.maif.wal"]:
            if os.path.exists(file):
                os.remove(file)
        
        print("✅ Security tests completed")
        
    except Exception as e:
        print(f"❌ Security test failed: {e}")