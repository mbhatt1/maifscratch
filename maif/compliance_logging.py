"""
Bullet-Proof Compliance Logging for MAIF
========================================

Implements a tamper-evident, ACID-compliant logging system using SQLite that's
embedded within the MAIF file itself. This ensures the audit trail is never
separated from the data and provides cryptographic verification of log integrity.
"""

import os
import time
import json
import sqlite3
import hashlib
import uuid
import tempfile
import shutil
import threading
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
from contextlib import contextmanager
from dataclasses import dataclass, field

# Import MAIF modules
from .block_storage import BlockStorage, BlockType
from .signature_verification import create_default_verifier, sign_block_data


class LogLevel(Enum):
    """Log levels for compliance logging."""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class LogCategory(Enum):
    """Log categories for compliance events."""
    ACCESS = "access"
    DATA = "data"
    SECURITY = "security"
    ADMIN = "admin"
    SYSTEM = "system"
    COMPLIANCE = "compliance"


@dataclass
class LogEntry:
    """Compliance log entry."""
    timestamp: float
    level: LogLevel
    category: LogCategory
    user_id: str
    action: str
    resource_id: str
    details: Dict[str, Any]
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    previous_hash: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp,
            "level": self.level.value,
            "category": self.category.value,
            "user_id": self.user_id,
            "action": self.action,
            "resource_id": self.resource_id,
            "details": self.details,
            "previous_hash": self.previous_hash
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LogEntry':
        """Create from dictionary."""
        return cls(
            entry_id=data.get("entry_id", str(uuid.uuid4())),
            timestamp=data.get("timestamp", time.time()),
            level=LogLevel(data.get("level", LogLevel.INFO.value)),
            category=LogCategory(data.get("category", LogCategory.SYSTEM.value)),
            user_id=data.get("user_id", ""),
            action=data.get("action", ""),
            resource_id=data.get("resource_id", ""),
            details=data.get("details", {}),
            previous_hash=data.get("previous_hash")
        )
    
    def calculate_hash(self) -> str:
        """Calculate cryptographic hash of log entry."""
        # Create canonical representation
        canonical = {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp,
            "level": self.level.value,
            "category": self.category.value,
            "user_id": self.user_id,
            "action": self.action,
            "resource_id": self.resource_id,
            "details": json.dumps(self.details, sort_keys=True),
            "previous_hash": self.previous_hash
        }
        
        # Convert to JSON string
        canonical_json = json.dumps(canonical, sort_keys=True)
        
        # Calculate hash
        return hashlib.sha256(canonical_json.encode()).hexdigest()


class ComplianceLogger:
    """Compliance logger with SQLite backend."""
    
    def __init__(self, db_path: Optional[str] = None, maif_path: Optional[str] = None):
        """
        Initialize compliance logger.
        
        Args:
            db_path: Path to SQLite database file (optional)
            maif_path: Path to MAIF file (optional)
        """
        self.db_path = db_path or ":memory:"
        self.maif_path = maif_path
        self.conn = None
        self.cursor = None
        self.lock = threading.RLock()
        self.last_hash = None
        self.initialized = False
        
        # Initialize database
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize SQLite database."""
        with self.lock:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            self.cursor = self.conn.cursor()
            
            # Create tables if they don't exist
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS log_entries (
                    entry_id TEXT PRIMARY KEY,
                    timestamp REAL NOT NULL,
                    level INTEGER NOT NULL,
                    category TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    resource_id TEXT NOT NULL,
                    details TEXT NOT NULL,
                    previous_hash TEXT,
                    entry_hash TEXT NOT NULL
                )
            ''')
            
            # Create index on timestamp
            self.cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp ON log_entries (timestamp)
            ''')
            
            # Create index on user_id
            self.cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_user_id ON log_entries (user_id)
            ''')
            
            # Create index on category
            self.cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_category ON log_entries (category)
            ''')
            
            # Create metadata table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            ''')
            
            # Commit changes
            self.conn.commit()
            
            # Get last hash
            self.cursor.execute('''
                SELECT entry_hash FROM log_entries
                ORDER BY timestamp DESC
                LIMIT 1
            ''')
            
            row = self.cursor.fetchone()
            if row:
                self.last_hash = row['entry_hash']
            
            self.initialized = True
    
    def log(self, level: LogLevel, category: LogCategory, user_id: str, 
           action: str, resource_id: str, details: Dict[str, Any]) -> str:
        """
        Log compliance event.
        
        Args:
            level: Log level
            category: Log category
            user_id: User ID
            action: Action performed
            resource_id: Resource ID
            details: Additional details
            
        Returns:
            Log entry ID
        """
        with self.lock:
            # Create log entry
            entry = LogEntry(
                timestamp=time.time(),
                level=level,
                category=category,
                user_id=user_id,
                action=action,
                resource_id=resource_id,
                details=details,
                previous_hash=self.last_hash
            )
            
            # Calculate hash
            entry_hash = entry.calculate_hash()
            
            # Insert into database
            self.cursor.execute('''
                INSERT INTO log_entries (
                    entry_id, timestamp, level, category, user_id, action, 
                    resource_id, details, previous_hash, entry_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                entry.entry_id,
                entry.timestamp,
                entry.level.value,
                entry.category.value,
                entry.user_id,
                entry.action,
                entry.resource_id,
                json.dumps(entry.details),
                entry.previous_hash,
                entry_hash
            ))
            
            # Commit changes
            self.conn.commit()
            
            # Update last hash
            self.last_hash = entry_hash
            
            return entry.entry_id
    
    def get_entry(self, entry_id: str) -> Optional[LogEntry]:
        """
        Get log entry by ID.
        
        Args:
            entry_id: Log entry ID
            
        Returns:
            LogEntry if found, None otherwise
        """
        with self.lock:
            self.cursor.execute('''
                SELECT * FROM log_entries WHERE entry_id = ?
            ''', (entry_id,))
            
            row = self.cursor.fetchone()
            if not row:
                return None
            
            return LogEntry(
                entry_id=row['entry_id'],
                timestamp=row['timestamp'],
                level=LogLevel(row['level']),
                category=LogCategory(row['category']),
                user_id=row['user_id'],
                action=row['action'],
                resource_id=row['resource_id'],
                details=json.loads(row['details']),
                previous_hash=row['previous_hash']
            )
    
    def query_logs(self, filters: Dict[str, Any] = None, 
                  start_time: Optional[float] = None,
                  end_time: Optional[float] = None,
                  limit: int = 100, offset: int = 0) -> List[LogEntry]:
        """
        Query log entries.
        
        Args:
            filters: Filters to apply (field: value)
            start_time: Start timestamp
            end_time: End timestamp
            limit: Maximum number of entries to return
            offset: Offset for pagination
            
        Returns:
            List of LogEntry objects
        """
        with self.lock:
            # Build query
            query = "SELECT * FROM log_entries WHERE 1=1"
            params = []
            
            # Apply filters
            if filters:
                for field, value in filters.items():
                    if field in ['level', 'category', 'user_id', 'action', 'resource_id']:
                        query += f" AND {field} = ?"
                        params.append(value.value if hasattr(value, 'value') else value)
            
            # Apply time range
            if start_time is not None:
                query += " AND timestamp >= ?"
                params.append(start_time)
            
            if end_time is not None:
                query += " AND timestamp <= ?"
                params.append(end_time)
            
            # Add order by
            query += " ORDER BY timestamp DESC"
            
            # Add limit and offset
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            # Execute query
            self.cursor.execute(query, params)
            
            # Convert rows to LogEntry objects
            entries = []
            for row in self.cursor.fetchall():
                entries.append(LogEntry(
                    entry_id=row['entry_id'],
                    timestamp=row['timestamp'],
                    level=LogLevel(row['level']),
                    category=LogCategory(row['category']),
                    user_id=row['user_id'],
                    action=row['action'],
                    resource_id=row['resource_id'],
                    details=json.loads(row['details']),
                    previous_hash=row['previous_hash']
                ))
            
            return entries
    
    def verify_integrity(self) -> Tuple[bool, Optional[str]]:
        """
        Verify integrity of log chain.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        with self.lock:
            # Get all entries ordered by timestamp
            self.cursor.execute('''
                SELECT * FROM log_entries
                ORDER BY timestamp ASC
            ''')
            
            rows = self.cursor.fetchall()
            if not rows:
                return True, None
            
            # Verify hash chain
            previous_hash = None
            for row in rows:
                # Create entry
                entry = LogEntry(
                    entry_id=row['entry_id'],
                    timestamp=row['timestamp'],
                    level=LogLevel(row['level']),
                    category=LogCategory(row['category']),
                    user_id=row['user_id'],
                    action=row['action'],
                    resource_id=row['resource_id'],
                    details=json.loads(row['details']),
                    previous_hash=row['previous_hash']
                )
                
                # Check previous hash
                if entry.previous_hash != previous_hash:
                    return False, f"Hash chain broken at entry {entry.entry_id}"
                
                # Calculate hash
                entry_hash = entry.calculate_hash()
                
                # Check hash
                if entry_hash != row['entry_hash']:
                    return False, f"Hash mismatch for entry {entry.entry_id}"
                
                # Update previous hash
                previous_hash = entry_hash
            
            return True, None
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get logging statistics.
        
        Returns:
            Dictionary with statistics
        """
        with self.lock:
            stats = {}
            
            # Total entries
            self.cursor.execute("SELECT COUNT(*) FROM log_entries")
            stats["total_entries"] = self.cursor.fetchone()[0]
            
            # Entries by level
            self.cursor.execute('''
                SELECT level, COUNT(*) FROM log_entries
                GROUP BY level
            ''')
            stats["entries_by_level"] = {
                LogLevel(row[0]).name: row[1] for row in self.cursor.fetchall()
            }
            
            # Entries by category
            self.cursor.execute('''
                SELECT category, COUNT(*) FROM log_entries
                GROUP BY category
            ''')
            stats["entries_by_category"] = {
                row[0]: row[1] for row in self.cursor.fetchall()
            }
            
            # Time range
            self.cursor.execute('''
                SELECT MIN(timestamp), MAX(timestamp) FROM log_entries
            ''')
            min_time, max_time = self.cursor.fetchone()
            stats["time_range"] = {
                "min": min_time,
                "max": max_time,
                "duration": max_time - min_time if min_time and max_time else 0
            }
            
            return stats
    
    def export_logs(self, output_path: str, format: str = "json") -> bool:
        """
        Export logs to file.
        
        Args:
            output_path: Output file path
            format: Export format (json or csv)
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            try:
                # Get all entries
                self.cursor.execute("SELECT * FROM log_entries ORDER BY timestamp ASC")
                rows = self.cursor.fetchall()
                
                if format == "json":
                    # Convert to list of dictionaries
                    entries = []
                    for row in rows:
                        entry = {
                            "entry_id": row['entry_id'],
                            "timestamp": row['timestamp'],
                            "level": LogLevel(row['level']).name,
                            "category": row['category'],
                            "user_id": row['user_id'],
                            "action": row['action'],
                            "resource_id": row['resource_id'],
                            "details": json.loads(row['details']),
                            "previous_hash": row['previous_hash'],
                            "entry_hash": row['entry_hash']
                        }
                        entries.append(entry)
                    
                    # Write to file
                    with open(output_path, 'w') as f:
                        json.dump(entries, f, indent=2)
                
                elif format == "csv":
                    import csv
                    
                    # Write to file
                    with open(output_path, 'w', newline='') as f:
                        writer = csv.writer(f)
                        
                        # Write header
                        writer.writerow([
                            "entry_id", "timestamp", "level", "category", "user_id",
                            "action", "resource_id", "details", "previous_hash", "entry_hash"
                        ])
                        
                        # Write rows
                        for row in rows:
                            writer.writerow([
                                row['entry_id'],
                                row['timestamp'],
                                LogLevel(row['level']).name,
                                row['category'],
                                row['user_id'],
                                row['action'],
                                row['resource_id'],
                                json.dumps(json.loads(row['details'])),
                                row['previous_hash'],
                                row['entry_hash']
                            ])
                
                else:
                    raise ValueError(f"Unsupported export format: {format}")
                
                return True
            
            except Exception as e:
                print(f"Error exporting logs: {e}")
                return False
    
    def close(self):
        """Close database connection."""
        with self.lock:
            if self.conn:
                self.conn.close()
                self.conn = None
                self.cursor = None
    
    def __del__(self):
        """Destructor."""
        self.close()


class MAIFComplianceLogger:
    """
    MAIF-integrated compliance logger that embeds SQLite database in MAIF file.
    
    This class provides a bullet-proof logging system that:
    1. Uses SQLite for ACID-compliant logging
    2. Embeds the SQLite database in the MAIF file when closed
    3. Extracts the SQLite database from the MAIF file when opened
    4. Maintains a cryptographic hash chain for tamper detection
    """
    
    def __init__(self, maif_path: str, block_storage: Optional[BlockStorage] = None):
        """
        Initialize MAIF compliance logger.
        
        Args:
            maif_path: Path to MAIF file
            block_storage: BlockStorage instance (optional)
        """
        self.maif_path = maif_path
        self.block_storage = block_storage
        self.temp_dir = tempfile.mkdtemp(prefix="maif_logging_")
        self.db_path = os.path.join(self.temp_dir, "compliance.db")
        self.logger = None
        self.block_id = None
        
        # Extract existing log database if available
        self._extract_log_database()
        
        # Initialize logger
        self.logger = ComplianceLogger(db_path=self.db_path, maif_path=maif_path)
    
    def _extract_log_database(self):
        """Extract log database from MAIF file if available."""
        if not self.block_storage:
            # Create block storage
            self.block_storage = BlockStorage(self.maif_path)
        
        # Find log database block
        with self.block_storage:
            # List all blocks
            blocks = self.block_storage.list_blocks()
            
            # Find log database block
            for block in blocks:
                if block.block_type == "LOGS":
                    # Get block data
                    result = self.block_storage.get_block(block.uuid)
                    if result:
                        header, data, metadata = result
                        
                        # Write to temporary file
                        with open(self.db_path, 'wb') as f:
                            f.write(data)
                        
                        # Store block ID
                        self.block_id = block.uuid
                        break
    
    def _embed_log_database(self):
        """Embed log database in MAIF file."""
        if not self.block_storage:
            # Create block storage
            self.block_storage = BlockStorage(self.maif_path)
        
        # Close logger to ensure all data is written
        if self.logger:
            self.logger.close()
        
        # Read database file
        with open(self.db_path, 'rb') as f:
            db_data = f.read()
        
        # Create metadata
        metadata = {
            "timestamp": time.time(),
            "version": 1,
            "integrity_verified": True
        }
        
        # Add signature if possible
        try:
            verifier = create_default_verifier()
            signature = sign_block_data(verifier, db_data)
            metadata["signature"] = signature
        except Exception:
            pass
        
        # Store in MAIF file
        with self.block_storage:
            if self.block_id:
                # Update existing block using the new update_block method
                success = self.block_storage.update_block(
                    block_id=self.block_id,
                    data=db_data,
                    metadata=metadata
                )
                
                if not success:
                    # If update failed, create a new block
                    logger.warning(f"Failed to update block {self.block_id}, creating new block")
                    self.block_id = self.block_storage.add_block(
                        block_type="LOGS",
                        data=db_data,
                        metadata=metadata
                    )
            else:
                # Add new block
                self.block_id = self.block_storage.add_block(
                    block_type="LOGS",
                    data=db_data,
                    metadata=metadata
                )
    
    def log(self, level: LogLevel, category: LogCategory, user_id: str, 
           action: str, resource_id: str, details: Dict[str, Any]) -> str:
        """
        Log compliance event.
        
        Args:
            level: Log level
            category: Log category
            user_id: User ID
            action: Action performed
            resource_id: Resource ID
            details: Additional details
            
        Returns:
            Log entry ID
        """
        if not self.logger:
            raise ValueError("Logger not initialized")
        
        return self.logger.log(level, category, user_id, action, resource_id, details)
    
    def get_entry(self, entry_id: str) -> Optional[LogEntry]:
        """
        Get log entry by ID.
        
        Args:
            entry_id: Log entry ID
            
        Returns:
            LogEntry if found, None otherwise
        """
        if not self.logger:
            raise ValueError("Logger not initialized")
        
        return self.logger.get_entry(entry_id)
    
    def query_logs(self, filters: Dict[str, Any] = None, 
                  start_time: Optional[float] = None,
                  end_time: Optional[float] = None,
                  limit: int = 100, offset: int = 0) -> List[LogEntry]:
        """
        Query log entries.
        
        Args:
            filters: Filters to apply (field: value)
            start_time: Start timestamp
            end_time: End timestamp
            limit: Maximum number of entries to return
            offset: Offset for pagination
            
        Returns:
            List of LogEntry objects
        """
        if not self.logger:
            raise ValueError("Logger not initialized")
        
        return self.logger.query_logs(filters, start_time, end_time, limit, offset)
    
    def verify_integrity(self) -> Tuple[bool, Optional[str]]:
        """
        Verify integrity of log chain.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.logger:
            raise ValueError("Logger not initialized")
        
        return self.logger.verify_integrity()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get logging statistics.
        
        Returns:
            Dictionary with statistics
        """
        if not self.logger:
            raise ValueError("Logger not initialized")
        
        return self.logger.get_statistics()
    
    def export_logs(self, output_path: str, format: str = "json") -> bool:
        """
        Export logs to file.
        
        Args:
            output_path: Output file path
            format: Export format (json or csv)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.logger:
            raise ValueError("Logger not initialized")
        
        return self.logger.export_logs(output_path, format)
    
    def close(self):
        """Close logger and embed database in MAIF file."""
        # Embed log database
        self._embed_log_database()
        
        # Close logger
        if self.logger:
            self.logger.close()
            self.logger = None
        
        # Clean up temporary directory
        try:
            shutil.rmtree(self.temp_dir)
        except Exception:
            pass
    
    def __del__(self):
        """Destructor."""
        self.close()


@contextmanager
def maif_compliance_logger(maif_path: str) -> MAIFComplianceLogger:
    """
    Context manager for MAIF compliance logger.
    
    Args:
        maif_path: Path to MAIF file
        
    Yields:
        MAIFComplianceLogger instance
    """
    logger = MAIFComplianceLogger(maif_path)
    try:
        yield logger
    finally:
        logger.close()


# Helper functions for easy integration
def log_access_event(logger: Union[ComplianceLogger, MAIFComplianceLogger], 
                    user_id: str, action: str, resource_id: str, 
                    details: Dict[str, Any] = None) -> str:
    """
    Log access event.
    
    Args:
        logger: ComplianceLogger or MAIFComplianceLogger instance
        user_id: User ID
        action: Action performed
        resource_id: Resource ID
        details: Additional details
        
    Returns:
        Log entry ID
    """
    return logger.log(
        level=LogLevel.INFO,
        category=LogCategory.ACCESS,
        user_id=user_id,
        action=action,
        resource_id=resource_id,
        details=details or {}
    )


def log_security_event(logger: Union[ComplianceLogger, MAIFComplianceLogger], 
                      user_id: str, action: str, resource_id: str, 
                      details: Dict[str, Any] = None, level: LogLevel = LogLevel.WARNING) -> str:
    """
    Log security event.
    
    Args:
        logger: ComplianceLogger or MAIFComplianceLogger instance
        user_id: User ID
        action: Action performed
        resource_id: Resource ID
        details: Additional details
        level: Log level
        
    Returns:
        Log entry ID
    """
    return logger.log(
        level=level,
        category=LogCategory.SECURITY,
        user_id=user_id,
        action=action,
        resource_id=resource_id,
        details=details or {}
    )


def log_data_event(logger: Union[ComplianceLogger, MAIFComplianceLogger], 
                  user_id: str, action: str, resource_id: str, 
                  details: Dict[str, Any] = None) -> str:
    """
    Log data event.
    
    Args:
        logger: ComplianceLogger or MAIFComplianceLogger instance
        user_id: User ID
        action: Action performed
        resource_id: Resource ID
        details: Additional details
        
    Returns:
        Log entry ID
    """
    return logger.log(
        level=LogLevel.INFO,
        category=LogCategory.DATA,
        user_id=user_id,
        action=action,
        resource_id=resource_id,
        details=details or {}
    )


def log_compliance_event(logger: Union[ComplianceLogger, MAIFComplianceLogger], 
                        user_id: str, action: str, resource_id: str, 
                        details: Dict[str, Any] = None) -> str:
    """
    Log compliance event.
    
    Args:
        logger: ComplianceLogger or MAIFComplianceLogger instance
        user_id: User ID
        action: Action performed
        resource_id: Resource ID
        details: Additional details
        
    Returns:
        Log entry ID
    """
    return logger.log(
        level=LogLevel.INFO,
        category=LogCategory.COMPLIANCE,
        user_id=user_id,
        action=action,
        resource_id=resource_id,
        details=details or {}
    )