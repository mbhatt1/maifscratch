"""
Stream-Level Access Control for MAIF
====================================

Provides granular access control enforcement during streaming operations,
including per-block permissions, time-based access, rate limiting, and
content-based access control.
"""

import time
import threading
from typing import Dict, List, Optional, Set, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from collections import defaultdict, deque


class AccessLevel(Enum):
    """Access levels for stream operations."""
    NONE = 0
    READ = 1
    WRITE = 2
    ADMIN = 4


class AccessDecision(Enum):
    """Access control decisions."""
    ALLOW = "allow"
    DENY = "deny"
    RATE_LIMITED = "rate_limited"
    EXPIRED = "expired"
    CONTENT_BLOCKED = "content_blocked"


@dataclass
class StreamAccessRule:
    """Individual access control rule for streaming."""
    rule_id: str
    user_id: str
    resource_pattern: str  # Regex pattern for MAIF files/blocks
    access_level: AccessLevel
    
    # Time-based access
    valid_from: Optional[float] = None
    valid_until: Optional[float] = None
    
    # Rate limiting
    max_bytes_per_second: Optional[int] = None
    max_blocks_per_second: Optional[int] = None
    
    # Content-based restrictions
    allowed_block_types: Optional[Set[str]] = None
    denied_block_types: Optional[Set[str]] = None
    
    # Custom validation function
    custom_validator: Optional[Callable[[str, str, bytes], bool]] = None
    
    # Metadata
    created_at: float = field(default_factory=time.time)
    description: str = ""


@dataclass
class StreamSession:
    """Active streaming session with access tracking."""
    session_id: str
    user_id: str
    resource_path: str
    start_time: float
    
    # Access tracking
    bytes_read: int = 0
    blocks_read: int = 0
    bytes_written: int = 0
    blocks_written: int = 0
    
    # Rate limiting state
    last_access_time: float = field(default_factory=time.time)
    access_history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # Security state
    access_violations: List[Dict[str, Any]] = field(default_factory=list)
    is_active: bool = True


class StreamAccessController:
    """Main access control system for MAIF streaming."""
    
    def __init__(self):
        self.rules: Dict[str, StreamAccessRule] = {}
        self.sessions: Dict[str, StreamSession] = {}
        self.global_policies: Dict[str, Any] = {}
        self._lock = threading.RLock()
        
        # Rate limiting tracking
        self._rate_buckets: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=1000)))
        
        # Audit logging
        self.audit_log: List[Dict[str, Any]] = []
        self.max_audit_entries = 10000
    
    def add_rule(self, rule: StreamAccessRule) -> None:
        """Add an access control rule."""
        with self._lock:
            self.rules[rule.rule_id] = rule
            self._log_audit("rule_added", {
                "rule_id": rule.rule_id,
                "user_id": rule.user_id,
                "access_level": rule.access_level.name
            })
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove an access control rule."""
        with self._lock:
            if rule_id in self.rules:
                rule = self.rules.pop(rule_id)
                self._log_audit("rule_removed", {
                    "rule_id": rule_id,
                    "user_id": rule.user_id
                })
                return True
            return False
    
    def create_session(self, user_id: str, resource_path: str) -> str:
        """Create a new streaming session."""
        session_id = hashlib.sha256(f"{user_id}:{resource_path}:{time.time()}".encode()).hexdigest()[:16]
        
        with self._lock:
            session = StreamSession(
                session_id=session_id,
                user_id=user_id,
                resource_path=resource_path,
                start_time=time.time()
            )
            self.sessions[session_id] = session
            
            self._log_audit("session_created", {
                "session_id": session_id,
                "user_id": user_id,
                "resource_path": resource_path
            })
            
            return session_id
    
    def close_session(self, session_id: str) -> None:
        """Close a streaming session."""
        with self._lock:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                session.is_active = False
                
                self._log_audit("session_closed", {
                    "session_id": session_id,
                    "user_id": session.user_id,
                    "duration": time.time() - session.start_time,
                    "bytes_read": session.bytes_read,
                    "blocks_read": session.blocks_read,
                    "violations": len(session.access_violations)
                })
    
    def check_stream_access(self, session_id: str, operation: str, 
                          block_type: str = None, block_data: bytes = None) -> Tuple[AccessDecision, str]:
        """
        Check access for a streaming operation.
        
        Args:
            session_id: Active session ID
            operation: 'read' or 'write'
            block_type: Type of block being accessed
            block_data: Block data (for content-based rules)
            
        Returns:
            Tuple of (decision, reason)
        """
        with self._lock:
            if session_id not in self.sessions:
                return AccessDecision.DENY, "Invalid session"
            
            session = self.sessions[session_id]
            if not session.is_active:
                return AccessDecision.DENY, "Session closed"
            
            # Find applicable rules
            applicable_rules = self._find_applicable_rules(session.user_id, session.resource_path)
            
            if not applicable_rules:
                return AccessDecision.DENY, "No applicable access rules"
            
            # Check each rule
            for rule in applicable_rules:
                decision, reason = self._evaluate_rule(rule, session, operation, block_type, block_data)
                
                if decision != AccessDecision.ALLOW:
                    # Log violation
                    violation = {
                        "timestamp": time.time(),
                        "rule_id": rule.rule_id,
                        "operation": operation,
                        "block_type": block_type,
                        "decision": decision.value,
                        "reason": reason
                    }
                    session.access_violations.append(violation)
                    
                    self._log_audit("access_denied", {
                        "session_id": session_id,
                        "user_id": session.user_id,
                        "operation": operation,
                        "decision": decision.value,
                        "reason": reason
                    })
                    
                    return decision, reason
            
            # All rules passed - update session stats
            self._update_session_stats(session, operation, block_data)
            
            return AccessDecision.ALLOW, "Access granted"
    
    def _find_applicable_rules(self, user_id: str, resource_path: str) -> List[StreamAccessRule]:
        """Find rules applicable to a user and resource."""
        import re
        applicable = []
        
        for rule in self.rules.values():
            if rule.user_id == user_id or rule.user_id == "*":
                if re.match(rule.resource_pattern, resource_path):
                    applicable.append(rule)
        
        return applicable
    
    def _evaluate_rule(self, rule: StreamAccessRule, session: StreamSession, 
                      operation: str, block_type: str, block_data: bytes) -> Tuple[AccessDecision, str]:
        """Evaluate a single access rule."""
        current_time = time.time()
        
        # Check access level
        required_level = AccessLevel.READ if operation == "read" else AccessLevel.WRITE
        if rule.access_level.value < required_level.value:
            return AccessDecision.DENY, f"Insufficient access level: {rule.access_level.name}"
        
        # Check time-based access
        if rule.valid_from and current_time < rule.valid_from:
            return AccessDecision.DENY, "Access not yet valid"
        
        if rule.valid_until and current_time > rule.valid_until:
            return AccessDecision.EXPIRED, "Access expired"
        
        # Check content-based restrictions
        if block_type:
            if rule.allowed_block_types and block_type not in rule.allowed_block_types:
                return AccessDecision.CONTENT_BLOCKED, f"Block type '{block_type}' not allowed"
            
            if rule.denied_block_types and block_type in rule.denied_block_types:
                return AccessDecision.CONTENT_BLOCKED, f"Block type '{block_type}' denied"
        
        # Check rate limiting
        if operation == "read" and rule.max_bytes_per_second:
            if not self._check_rate_limit(session, "bytes_read", rule.max_bytes_per_second, len(block_data) if block_data else 0):
                return AccessDecision.RATE_LIMITED, "Byte rate limit exceeded"
        
        if rule.max_blocks_per_second:
            if not self._check_rate_limit(session, "blocks_read" if operation == "read" else "blocks_written", rule.max_blocks_per_second, 1):
                return AccessDecision.RATE_LIMITED, "Block rate limit exceeded"
        
        # Check custom validator
        if rule.custom_validator and block_data:
            try:
                if not rule.custom_validator(session.user_id, block_type, block_data):
                    return AccessDecision.CONTENT_BLOCKED, "Custom validation failed"
            except Exception as e:
                return AccessDecision.DENY, f"Custom validator error: {str(e)}"
        
        return AccessDecision.ALLOW, "Rule passed"
    
    def _check_rate_limit(self, session: StreamSession, metric: str, limit_per_second: int, increment: int) -> bool:
        """Check if operation is within rate limits."""
        current_time = time.time()
        bucket_key = f"{session.session_id}:{metric}"
        bucket = self._rate_buckets[session.session_id][metric]
        
        # Remove old entries (older than 1 second)
        while bucket and bucket[0][0] < current_time - 1.0:
            bucket.popleft()
        
        # Calculate current rate
        current_usage = sum(entry[1] for entry in bucket)
        
        if current_usage + increment > limit_per_second:
            return False
        
        # Add current operation
        bucket.append((current_time, increment))
        return True
    
    def _update_session_stats(self, session: StreamSession, operation: str, block_data: bytes) -> None:
        """Update session statistics."""
        if operation == "read":
            session.blocks_read += 1
            if block_data:
                session.bytes_read += len(block_data)
        else:
            session.blocks_written += 1
            if block_data:
                session.bytes_written += len(block_data)
        
        session.last_access_time = time.time()
        session.access_history.append({
            "timestamp": time.time(),
            "operation": operation,
            "bytes": len(block_data) if block_data else 0
        })
    
    def _log_audit(self, event_type: str, details: Dict[str, Any]) -> None:
        """Log audit event."""
        audit_entry = {
            "timestamp": time.time(),
            "event_type": event_type,
            "details": details
        }
        
        self.audit_log.append(audit_entry)
        
        # Trim audit log if too large
        if len(self.audit_log) > self.max_audit_entries:
            self.audit_log = self.audit_log[-self.max_audit_entries//2:]
    
    def get_session_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a session."""
        with self._lock:
            if session_id not in self.sessions:
                return None
            
            session = self.sessions[session_id]
            return {
                "session_id": session_id,
                "user_id": session.user_id,
                "resource_path": session.resource_path,
                "start_time": session.start_time,
                "duration": time.time() - session.start_time,
                "bytes_read": session.bytes_read,
                "blocks_read": session.blocks_read,
                "bytes_written": session.bytes_written,
                "blocks_written": session.blocks_written,
                "violations": len(session.access_violations),
                "is_active": session.is_active,
                "last_access": session.last_access_time
            }
    
    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit log entries."""
        with self._lock:
            return self.audit_log[-limit:] if limit else self.audit_log.copy()


class SecureStreamReader:
    """Stream reader with integrated access control."""
    
    def __init__(self, maif_path: str, user_id: str, access_controller: StreamAccessController):
        self.maif_path = maif_path
        self.user_id = user_id
        self.access_controller = access_controller
        self.session_id = None
        self._base_reader = None
    
    def __enter__(self):
        # Create session
        self.session_id = self.access_controller.create_session(self.user_id, self.maif_path)
        
        # Initialize base reader
        from .streaming import MAIFStreamReader, StreamingConfig
        config = StreamingConfig()
        self._base_reader = MAIFStreamReader(self.maif_path, config)
        self._base_reader.__enter__()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.session_id:
            self.access_controller.close_session(self.session_id)
        
        if self._base_reader:
            self._base_reader.__exit__(exc_type, exc_val, exc_tb)
    
    def stream_blocks_secure(self):
        """Stream blocks with access control enforcement."""
        if not self.session_id or not self._base_reader:
            raise RuntimeError("SecureStreamReader not properly initialized")
        
        for block_type, block_data in self._base_reader.stream_blocks():
            # Check access for this block
            decision, reason = self.access_controller.check_stream_access(
                self.session_id, "read", block_type, block_data
            )
            
            if decision != AccessDecision.ALLOW:
                raise PermissionError(f"Stream access denied: {reason}")
            
            yield block_type, block_data
    
    def get_session_stats(self) -> Optional[Dict[str, Any]]:
        """Get current session statistics."""
        if self.session_id:
            return self.access_controller.get_session_stats(self.session_id)
        return None


# Predefined access control policies
class StreamAccessPolicies:
    """Common access control policies for streaming."""
    
    @staticmethod
    def create_time_limited_rule(user_id: str, resource_pattern: str, 
                               duration_seconds: int, access_level: AccessLevel = AccessLevel.READ) -> StreamAccessRule:
        """Create a time-limited access rule."""
        current_time = time.time()
        return StreamAccessRule(
            rule_id=f"time_limited_{user_id}_{int(current_time)}",
            user_id=user_id,
            resource_pattern=resource_pattern,
            access_level=access_level,
            valid_from=current_time,
            valid_until=current_time + duration_seconds,
            description=f"Time-limited access for {duration_seconds} seconds"
        )
    
    @staticmethod
    def create_rate_limited_rule(user_id: str, resource_pattern: str,
                               max_mbps: int, access_level: AccessLevel = AccessLevel.READ) -> StreamAccessRule:
        """Create a rate-limited access rule."""
        return StreamAccessRule(
            rule_id=f"rate_limited_{user_id}_{max_mbps}mbps",
            user_id=user_id,
            resource_pattern=resource_pattern,
            access_level=access_level,
            max_bytes_per_second=max_mbps * 1024 * 1024,
            description=f"Rate limited to {max_mbps} MB/s"
        )
    
    @staticmethod
    def create_content_filtered_rule(user_id: str, resource_pattern: str,
                                   allowed_types: Set[str], access_level: AccessLevel = AccessLevel.READ) -> StreamAccessRule:
        """Create a content-filtered access rule."""
        return StreamAccessRule(
            rule_id=f"content_filtered_{user_id}_{hash(frozenset(allowed_types))}",
            user_id=user_id,
            resource_pattern=resource_pattern,
            access_level=access_level,
            allowed_block_types=allowed_types,
            description=f"Content filtered to types: {', '.join(allowed_types)}"
        )