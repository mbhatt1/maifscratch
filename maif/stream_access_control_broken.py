"""
Stream-Level Access Control for MAIF
====================================

Provides granular access control enforcement during streaming operations,
including per-block permissions, time-based access, rate limiting, and
content-based access control with advanced security features.
"""

import time
import threading
import secrets
import hmac
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
    """Active streaming session with access tracking and security features."""
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
    
    # Anti-replay protection
    nonce_history: Set[str] = field(default_factory=set)
    last_request_timestamp: float = 0.0
    
    # Multi-factor authentication
    mfa_verified: bool = False
    mfa_required: bool = False
    mfa_challenge_time: Optional[float] = None
    
    # Behavioral analysis
    access_pattern_hash: str = ""
    suspicious_activity_score: float = 0.0


class StreamAccessController:
    """Main access control system for MAIF streaming with advanced security."""
    
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
        
        # Security enhancements
        self._timing_randomization = True
        self._min_response_time = 0.001  # 1ms minimum response time
        self._max_response_time = 0.010  # 10ms maximum response time
        
        # Anti-replay protection
        self._global_nonce_history: Set[str] = set()
        self._nonce_cleanup_interval = 3600  # 1 hour
        self._last_nonce_cleanup = time.time()
        
        # MFA settings
        self._mfa_secret_key = secrets.token_bytes(32)
        self._mfa_timeout = 300  # 5 minutes
        
        # Behavioral analysis
        self._behavioral_patterns: Dict[str, List[Dict]] = defaultdict(list)
        self._anomaly_threshold = 0.7
    
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
        # Basic access control implementation
        if session_id not in self.active_sessions:
            return AccessDecision.DENY, "Invalid session"
        
        session = self.active_sessions[session_id]
        
        # Check if session is expired
        if time.time() - session.start_time > session.max_duration:
            return AccessDecision.DENY, "Session expired"
        
        # Apply access rules
        for rule in self.access_rules:
            if rule.applies_to(session.user_id, operation, block_type):
                decision = rule.evaluate(session, block_data)
                if decision != AccessDecision.ALLOW:
                    return decision, f"Access denied by rule: {rule.name}"
        
        return AccessDecision.ALLOW, "Access granted"
    
    def check_stream_access_secure(self, session_id: str, operation: str,
                                  block_type: str = None, block_data: bytes = None,
                                  request_nonce: str = None, request_timestamp: float = None) -> Tuple[AccessDecision, str]:
        """
        Enhanced access check with anti-replay, timing attack protection, and MFA.
        
        Args:
            session_id: Active session ID
            operation: 'read' or 'write'
            block_type: Type of block being accessed
            block_data: Block data (for content-based rules)
            request_nonce: Unique request identifier (anti-replay)
            request_timestamp: Request timestamp (anti-replay)
            
        Returns:
            Tuple of (decision, reason)
        """
        start_time = time.time()
        
        try:
            # 1. Anti-replay protection
            if request_nonce and request_timestamp:
                replay_check = self._check_anti_replay(session_id, request_nonce, request_timestamp)
                if replay_check != AccessDecision.ALLOW:
                    return self._timing_safe_response(start_time, replay_check, "Replay attack detected")
            
            # 2. MFA verification for sensitive operations
            mfa_check = self._check_mfa_requirement(session_id, operation, block_type)
            if mfa_check != AccessDecision.ALLOW:
                return self._timing_safe_response(start_time, mfa_check, "MFA verification required")
            
            # 3. Behavioral analysis
            behavioral_check = self._analyze_behavioral_pattern(session_id, operation, block_type)
            if behavioral_check != AccessDecision.ALLOW:
                return self._timing_safe_response(start_time, behavioral_check, "Suspicious behavioral pattern")
            
            # 4. Standard access control check
            decision, reason = self.check_stream_access(session_id, operation, block_type, block_data)
            
            # 5. Update behavioral patterns
            self._update_behavioral_pattern(session_id, operation, block_type, decision)
            
            return self._timing_safe_response(start_time, decision, reason)
            
        except Exception as e:
            return self._timing_safe_response(start_time, AccessDecision.DENY, f"Security check failed: {str(e)}")
    
    def _check_anti_replay(self, session_id: str, nonce: str, timestamp: float) -> AccessDecision:
        """Check for replay attacks using nonce and timestamp validation."""
        current_time = time.time()
        
        # Check timestamp freshness (within 30 seconds)
        if abs(current_time - timestamp) > 30:
            return AccessDecision.DENY
        
        # Check global nonce uniqueness
        if nonce in self._global_nonce_history:
            return AccessDecision.DENY
        
        # Check session-specific nonce
        if session_id in self.sessions:
            session = self.sessions[session_id]
            if nonce in session.nonce_history:
                return AccessDecision.DENY
            
            # Check timestamp ordering (must be newer than last request)
            if timestamp <= session.last_request_timestamp:
                return AccessDecision.DENY
            
            # Update session state
            session.nonce_history.add(nonce)
            session.last_request_timestamp = timestamp
            
            # Limit nonce history size
            if len(session.nonce_history) > 1000:
                # Remove oldest nonces (approximate)
                session.nonce_history = set(list(session.nonce_history)[-500:])
        
        # Add to global nonce history
        self._global_nonce_history.add(nonce)
        
        # Cleanup old nonces periodically
        if current_time - self._last_nonce_cleanup > self._nonce_cleanup_interval:
            self._cleanup_old_nonces()
        
        return AccessDecision.ALLOW
    
    def _check_mfa_requirement(self, session_id: str, operation: str, block_type: str) -> AccessDecision:
        """Check if MFA is required and verified for this operation."""
        if session_id not in self.sessions:
            return AccessDecision.DENY
        
        session = self.sessions[session_id]
        
        # Determine if MFA is required based on operation sensitivity
        requires_mfa = (
            session.mfa_required or
            operation == "write" or
            block_type in ["SECU", "ACLS", "PROV"] or  # Security-sensitive blocks
            session.suspicious_activity_score > 0.5
        )
        
        if requires_mfa and not session.mfa_verified:
            return AccessDecision.DENY
        
        # Check MFA timeout
        if session.mfa_verified and session.mfa_challenge_time:
            if time.time() - session.mfa_challenge_time > self._mfa_timeout:
                session.mfa_verified = False
                return AccessDecision.DENY
        
        return AccessDecision.ALLOW
    
    def _analyze_behavioral_pattern(self, session_id: str, operation: str, block_type: str) -> AccessDecision:
        """Analyze user behavior for anomalies."""
        if session_id not in self.sessions:
            return AccessDecision.DENY
        
        session = self.sessions[session_id]
        current_time = time.time()
        
        # Create behavior signature
        behavior = {
            "operation": operation,
            "block_type": block_type,
            "timestamp": current_time,
            "hour_of_day": int((current_time % 86400) / 3600),
            "day_of_week": int((current_time / 86400) % 7)
        }
        
        # Get user's historical patterns
        user_patterns = self._behavioral_patterns[session.user_id]
        
        if len(user_patterns) < 10:
            # Not enough data for analysis
            user_patterns.append(behavior)
            return AccessDecision.ALLOW
        
        # Simple anomaly detection based on patterns
        anomaly_score = self._calculate_anomaly_score(behavior, user_patterns)
        session.suspicious_activity_score = anomaly_score
        
        if anomaly_score > self._anomaly_threshold:
            # Require MFA for suspicious activity
            session.mfa_required = True
            if not session.mfa_verified:
                return AccessDecision.DENY
        
        # Update patterns (keep last 100 behaviors)
        user_patterns.append(behavior)
        if len(user_patterns) > 100:
            user_patterns.pop(0)
        
        return AccessDecision.ALLOW
    
    def _calculate_anomaly_score(self, current_behavior: Dict, historical_patterns: List[Dict]) -> float:
        """Calculate anomaly score based on historical patterns."""
        if not historical_patterns:
            return 0.0
        
        # Simple scoring based on common patterns
        score = 0.0
        
        # Check operation frequency
        operation_count = sum(1 for p in historical_patterns if p["operation"] == current_behavior["operation"])
        operation_frequency = operation_count / len(historical_patterns)
        if operation_frequency < 0.1:  # Rare operation
            score += 0.3
        
        # Check time-of-day patterns
        hour = current_behavior["hour_of_day"]
        hour_count = sum(1 for p in historical_patterns if abs(p["hour_of_day"] - hour) <= 1)
        hour_frequency = hour_count / len(historical_patterns)
        if hour_frequency < 0.1:  # Unusual time
            score += 0.4
        
        # Check block type patterns
        if current_behavior["block_type"]:
            block_count = sum(1 for p in historical_patterns if p["block_type"] == current_behavior["block_type"])
            block_frequency = block_count / len(historical_patterns)
            if block_frequency < 0.05:  # Very rare block type
                score += 0.3
        
        return min(score, 1.0)
    
    def _timing_safe_response(self, start_time: float, decision: AccessDecision, reason: str) -> Tuple[AccessDecision, str]:
        """Return response with timing attack protection."""
        if not self._timing_randomization:
            return decision, reason
        
        elapsed = time.time() - start_time
        
        # Add random delay to normalize response times
        if elapsed < self._min_response_time:
            delay = self._min_response_time - elapsed + secrets.randbelow(int(self._max_response_time * 1000)) / 1000
            time.sleep(delay)
        elif elapsed > self._max_response_time:
            # Response took too long, add small random delay
            time.sleep(secrets.randbelow(5) / 1000)  # 0-5ms
        else:
            # Add small random delay to mask timing differences
            time.sleep(secrets.randbelow(3) / 1000)  # 0-3ms
        
        return decision, reason
    
    def _update_behavioral_pattern(self, session_id: str, operation: str, block_type: str, decision: AccessDecision) -> None:
        """Update behavioral patterns based on access decision."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            
            # Adjust suspicion score based on access patterns
            if decision == AccessDecision.DENY:
                session.suspicious_activity_score = min(session.suspicious_activity_score + 0.1, 1.0)
            else:
                session.suspicious_activity_score = max(session.suspicious_activity_score - 0.05, 0.0)
    
    def _cleanup_old_nonces(self) -> None:
        """Clean up old nonces to prevent memory growth."""
        # This is a simplified cleanup - in production, you'd want more sophisticated cleanup
        if len(self._global_nonce_history) > 10000:
            # Keep only recent nonces (this is approximate)
            self._global_nonce_history = set(list(self._global_nonce_history)[-5000:])
        
        self._last_nonce_cleanup = time.time()
    
    def initiate_mfa_challenge(self, session_id: str) -> Optional[str]:
        """Initiate MFA challenge for a session."""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        # Generate MFA challenge (simplified - in production use proper TOTP/SMS)
        challenge = secrets.token_hex(16)
        session.mfa_challenge_time = time.time()
        
        self._log_audit("mfa_challenge_initiated", {
            "session_id": session_id,
            "user_id": session.user_id,
            "challenge_time": session.mfa_challenge_time
        })
        
        return challenge
    
    def verify_mfa_response(self, session_id: str, response: str, expected_response: str) -> bool:
        """Verify MFA response."""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        
        # Verify response (simplified - in production use proper verification)
        if hmac.compare_digest(response, expected_response):
            session.mfa_verified = True
            session.mfa_challenge_time = time.time()
            
            self._log_audit("mfa_verification_success", {
                "session_id": session_id,
                "user_id": session.user_id
            })
            
            return True
        else:
            session.suspicious_activity_score = min(session.suspicious_activity_score + 0.2, 1.0)
            
            self._log_audit("mfa_verification_failed", {
                "session_id": session_id,
                "user_id": session.user_id
            })
            
            return False
        
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