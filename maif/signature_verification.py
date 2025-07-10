"""
MAIF Signature Verification
===========================

Implements robust cryptographic signature verification for MAIF blocks
to ensure data integrity and authenticity.
"""

import hashlib
import hmac
import time
import base64
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import secrets
import json

class SignatureAlgorithm(Enum):
    """Supported signature algorithms."""
    HMAC_SHA256 = "hmac-sha256"
    HMAC_SHA512 = "hmac-sha512"
    ED25519 = "ed25519"
    ECDSA_P256 = "ecdsa-p256"


class VerificationResult(Enum):
    """Signature verification results."""
    VALID = "valid"
    INVALID = "invalid"
    KEY_NOT_FOUND = "key_not_found"
    ALGORITHM_UNSUPPORTED = "algorithm_unsupported"
    EXPIRED = "expired"
    MALFORMED = "malformed"


@dataclass
class SignatureInfo:
    """Signature metadata."""
    algorithm: SignatureAlgorithm
    key_id: str
    timestamp: float
    nonce: str
    signature: bytes
    expiration: Optional[float] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SignatureInfo':
        """Create SignatureInfo from dictionary."""
        try:
            algorithm = SignatureAlgorithm(data.get("algorithm", "hmac-sha256"))
            key_id = data.get("key_id", "")
            timestamp = float(data.get("timestamp", 0))
            nonce = data.get("nonce", "")
            signature = base64.b64decode(data.get("signature", ""))
            expiration = float(data.get("expiration")) if "expiration" in data else None
            
            return cls(
                algorithm=algorithm,
                key_id=key_id,
                timestamp=timestamp,
                nonce=nonce,
                signature=signature,
                expiration=expiration
            )
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid signature info: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "algorithm": self.algorithm.value,
            "key_id": self.key_id,
            "timestamp": self.timestamp,
            "nonce": self.nonce,
            "signature": base64.b64encode(self.signature).decode('utf-8')
        }
        
        if self.expiration is not None:
            result["expiration"] = self.expiration
            
        return result


class KeyStore:
    """Secure key storage for signature verification."""
    
    def __init__(self):
        self.keys: Dict[str, Dict[str, Any]] = {}
        self.trusted_issuers: List[str] = []
        
    def add_key(self, key_id: str, key_data: bytes, algorithm: SignatureAlgorithm, 
                issuer: str = "self", expiration: Optional[float] = None):
        """Add a verification key."""
        self.keys[key_id] = {
            "key_data": key_data,
            "algorithm": algorithm,
            "issuer": issuer,
            "added_at": time.time(),
            "expiration": expiration
        }
        
    def get_key(self, key_id: str) -> Optional[Dict[str, Any]]:
        """Get key by ID."""
        key = self.keys.get(key_id)
        
        if key and key.get("expiration") and time.time() > key["expiration"]:
            # Key expired
            return None
            
        return key
        
    def add_trusted_issuer(self, issuer_id: str):
        """Add trusted key issuer."""
        if issuer_id not in self.trusted_issuers:
            self.trusted_issuers.append(issuer_id)
            
    def is_trusted_issuer(self, issuer_id: str) -> bool:
        """Check if issuer is trusted."""
        return issuer_id in self.trusted_issuers


class SignatureVerifier:
    """Verifies cryptographic signatures for MAIF blocks."""
    
    def __init__(self, key_store: Optional[KeyStore] = None):
        self.key_store = key_store or KeyStore()
        self.nonce_cache: Dict[str, float] = {}
        self.nonce_cleanup_threshold = 1000
        self.verification_results: List[Dict[str, Any]] = []
        
    def verify_signature(self, data: bytes, signature_info: SignatureInfo, 
                        max_age: Optional[float] = None) -> VerificationResult:
        """
        Verify signature with comprehensive checks.
        
        Args:
            data: The data that was signed
            signature_info: Signature metadata
            max_age: Maximum age of signature in seconds
            
        Returns:
            VerificationResult enum indicating result
        """
        # Record verification attempt
        self._record_verification_attempt(signature_info, data)
        
        # Check signature timestamp
        current_time = time.time()
        if max_age and (current_time - signature_info.timestamp > max_age):
            return VerificationResult.EXPIRED
            
        # Check expiration if set
        if signature_info.expiration and current_time > signature_info.expiration:
            return VerificationResult.EXPIRED
            
        # Check for replay attacks using nonce
        if not self._validate_nonce(signature_info.nonce, signature_info.timestamp):
            return VerificationResult.INVALID
            
        # Get key
        key_data = self.key_store.get_key(signature_info.key_id)
        if not key_data:
            return VerificationResult.KEY_NOT_FOUND
            
        # Verify signature based on algorithm
        if signature_info.algorithm == SignatureAlgorithm.HMAC_SHA256:
            return self._verify_hmac_sha256(data, signature_info, key_data["key_data"])
        elif signature_info.algorithm == SignatureAlgorithm.HMAC_SHA512:
            return self._verify_hmac_sha512(data, signature_info, key_data["key_data"])
        elif signature_info.algorithm == SignatureAlgorithm.ED25519:
            return self._verify_ed25519(data, signature_info, key_data["key_data"])
        elif signature_info.algorithm == SignatureAlgorithm.ECDSA_P256:
            return self._verify_ecdsa_p256(data, signature_info, key_data["key_data"])
        else:
            return VerificationResult.ALGORITHM_UNSUPPORTED
    
    def _verify_hmac_sha256(self, data: bytes, signature_info: SignatureInfo, key: bytes) -> VerificationResult:
        """Verify HMAC-SHA256 signature."""
        try:
            # Create message to sign (data + metadata)
            message = data + f"{signature_info.key_id}{signature_info.timestamp}{signature_info.nonce}".encode()
            
            # Calculate expected signature
            expected_signature = hmac.new(key, message, hashlib.sha256).digest()
            
            # Constant-time comparison to prevent timing attacks
            if hmac.compare_digest(expected_signature, signature_info.signature):
                return VerificationResult.VALID
            else:
                return VerificationResult.INVALID
                
        except Exception:
            return VerificationResult.MALFORMED
    
    def _verify_hmac_sha512(self, data: bytes, signature_info: SignatureInfo, key: bytes) -> VerificationResult:
        """Verify HMAC-SHA512 signature."""
        try:
            # Create message to sign (data + metadata)
            message = data + f"{signature_info.key_id}{signature_info.timestamp}{signature_info.nonce}".encode()
            
            # Calculate expected signature
            expected_signature = hmac.new(key, message, hashlib.sha512).digest()
            
            # Constant-time comparison to prevent timing attacks
            if hmac.compare_digest(expected_signature, signature_info.signature):
                return VerificationResult.VALID
            else:
                return VerificationResult.INVALID
                
        except Exception:
            return VerificationResult.MALFORMED
    
    def _verify_ed25519(self, data: bytes, signature_info: SignatureInfo, key: bytes) -> VerificationResult:
        """Verify Ed25519 signature."""
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
            
            # Create message to verify (data + metadata)
            message = data + f"{signature_info.key_id}{signature_info.timestamp}{signature_info.nonce}".encode()
            
            # Load public key
            public_key = Ed25519PublicKey.from_public_bytes(key)
            
            # Verify signature
            try:
                public_key.verify(signature_info.signature, message)
                return VerificationResult.VALID
            except Exception:
                return VerificationResult.INVALID
            
        except ImportError:
            # If cryptography library is not available
            return VerificationResult.ALGORITHM_UNSUPPORTED
        except Exception:
            return VerificationResult.MALFORMED
    
    def _verify_ecdsa_p256(self, data: bytes, signature_info: SignatureInfo, key: bytes) -> VerificationResult:
        """Verify ECDSA P-256 signature."""
        try:
            from cryptography.hazmat.primitives.asymmetric import ec
            from cryptography.hazmat.primitives import hashes
            from cryptography.exceptions import InvalidSignature
            
            # Create message to verify (data + metadata)
            message = data + f"{signature_info.key_id}{signature_info.timestamp}{signature_info.nonce}".encode()
            
            try:
                # Load public key
                public_key = ec.EllipticCurvePublicKey.from_encoded_point(
                    ec.SECP256R1(),
                    key
                )
                
                # Verify signature
                try:
                    public_key.verify(
                        signature_info.signature,
                        message,
                        ec.ECDSA(hashes.SHA256())
                    )
                    return VerificationResult.VALID
                except InvalidSignature:
                    return VerificationResult.INVALID
                    
            except ValueError:
                # Invalid key format
                return VerificationResult.KEY_NOT_FOUND
                
        except ImportError:
            # If cryptography library is not available
            return VerificationResult.ALGORITHM_UNSUPPORTED
        except Exception:
            return VerificationResult.MALFORMED
    
    def _validate_nonce(self, nonce: str, timestamp: float) -> bool:
        """Validate nonce to prevent replay attacks."""
        # Check if nonce was already used
        if nonce in self.nonce_cache:
            return False
            
        # Add nonce to cache with timestamp
        self.nonce_cache[nonce] = timestamp
        
        # Clean up old nonces periodically
        if len(self.nonce_cache) > self.nonce_cleanup_threshold:
            self._cleanup_nonces()
            
        return True
    
    def _cleanup_nonces(self):
        """Remove old nonces from cache."""
        current_time = time.time()
        expired_time = current_time - 3600  # 1 hour expiration
        
        # Remove expired nonces
        self.nonce_cache = {
            nonce: ts for nonce, ts in self.nonce_cache.items()
            if ts > expired_time
        }
    
    def _record_verification_attempt(self, signature_info: SignatureInfo, data: bytes):
        """Record verification attempt for auditing."""
        self.verification_results.append({
            "timestamp": time.time(),
            "key_id": signature_info.key_id,
            "algorithm": signature_info.algorithm.value,
            "data_hash": hashlib.sha256(data).hexdigest()[:16],
            "signature_timestamp": signature_info.timestamp
        })
        
        # Limit history size
        if len(self.verification_results) > 1000:
            self.verification_results = self.verification_results[-1000:]
    
    def generate_signature_info(self, data: bytes, key_id: str, 
                               algorithm: SignatureAlgorithm = SignatureAlgorithm.HMAC_SHA256,
                               expiration: Optional[float] = None) -> Optional[SignatureInfo]:
        """
        Generate signature info for data.
        
        Args:
            data: Data to sign
            key_id: Key ID to use for signing
            algorithm: Signature algorithm
            expiration: Optional expiration timestamp
            
        Returns:
            SignatureInfo or None if key not found
        """
        # Get key
        key_data = self.key_store.get_key(key_id)
        if not key_data:
            return None
            
        # Generate nonce and timestamp
        nonce = secrets.token_hex(16)
        timestamp = time.time()
        
        # Create message to sign
        message = data + f"{key_id}{timestamp}{nonce}".encode()
        
        # Generate signature based on algorithm
        if algorithm == SignatureAlgorithm.HMAC_SHA256:
            signature = hmac.new(key_data["key_data"], message, hashlib.sha256).digest()
        elif algorithm == SignatureAlgorithm.HMAC_SHA512:
            signature = hmac.new(key_data["key_data"], message, hashlib.sha512).digest()
        elif algorithm == SignatureAlgorithm.ED25519:
            try:
                from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
                
                # For Ed25519, we need a private key
                # In a real implementation, we would retrieve the private key from a secure store
                # For this example, we'll derive it from the key_data
                private_key = Ed25519PrivateKey.from_private_bytes(key_data["key_data"][:32])
                signature = private_key.sign(message)
            except ImportError:
                raise ValueError("Ed25519 algorithm requires the cryptography library")
        elif algorithm == SignatureAlgorithm.ECDSA_P256:
            try:
                from cryptography.hazmat.primitives.asymmetric import ec
                from cryptography.hazmat.primitives import hashes
                
                # For ECDSA, we need a private key
                # In a real implementation, we would retrieve the private key from a secure store
                # For this example, we'll derive it from the key_data
                private_key = ec.derive_private_key(
                    int.from_bytes(key_data["key_data"][:32], byteorder='big'),
                    ec.SECP256R1()
                )
                signature = private_key.sign(
                    message,
                    ec.ECDSA(hashes.SHA256())
                )
            except ImportError:
                raise ValueError("ECDSA_P256 algorithm requires the cryptography library")
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        # Create signature info
        return SignatureInfo(
            algorithm=algorithm,
            key_id=key_id,
            timestamp=timestamp,
            nonce=nonce,
            signature=signature,
            expiration=expiration
        )
    
    def get_verification_stats(self) -> Dict[str, Any]:
        """Get verification statistics."""
        if not self.verification_results:
            return {"total": 0}
            
        current_time = time.time()
        recent_results = [r for r in self.verification_results 
                         if current_time - r["timestamp"] < 3600]
        
        return {
            "total": len(self.verification_results),
            "recent_hour": len(recent_results),
            "nonce_cache_size": len(self.nonce_cache),
            "algorithms": {alg.value: 0 for alg in SignatureAlgorithm}
        }


# Helper functions for easy integration
def create_default_verifier() -> SignatureVerifier:
    """Create default signature verifier with basic keys."""
    key_store = KeyStore()
    
    # Add a default key for testing
    default_key = secrets.token_bytes(32)
    key_store.add_key(
        key_id="default",
        key_data=default_key,
        algorithm=SignatureAlgorithm.HMAC_SHA256,
        issuer="self"
    )
    
    return SignatureVerifier(key_store)


def sign_block_data(verifier: SignatureVerifier, block_data: bytes, 
                   key_id: str = "default") -> Dict[str, Any]:
    """Sign block data and return signature metadata."""
    signature_info = verifier.generate_signature_info(
        data=block_data,
        key_id=key_id
    )
    
    if not signature_info:
        raise ValueError(f"Failed to generate signature with key {key_id}")
        
    return signature_info.to_dict()


def verify_block_signature(verifier: SignatureVerifier, block_data: bytes, 
                          signature_metadata: Dict[str, Any]) -> bool:
    """Verify block signature from metadata."""
    try:
        signature_info = SignatureInfo.from_dict(signature_metadata)
        result = verifier.verify_signature(block_data, signature_info)
        return result == VerificationResult.VALID
    except Exception:
        return False