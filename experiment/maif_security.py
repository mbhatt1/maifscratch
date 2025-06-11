"""
MAIF Security Implementation
OAuth + Hardware Attestation + Cryptographic Trust Chain
Addresses the trust bootstrap problem identified in the MAIF paper
"""

import hashlib
import hmac
import time
import json
import base64
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import secrets
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import jwt


class TrustLevel(Enum):
    """Trust levels in the MAIF security model"""
    UNTRUSTED = 0
    BASIC_AUTH = 1
    OAUTH_VERIFIED = 2
    HARDWARE_ATTESTED = 3
    CRYPTOGRAPHICALLY_PROVEN = 4
    MULTI_PARTY_VERIFIED = 5


@dataclass
class HardwareAttestation:
    """Hardware attestation data"""
    
    platform_id: str
    tpm_quote: bytes
    pcr_values: Dict[int, str]  # Platform Configuration Registers
    attestation_signature: bytes
    nonce: str
    timestamp: float
    
    def verify_freshness(self, max_age_seconds: int = 300) -> bool:
        """Verify attestation is fresh (within 5 minutes by default)"""
        return (time.time() - self.timestamp) < max_age_seconds
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "platform_id": self.platform_id,
            "tpm_quote": base64.b64encode(self.tpm_quote).decode(),
            "pcr_values": self.pcr_values,
            "attestation_signature": base64.b64encode(self.attestation_signature).decode(),
            "nonce": self.nonce,
            "timestamp": self.timestamp
        }


@dataclass 
class OAuthToken:
    """OAuth 2.0 token with MAIF extensions"""
    
    access_token: str
    token_type: str
    expires_in: int
    scope: List[str]
    agent_id: str
    trust_level: TrustLevel
    issued_at: float
    
    def is_expired(self) -> bool:
        """Check if token is expired"""
        return time.time() > (self.issued_at + self.expires_in)
    
    def has_scope(self, required_scope: str) -> bool:
        """Check if token has required scope"""
        return required_scope in self.scope


@dataclass
class MAIFSecurityContext:
    """Complete security context for MAIF operations"""
    
    agent_id: str
    oauth_token: Optional[OAuthToken]
    hardware_attestation: Optional[HardwareAttestation]
    trust_level: TrustLevel
    cryptographic_keys: Dict[str, Any]
    permissions: List[str]
    
    def can_write(self) -> bool:
        """Check if context allows write operations"""
        return (
            self.trust_level.value >= TrustLevel.OAUTH_VERIFIED.value and
            "maif:write" in self.permissions
        )
    
    def can_read_sensitive(self) -> bool:
        """Check if context allows reading sensitive data"""
        return (
            self.trust_level.value >= TrustLevel.HARDWARE_ATTESTED.value and
            "maif:read_sensitive" in self.permissions
        )


class MAIFSecurityManager:
    """High-performance security manager for MAIF"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # OAuth configuration
        self.oauth_providers = config.get("oauth_providers", {})
        
        # Hardware attestation roots of trust
        self.trusted_tpm_roots = config.get("trusted_tpm_roots", [])
        self.trusted_sgx_roots = config.get("trusted_sgx_roots", [])
        
        # Cryptographic keys
        self.signing_key = self._load_or_generate_key("signing_key")
        self.encryption_key = self._load_or_generate_key("encryption_key")
        
        # Trust policies
        self.min_trust_level_write = TrustLevel(config.get("min_trust_level_write", 2))
        self.min_trust_level_read_sensitive = TrustLevel(config.get("min_trust_level_read_sensitive", 3))
        
        # Security audit log
        self.audit_log = []
    
    def _load_or_generate_key(self, key_name: str) -> rsa.RSAPrivateKey:
        """Load existing key or generate new one"""
        # In production, load from secure key store (HSM, KMS, etc.)
        key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        print(f"Generated new {key_name}")
        return key
    
    async def bootstrap_trust(
        self,
        agent_id: str,
        oauth_token: str,
        hardware_attestation: Optional[HardwareAttestation] = None
    ) -> MAIFSecurityContext:
        """Bootstrap trust using OAuth + optional hardware attestation"""
        
        # Step 1: Verify OAuth token
        oauth_verified = await self._verify_oauth_token(oauth_token)
        if not oauth_verified:
            raise SecurityError("Invalid OAuth token")
        
        oauth_data = jwt.decode(oauth_token, options={"verify_signature": False})  # Already verified above
        trust_level = TrustLevel.OAUTH_VERIFIED
        
        # Step 2: Verify hardware attestation if provided
        if hardware_attestation:
            hw_verified = await self._verify_hardware_attestation(hardware_attestation)
            if hw_verified:
                trust_level = TrustLevel.HARDWARE_ATTESTED
                self._audit_log("HARDWARE_ATTESTATION_SUCCESS", agent_id, hardware_attestation.platform_id)
            else:
                self._audit_log("HARDWARE_ATTESTATION_FAILED", agent_id, hardware_attestation.platform_id)
        
        # Step 3: Generate agent-specific cryptographic keys
        agent_keys = await self._generate_agent_keys(agent_id, trust_level)
        
        # Step 4: Determine permissions based on trust level and OAuth scopes
        permissions = self._calculate_permissions(oauth_data.get("scope", []), trust_level)
        
        # Step 5: Create security context
        context = MAIFSecurityContext(
            agent_id=agent_id,
            oauth_token=OAuthToken(
                access_token=oauth_token,
                token_type="Bearer",
                expires_in=oauth_data.get("exp", 3600),
                scope=oauth_data.get("scope", []),
                agent_id=agent_id,
                trust_level=trust_level,
                issued_at=time.time()
            ),
            hardware_attestation=hardware_attestation,
            trust_level=trust_level,
            cryptographic_keys=agent_keys,
            permissions=permissions
        )
        
        self._audit_log("TRUST_BOOTSTRAP_SUCCESS", agent_id, trust_level.name)
        return context
    
    async def _verify_oauth_token(self, token: str) -> bool:
        """Verify OAuth token with identity provider"""
        try:
            # In production: Verify with actual OAuth provider
            # For now, simulate verification
            decoded = jwt.decode(token, options={"verify_signature": False})
            
            # Check expiration
            if decoded.get("exp", 0) < time.time():
                return False
            
            # Check issuer is trusted
            issuer = decoded.get("iss", "")
            if issuer not in self.oauth_providers:
                return False
            
            return True
            
        except Exception as e:
            print(f"OAuth verification failed: {e}")
            return False
    
    async def _verify_hardware_attestation(self, attestation: HardwareAttestation) -> bool:
        """Verify hardware attestation using TPM/SGX"""
        
        # Check freshness
        if not attestation.verify_freshness():
            return False
        
        # Verify TPM quote signature
        if not self._verify_tpm_signature(attestation):
            return False
        
        # Verify PCR values against known good values
        if not self._verify_pcr_values(attestation.pcr_values):
            return False
        
        return True
    
    def _verify_tpm_signature(self, attestation: HardwareAttestation) -> bool:
        """Verify TPM attestation signature"""
        # In production: Verify against TPM endorsement key chain
        # For now, simulate verification
        
        # Check platform is in trusted list
        if attestation.platform_id not in [root["platform_id"] for root in self.trusted_tpm_roots]:
            return False
        
        # Simulate signature verification
        expected_data = f"{attestation.platform_id}{attestation.nonce}{attestation.timestamp}"
        return len(attestation.attestation_signature) > 0
    
    def _verify_pcr_values(self, pcr_values: Dict[int, str]) -> bool:
        """Verify Platform Configuration Register values"""
        
        # Critical PCRs for boot integrity
        critical_pcrs = [0, 1, 2, 3, 7]  # BIOS, UEFI, boot loader, etc.
        
        for pcr in critical_pcrs:
            if pcr not in pcr_values:
                return False
            
            # In production: Check against known good values
            # For now, just check they exist and are reasonable length
            if len(pcr_values[pcr]) != 64:  # SHA256 hex length
                return False
        
        return True
    
    async def _generate_agent_keys(self, agent_id: str, trust_level: TrustLevel) -> Dict[str, Any]:
        """Generate cryptographic keys for agent"""
        
        # Generate agent-specific signing key
        agent_signing_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048 if trust_level.value >= 3 else 1024
        )
        
        # Generate symmetric encryption key for data
        symmetric_key = secrets.token_bytes(32)  # 256-bit AES key
        
        # Create key derivation for different purposes
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=agent_id.encode(),
            iterations=100000,
        )
        
        derived_key = kdf.derive(symmetric_key)
        
        return {
            "signing_key": agent_signing_key,
            "signing_key_public": agent_signing_key.public_key(),
            "symmetric_key": symmetric_key,
            "derived_key": derived_key,
            "key_id": hashlib.sha256(agent_id.encode()).hexdigest()[:16]
        }
    
    def _calculate_permissions(self, oauth_scopes: List[str], trust_level: TrustLevel) -> List[str]:
        """Calculate MAIF permissions based on OAuth scopes and trust level"""
        
        permissions = []
        
        # Basic permissions for all authenticated users
        if trust_level.value >= TrustLevel.OAUTH_VERIFIED.value:
            permissions.extend(["maif:read_public", "maif:write_own"])
        
        # Enhanced permissions for hardware-attested agents
        if trust_level.value >= TrustLevel.HARDWARE_ATTESTED.value:
            permissions.extend(["maif:read_sensitive", "maif:write_shared"])
        
        # Administrative permissions for highest trust
        if trust_level.value >= TrustLevel.MULTI_PARTY_VERIFIED.value:
            permissions.extend(["maif:admin", "maif:key_management"])
        
        # OAuth scope mapping
        scope_mapping = {
            "maif:full": ["maif:read_public", "maif:read_sensitive", "maif:write_own", "maif:write_shared"],
            "maif:read": ["maif:read_public"],
            "maif:write": ["maif:write_own"],
            "maif:admin": ["maif:admin", "maif:key_management"]
        }
        
        for scope in oauth_scopes:
            if scope in scope_mapping:
                permissions.extend(scope_mapping[scope])
        
        return list(set(permissions))  # Remove duplicates
    
    def sign_maif_block(self, block_data: bytes, context: MAIFSecurityContext) -> bytes:
        """Sign MAIF block with agent's key"""
        
        if not context.can_write():
            raise SecurityError("Insufficient permissions for signing")
        
        signing_key = context.cryptographic_keys["signing_key"]
        
        # Create signature
        signature = signing_key.sign(
            block_data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        # Create signature metadata
        signature_metadata = {
            "algorithm": "RSA_PSS_SHA256",
            "key_id": context.cryptographic_keys["key_id"],
            "agent_id": context.agent_id,
            "trust_level": context.trust_level.value,
            "timestamp": time.time(),
            "signature": base64.b64encode(signature).decode()
        }
        
        self._audit_log("BLOCK_SIGNED", context.agent_id, context.cryptographic_keys["key_id"])
        
        return json.dumps(signature_metadata).encode()
    
    def verify_maif_block_signature(
        self,
        block_data: bytes,
        signature_data: bytes,
        expected_agent_id: str
    ) -> Tuple[bool, TrustLevel]:
        """Verify MAIF block signature"""
        
        try:
            signature_metadata = json.loads(signature_data.decode())
            
            # Basic checks
            if signature_metadata["agent_id"] != expected_agent_id:
                return False, TrustLevel.UNTRUSTED
            
            # Check signature age
            signature_age = time.time() - signature_metadata["timestamp"]
            if signature_age > 86400:  # 24 hours
                return False, TrustLevel.UNTRUSTED
            
            # Get public key for agent (in production: from key store)
            # For now, simulate key retrieval
            public_key = self._get_agent_public_key(expected_agent_id)
            if not public_key:
                return False, TrustLevel.UNTRUSTED
            
            signature = base64.b64decode(signature_metadata["signature"])
            
            # Verify signature
            public_key.verify(
                signature,
                block_data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            trust_level = TrustLevel(signature_metadata["trust_level"])
            self._audit_log("SIGNATURE_VERIFIED", expected_agent_id, signature_metadata["key_id"])
            
            return True, trust_level
            
        except Exception as e:
            print(f"Signature verification failed: {e}")
            self._audit_log("SIGNATURE_VERIFICATION_FAILED", expected_agent_id, str(e))
            return False, TrustLevel.UNTRUSTED
    
    def _get_agent_public_key(self, agent_id: str):
        """Retrieve agent's public key (mock implementation)"""
        # In production: Query secure key store
        return None
    
    def encrypt_sensitive_data(self, data: bytes, context: MAIFSecurityContext) -> bytes:
        """Encrypt sensitive data with agent's key"""
        
        symmetric_key = context.cryptographic_keys["symmetric_key"]
        
        # Generate random IV
        iv = secrets.token_bytes(16)
        
        # Encrypt data
        cipher = Cipher(algorithms.AES(symmetric_key), modes.CBC(iv))
        encryptor = cipher.encryptor()
        
        # Pad data to block size
        pad_length = 16 - (len(data) % 16)
        padded_data = data + bytes([pad_length] * pad_length)
        
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        
        # Return IV + encrypted data
        return iv + encrypted_data
    
    def decrypt_sensitive_data(self, encrypted_data: bytes, context: MAIFSecurityContext) -> bytes:
        """Decrypt sensitive data with agent's key"""
        
        if not context.can_read_sensitive():
            raise SecurityError("Insufficient permissions for decryption")
        
        symmetric_key = context.cryptographic_keys["symmetric_key"]
        
        # Extract IV and data
        iv = encrypted_data[:16]
        ciphertext = encrypted_data[16:]
        
        # Decrypt
        cipher = Cipher(algorithms.AES(symmetric_key), modes.CBC(iv))
        decryptor = cipher.decryptor()
        
        padded_data = decryptor.update(ciphertext) + decryptor.finalize()
        
        # Remove padding
        pad_length = padded_data[-1]
        data = padded_data[:-pad_length]
        
        return data
    
    def _audit_log(self, event: str, agent_id: str, details: str):
        """Log security events for audit"""
        log_entry = {
            "timestamp": time.time(),
            "event": event,
            "agent_id": agent_id,
            "details": details
        }
        
        self.audit_log.append(log_entry)
        print(f"AUDIT: {event} - {agent_id} - {details}")
        
        # In production: Send to secure audit system
    
    def get_audit_log(self, agent_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get audit log entries"""
        if agent_id:
            return [entry for entry in self.audit_log if entry["agent_id"] == agent_id]
        return self.audit_log.copy()


class SecurityError(Exception):
    """MAIF security-related error"""
    pass


# Example configuration
def create_maif_security_config() -> Dict[str, Any]:
    """Create example security configuration"""
    return {
        "oauth_providers": {
            "https://accounts.google.com": {
                "client_id": "example-client-id",
                "jwks_uri": "https://www.googleapis.com/oauth2/v3/certs"
            },
            "https://login.microsoftonline.com": {
                "client_id": "example-client-id",
                "jwks_uri": "https://login.microsoftonline.com/common/discovery/v2.0/keys"
            }
        },
        "trusted_tpm_roots": [
            {
                "platform_id": "trusted-platform-1",
                "endorsement_key": "...",
                "manufacturer": "Intel"
            }
        ],
        "trusted_sgx_roots": [
            {
                "platform_id": "sgx-platform-1", 
                "root_ca": "...",
                "manufacturer": "Intel"
            }
        ],
        "min_trust_level_write": 2,  # OAuth verified
        "min_trust_level_read_sensitive": 3,  # Hardware attested
    }


# Factory function
def create_maif_security_manager(config: Optional[Dict[str, Any]] = None) -> MAIFSecurityManager:
    """Create MAIF security manager with configuration"""
    if config is None:
        config = create_maif_security_config()
    
    return MAIFSecurityManager(config) 