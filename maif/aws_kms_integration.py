"""
AWS KMS Integration for MAIF
============================

Provides integration with AWS Key Management Service (KMS) for secure key management,
signing, and verification operations.
"""

import base64
import json
import hashlib
import time
from typing import Dict, List, Optional, Any, Union, Tuple
import boto3
from botocore.exceptions import ClientError

from .signature_verification import (
    SignatureAlgorithm, SignatureInfo, VerificationResult, KeyStore
)


class KMSKeyStore(KeyStore):
    """Key store that integrates with AWS KMS for secure key management."""
    
    def __init__(self, region_name: str = "us-east-1", profile_name: Optional[str] = None):
        """
        Initialize KMS key store.
        
        Args:
            region_name: AWS region name
            profile_name: AWS profile name (optional)
        """
        super().__init__()
        
        # Initialize AWS session
        session = boto3.Session(profile_name=profile_name, region_name=region_name)
        self.kms_client = session.client('kms')
        
        # Cache for KMS keys
        self.kms_key_cache: Dict[str, Dict[str, Any]] = {}
        
        # Key alias mapping
        self.key_aliases: Dict[str, str] = {}
    
    def add_kms_key(self, key_id: str, key_alias: Optional[str] = None, 
                   algorithm: SignatureAlgorithm = SignatureAlgorithm.HMAC_SHA256):
        """
        Add a KMS key reference.
        
        Args:
            key_id: KMS key ID or ARN
            key_alias: Alias for the key (optional)
            algorithm: Signature algorithm to use with this key
        """
        # Store key reference
        self.keys[key_id] = {
            "key_data": None,  # No local key data for KMS keys
            "algorithm": algorithm,
            "issuer": "aws-kms",
            "added_at": time.time(),
            "kms_key_id": key_id
        }
        
        # Store alias mapping if provided
        if key_alias:
            self.key_aliases[key_alias] = key_id
    
    def get_key_by_alias(self, alias: str) -> Optional[Dict[str, Any]]:
        """Get key by alias."""
        if alias in self.key_aliases:
            return self.get_key(self.key_aliases[alias])
        return None
    
    def list_kms_keys(self) -> List[Dict[str, Any]]:
        """List available KMS keys."""
        try:
            response = self.kms_client.list_keys()
            return response.get('Keys', [])
        except ClientError as e:
            print(f"Error listing KMS keys: {e}")
            return []
    
    def create_kms_key(self, description: str, key_usage: str = 'SIGN_VERIFY',
                      customer_master_key_spec: str = 'ECC_NIST_P256') -> Optional[str]:
        """
        Create a new KMS key.
        
        Args:
            description: Key description
            key_usage: Key usage (ENCRYPT_DECRYPT, SIGN_VERIFY, etc.)
            customer_master_key_spec: Key spec (SYMMETRIC_DEFAULT, RSA_2048, ECC_NIST_P256, etc.)
            
        Returns:
            Key ID if successful, None otherwise
        """
        try:
            response = self.kms_client.create_key(
                Description=description,
                KeyUsage=key_usage,
                CustomerMasterKeySpec=customer_master_key_spec
            )
            
            key_id = response['KeyMetadata']['KeyId']
            
            # Add key to store
            algorithm = SignatureAlgorithm.ECDSA_P256 if customer_master_key_spec == 'ECC_NIST_P256' else SignatureAlgorithm.HMAC_SHA256
            self.add_kms_key(key_id, algorithm=algorithm)
            
            return key_id
        except ClientError as e:
            print(f"Error creating KMS key: {e}")
            return None
    
    def create_key_alias(self, key_id: str, alias_name: str) -> bool:
        """
        Create an alias for a KMS key.
        
        Args:
            key_id: KMS key ID
            alias_name: Alias name (without 'alias/' prefix)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure alias name has 'alias/' prefix
            if not alias_name.startswith('alias/'):
                alias_name = f'alias/{alias_name}'
            
            self.kms_client.create_alias(
                AliasName=alias_name,
                TargetKeyId=key_id
            )
            
            # Store alias mapping
            self.key_aliases[alias_name.replace('alias/', '')] = key_id
            
            return True
        except ClientError as e:
            print(f"Error creating key alias: {e}")
            return False


class KMSSignatureVerifier:
    """Signature verifier that uses AWS KMS for signing and verification."""
    
    def __init__(self, key_store: KMSKeyStore):
        """
        Initialize KMS signature verifier.
        
        Args:
            key_store: KMS key store
        """
        self.key_store = key_store
        self.kms_client = key_store.kms_client
        self.verification_results: List[Dict[str, Any]] = []
    
    def sign_data(self, data: bytes, key_id: str, 
                 message_type: str = 'DIGEST', 
                 signing_algorithm: str = 'ECDSA_SHA_256') -> Optional[bytes]:
        """
        Sign data using KMS.
        
        Args:
            data: Data to sign
            key_id: KMS key ID
            message_type: Message type (RAW or DIGEST)
            signing_algorithm: Signing algorithm
            
        Returns:
            Signature bytes if successful, None otherwise
        """
        try:
            # If message type is DIGEST, hash the data first
            if message_type == 'DIGEST':
                data = hashlib.sha256(data).digest()
            
            response = self.kms_client.sign(
                KeyId=key_id,
                Message=data,
                MessageType=message_type,
                SigningAlgorithm=signing_algorithm
            )
            
            return response['Signature']
        except ClientError as e:
            print(f"Error signing data with KMS: {e}")
            return None
    
    def verify_signature(self, data: bytes, signature: bytes, key_id: str,
                        message_type: str = 'DIGEST',
                        signing_algorithm: str = 'ECDSA_SHA_256') -> bool:
        """
        Verify signature using KMS.
        
        Args:
            data: Data that was signed
            signature: Signature to verify
            key_id: KMS key ID
            message_type: Message type (RAW or DIGEST)
            signing_algorithm: Signing algorithm
            
        Returns:
            True if signature is valid, False otherwise
        """
        try:
            # If message type is DIGEST, hash the data first
            if message_type == 'DIGEST':
                data = hashlib.sha256(data).digest()
            
            response = self.kms_client.verify(
                KeyId=key_id,
                Message=data,
                MessageType=message_type,
                Signature=signature,
                SigningAlgorithm=signing_algorithm
            )
            
            return response['SignatureValid']
        except ClientError as e:
            print(f"Error verifying signature with KMS: {e}")
            return False
    
    def generate_signature_info(self, data: bytes, key_id: str,
                               expiration: Optional[float] = None) -> Optional[SignatureInfo]:
        """
        Generate signature info using KMS.
        
        Args:
            data: Data to sign
            key_id: Key ID
            expiration: Optional expiration timestamp
            
        Returns:
            SignatureInfo or None if signing failed
        """
        # Get key metadata
        key_data = self.key_store.get_key(key_id)
        if not key_data:
            return None
        
        # Determine algorithm and KMS parameters
        algorithm = key_data.get('algorithm', SignatureAlgorithm.ECDSA_P256)
        kms_key_id = key_data.get('kms_key_id', key_id)
        
        # Map algorithm to KMS signing algorithm
        if algorithm == SignatureAlgorithm.ECDSA_P256:
            signing_algorithm = 'ECDSA_SHA_256'
        elif algorithm == SignatureAlgorithm.HMAC_SHA256:
            signing_algorithm = 'HMAC_SHA_256'
        else:
            # Default to ECDSA
            signing_algorithm = 'ECDSA_SHA_256'
        
        # Generate nonce and timestamp
        import secrets
        nonce = secrets.token_hex(16)
        timestamp = time.time()
        
        # Create message to sign
        message = data + f"{key_id}{timestamp}{nonce}".encode()
        
        # Sign with KMS
        signature = self.sign_data(
            message,
            kms_key_id,
            message_type='RAW',
            signing_algorithm=signing_algorithm
        )
        
        if not signature:
            return None
        
        # Create signature info
        return SignatureInfo(
            algorithm=algorithm,
            key_id=key_id,
            timestamp=timestamp,
            nonce=nonce,
            signature=signature,
            expiration=expiration
        )
    
    def verify_signature_info(self, data: bytes, signature_info: SignatureInfo) -> VerificationResult:
        """
        Verify signature info using KMS.
        
        Args:
            data: Data that was signed
            signature_info: Signature info
            
        Returns:
            VerificationResult
        """
        # Record verification attempt
        self._record_verification_attempt(signature_info, data)
        
        # Check signature timestamp
        current_time = time.time()
        if signature_info.expiration and current_time > signature_info.expiration:
            return VerificationResult.EXPIRED
        
        # Get key metadata
        key_data = self.key_store.get_key(signature_info.key_id)
        if not key_data:
            return VerificationResult.KEY_NOT_FOUND
        
        # Get KMS key ID
        kms_key_id = key_data.get('kms_key_id', signature_info.key_id)
        
        # Map algorithm to KMS signing algorithm
        if signature_info.algorithm == SignatureAlgorithm.ECDSA_P256:
            signing_algorithm = 'ECDSA_SHA_256'
        elif signature_info.algorithm == SignatureAlgorithm.HMAC_SHA256:
            signing_algorithm = 'HMAC_SHA_256'
        else:
            # Default to ECDSA
            signing_algorithm = 'ECDSA_SHA_256'
        
        # Create message to verify
        message = data + f"{signature_info.key_id}{signature_info.timestamp}{signature_info.nonce}".encode()
        
        # Verify with KMS
        is_valid = self.verify_signature(
            message,
            signature_info.signature,
            kms_key_id,
            message_type='RAW',
            signing_algorithm=signing_algorithm
        )
        
        return VerificationResult.VALID if is_valid else VerificationResult.INVALID
    
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


# Helper functions for easy integration
def create_kms_verifier(region_name: str = "us-east-1", 
                       profile_name: Optional[str] = None) -> Tuple[KMSKeyStore, KMSSignatureVerifier]:
    """
    Create KMS key store and signature verifier.
    
    Args:
        region_name: AWS region name
        profile_name: AWS profile name (optional)
        
    Returns:
        Tuple of (KMSKeyStore, KMSSignatureVerifier)
    """
    key_store = KMSKeyStore(region_name=region_name, profile_name=profile_name)
    verifier = KMSSignatureVerifier(key_store)
    return key_store, verifier


def sign_block_data_with_kms(verifier: KMSSignatureVerifier, block_data: bytes, 
                            key_id: str) -> Dict[str, Any]:
    """
    Sign block data using KMS and return signature metadata.
    
    Args:
        verifier: KMS signature verifier
        block_data: Block data to sign
        key_id: KMS key ID
        
    Returns:
        Signature metadata dictionary
    """
    signature_info = verifier.generate_signature_info(
        data=block_data,
        key_id=key_id
    )
    
    if not signature_info:
        raise ValueError(f"Failed to generate signature with KMS key {key_id}")
        
    return signature_info.to_dict()


def verify_block_signature_with_kms(verifier: KMSSignatureVerifier, block_data: bytes, 
                                  signature_metadata: Dict[str, Any]) -> bool:
    """
    Verify block signature using KMS.
    
    Args:
        verifier: KMS signature verifier
        block_data: Block data
        signature_metadata: Signature metadata
        
    Returns:
        True if signature is valid, False otherwise
    """
    try:
        from .signature_verification import SignatureInfo
        
        signature_info = SignatureInfo.from_dict(signature_metadata)
        result = verifier.verify_signature_info(block_data, signature_info)
        return result == VerificationResult.VALID
    except Exception as e:
        print(f"Error verifying signature with KMS: {e}")
        return False