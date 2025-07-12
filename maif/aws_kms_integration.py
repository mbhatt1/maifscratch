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
import logging
import datetime
import secrets
from typing import Dict, List, Optional, Any, Union, Tuple, Set
import boto3
from botocore.exceptions import ClientError, ConnectionError, EndpointConnectionError, Throttling

from .signature_verification import (
    SignatureAlgorithm, SignatureInfo, VerificationResult, KeyStore
)


# Configure logger
logger = logging.getLogger(__name__)


class KMSError(Exception):
    """Base exception for KMS integration errors."""
    pass


class KMSConnectionError(KMSError):
    """Exception for KMS connection errors."""
    pass


class KMSThrottlingError(KMSError):
    """Exception for KMS throttling errors."""
    pass


class KMSValidationError(KMSError):
    """Exception for KMS validation errors."""
    pass


class KMSPermissionError(KMSError):
    """Exception for KMS permission errors."""
    pass


class KMSKeyStore(KeyStore):
    """Key store that integrates with AWS KMS for secure key management."""
    
    # Set of known transient errors that can be retried
    RETRYABLE_ERRORS = {
        'ThrottlingException', 
        'Throttling', 
        'RequestLimitExceeded',
        'TooManyRequestsException',
        'ServiceUnavailable',
        'InternalServerError',
        'InternalFailure',
        'ServiceFailure'
    }
    
    def __init__(self, region_name: str = "us-east-1", profile_name: Optional[str] = None,
                max_retries: int = 3, base_delay: float = 0.5, max_delay: float = 5.0,
                key_cache_ttl: int = 3600):
        """
        Initialize KMS key store.
        
        Args:
            region_name: AWS region name
            profile_name: AWS profile name (optional)
            max_retries: Maximum number of retries for transient errors
            base_delay: Base delay for exponential backoff (seconds)
            max_delay: Maximum delay for exponential backoff (seconds)
            key_cache_ttl: Time-to-live for key cache entries (seconds)
            
        Raises:
            KMSConnectionError: If unable to initialize the KMS client
        """
        super().__init__()
        
        # Validate inputs
        if not region_name:
            raise KMSValidationError("region_name cannot be empty")
        
        if max_retries < 0:
            raise KMSValidationError("max_retries must be non-negative")
        
        if base_delay <= 0 or max_delay <= 0:
            raise KMSValidationError("base_delay and max_delay must be positive")
        
        if key_cache_ttl <= 0:
            raise KMSValidationError("key_cache_ttl must be positive")
        
        # Initialize AWS session
        try:
            logger.info(f"Initializing KMS client in region {region_name}")
            session = boto3.Session(profile_name=profile_name, region_name=region_name)
            self.kms_client = session.client('kms')
            
            # Retry configuration
            self.max_retries = max_retries
            self.base_delay = base_delay
            self.max_delay = max_delay
            
            # Cache for KMS keys with TTL
            self.kms_key_cache: Dict[str, Dict[str, Any]] = {}
            self.key_cache_ttl = key_cache_ttl
            self.key_cache_expiry: Dict[str, float] = {}
            
            # Key alias mapping
            self.key_aliases: Dict[str, str] = {}
            
            # Metrics for monitoring
            self.metrics = {
                "total_operations": 0,
                "successful_operations": 0,
                "failed_operations": 0,
                "retried_operations": 0,
                "operation_latencies": [],
                "key_rotations": 0,
                "cache_hits": 0,
                "cache_misses": 0
            }
            
            logger.info("KMS key store initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize KMS client: {e}", exc_info=True)
            raise KMSConnectionError(f"Failed to initialize KMS client: {e}")
    
    def _execute_with_retry(self, operation_name: str, operation_func, *args, **kwargs):
        """
        Execute an operation with exponential backoff retry for transient errors.
        
        Args:
            operation_name: Name of the operation for logging
            operation_func: Function to execute
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Result of the operation
            
        Raises:
            KMSError: For non-retryable errors or if max retries exceeded
        """
        start_time = time.time()
        self.metrics["total_operations"] += 1
        
        retries = 0
        last_exception = None
        
        while retries <= self.max_retries:
            try:
                logger.debug(f"Executing KMS {operation_name} (attempt {retries + 1}/{self.max_retries + 1})")
                result = operation_func(*args, **kwargs)
                
                # Record success metrics
                self.metrics["successful_operations"] += 1
                latency = time.time() - start_time
                self.metrics["operation_latencies"].append(latency)
                
                # Trim latencies list if it gets too large
                if len(self.metrics["operation_latencies"]) > 1000:
                    self.metrics["operation_latencies"] = self.metrics["operation_latencies"][-1000:]
                
                logger.debug(f"KMS {operation_name} completed successfully in {latency:.2f}s")
                return result
                
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')
                error_message = e.response.get('Error', {}).get('Message', '')
                
                # Check if this is a retryable error
                if error_code in self.RETRYABLE_ERRORS and retries < self.max_retries:
                    retries += 1
                    self.metrics["retried_operations"] += 1
                    delay = min(self.max_delay, self.base_delay * (2 ** retries))
                    
                    logger.warning(
                        f"KMS {operation_name} failed with retryable error: {error_code} - {error_message}. "
                        f"Retrying in {delay:.2f}s (attempt {retries}/{self.max_retries})"
                    )
                    
                    time.sleep(delay)
                    last_exception = e
                else:
                    # Non-retryable error or max retries exceeded
                    self.metrics["failed_operations"] += 1
                    
                    if error_code in self.RETRYABLE_ERRORS:
                        logger.error(
                            f"KMS {operation_name} failed after {retries} retries: {error_code} - {error_message}",
                            exc_info=True
                        )
                        raise KMSThrottlingError(f"{error_code}: {error_message}. Max retries exceeded.")
                    elif error_code in ('AccessDeniedException', 'AccessDenied'):
                        logger.error(
                            f"KMS {operation_name} failed due to permission error: {error_code} - {error_message}",
                            exc_info=True
                        )
                        raise KMSPermissionError(f"{error_code}: {error_message}")
                    else:
                        logger.error(
                            f"KMS {operation_name} failed with non-retryable error: {error_code} - {error_message}",
                            exc_info=True
                        )
                        raise KMSError(f"{error_code}: {error_message}")
                        
            except (ConnectionError, EndpointConnectionError) as e:
                # Network-related errors
                if retries < self.max_retries:
                    retries += 1
                    self.metrics["retried_operations"] += 1
                    delay = min(self.max_delay, self.base_delay * (2 ** retries))
                    
                    logger.warning(
                        f"KMS {operation_name} failed with connection error: {str(e)}. "
                        f"Retrying in {delay:.2f}s (attempt {retries}/{self.max_retries})"
                    )
                    
                    time.sleep(delay)
                    last_exception = e
                else:
                    self.metrics["failed_operations"] += 1
                    logger.error(f"KMS {operation_name} failed after {retries} retries due to connection error", exc_info=True)
                    raise KMSConnectionError(f"Connection error: {str(e)}. Max retries exceeded.")
            
            except Exception as e:
                # Unexpected errors
                self.metrics["failed_operations"] += 1
                logger.error(f"Unexpected error in KMS {operation_name}: {str(e)}", exc_info=True)
                raise KMSError(f"Unexpected error: {str(e)}")
        
        # This should not be reached, but just in case
        if last_exception:
            raise KMSError(f"Operation failed after {retries} retries: {str(last_exception)}")
        else:
            raise KMSError(f"Operation failed after {retries} retries")
    
    def _clean_key_cache(self):
        """Clean expired entries from key cache."""
        current_time = time.time()
        expired_keys = [
            key_id for key_id, expiry in self.key_cache_expiry.items()
            if current_time > expiry
        ]
        
        for key_id in expired_keys:
            if key_id in self.kms_key_cache:
                del self.kms_key_cache[key_id]
            if key_id in self.key_cache_expiry:
                del self.key_cache_expiry[key_id]
        
        if expired_keys:
            logger.debug(f"Cleaned key cache: {len(expired_keys)} expired entries removed")
    
    def _validate_key_permissions(self, key_id: str, required_operations: List[str]) -> bool:
        """
        Validate that the key has the required permissions.
        
        Args:
            key_id: KMS key ID
            required_operations: List of required KMS operations
            
        Returns:
            True if the key has the required permissions, False otherwise
        """
        try:
            def get_key_policy_operation():
                response = self.kms_client.get_key_policy(
                    KeyId=key_id,
                    PolicyName='default'
                )
                return response.get('Policy', '')
            
            policy_json = self._execute_with_retry("get_key_policy", get_key_policy_operation)
            
            # Parse policy
            if isinstance(policy_json, str):
                policy = json.loads(policy_json)
            else:
                policy = policy_json
            
            # Check if policy allows required operations
            # This is a simplified check - in a real implementation, you would need to
            # check the specific IAM principal and conditions
            statements = policy.get('Statement', [])
            allowed_actions = set()
            
            for statement in statements:
                effect = statement.get('Effect', '')
                if effect == 'Allow':
                    actions = statement.get('Action', [])
                    if isinstance(actions, str):
                        actions = [actions]
                    allowed_actions.update(actions)
            
            # Check if all required operations are allowed
            for operation in required_operations:
                kms_action = f"kms:{operation}"
                if kms_action not in allowed_actions and "kms:*" not in allowed_actions:
                    logger.warning(f"Key {key_id} does not have permission for {kms_action}")
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Failed to validate key permissions for {key_id}: {e}")
            return False
    
    def add_kms_key(self, key_id: str, key_alias: Optional[str] = None, 
                   algorithm: SignatureAlgorithm = SignatureAlgorithm.HMAC_SHA256,
                   key_type: str = "customer"):
        """
        Add a KMS key reference.
        
        Args:
            key_id: KMS key ID or ARN
            key_alias: Alias for the key (optional)
            algorithm: Signature algorithm to use with this key
            key_type: Type of key ("customer" for customer managed, "aws" for AWS managed)
            
        Raises:
            KMSValidationError: If key_id is empty
        """
        if not key_id:
            raise KMSValidationError("key_id cannot be empty")
        
        logger.info(f"Adding KMS key {key_id} with alias {key_alias}")
        
        # Store key reference
        self.keys[key_id] = {
            "key_data": None,  # No local key data for KMS keys
            "algorithm": algorithm,
            "issuer": "aws-kms",
            "added_at": time.time(),
            "kms_key_id": key_id,
            "key_type": key_type
        }
        
        # Store alias mapping if provided
        if key_alias:
            self.key_aliases[key_alias] = key_id
            logger.debug(f"Added alias {key_alias} for key {key_id}")
    
    def get_key_by_alias(self, alias: str) -> Optional[Dict[str, Any]]:
        """
        Get key by alias.
        
        Args:
            alias: Key alias
            
        Returns:
            Key data dictionary if found, None otherwise
        """
        if alias in self.key_aliases:
            key_id = self.key_aliases[alias]
            logger.debug(f"Found key {key_id} for alias {alias}")
            return self.get_key(key_id)
        
        logger.debug(f"No key found for alias {alias}")
        return None
    
    def list_kms_keys(self) -> List[Dict[str, Any]]:
        """
        List available KMS keys.
        
        Returns:
            List of key information dictionaries
            
        Raises:
            KMSError: If an error occurs while listing keys
        """
        logger.info("Listing available KMS keys")
        
        def list_keys_operation():
            response = self.kms_client.list_keys()
            return response.get('Keys', [])
        
        keys = self._execute_with_retry("list_keys", list_keys_operation)
        logger.info(f"Found {len(keys)} KMS keys")
        return keys
    
    def get_key_metadata(self, key_id: str) -> Dict[str, Any]:
        """
        Get metadata for a KMS key.
        
        Args:
            key_id: KMS key ID
            
        Returns:
            Key metadata dictionary
            
        Raises:
            KMSError: If an error occurs while getting key metadata
        """
        # Check cache first
        self._clean_key_cache()
        
        if key_id in self.kms_key_cache:
            self.metrics["cache_hits"] += 1
            logger.debug(f"Key cache hit for {key_id}")
            return self.kms_key_cache[key_id]
        
        self.metrics["cache_misses"] += 1
        logger.debug(f"Key cache miss for {key_id}")
        
        logger.info(f"Getting metadata for KMS key {key_id}")
        
        def describe_key_operation():
            response = self.kms_client.describe_key(KeyId=key_id)
            return response.get('KeyMetadata', {})
        
        metadata = self._execute_with_retry("describe_key", describe_key_operation)
        
        # Cache metadata with TTL
        self.kms_key_cache[key_id] = metadata
        self.key_cache_expiry[key_id] = time.time() + self.key_cache_ttl
        
        logger.debug(f"Cached metadata for key {key_id}")
        return metadata
    
    def create_kms_key(self, description: str, key_usage: str = 'SIGN_VERIFY',
                      customer_master_key_spec: str = 'ECC_NIST_P256',
                      tags: Optional[List[Dict[str, str]]] = None) -> Optional[str]:
        """
        Create a new KMS key.
        
        Args:
            description: Key description
            key_usage: Key usage (ENCRYPT_DECRYPT, SIGN_VERIFY, etc.)
            customer_master_key_spec: Key spec (SYMMETRIC_DEFAULT, RSA_2048, ECC_NIST_P256, etc.)
            tags: Optional list of tags to attach to the key
            
        Returns:
            Key ID if successful, None otherwise
            
        Raises:
            KMSValidationError: If input validation fails
            KMSError: If an error occurs during key creation
        """
        # Validate inputs
        if not description:
            raise KMSValidationError("description cannot be empty")
        
        valid_key_usages = ['ENCRYPT_DECRYPT', 'SIGN_VERIFY', 'GENERATE_VERIFY_MAC']
        if key_usage not in valid_key_usages:
            raise KMSValidationError(f"key_usage must be one of {valid_key_usages}")
        
        valid_key_specs = [
            'SYMMETRIC_DEFAULT', 'RSA_2048', 'RSA_3072', 'RSA_4096',
            'ECC_NIST_P256', 'ECC_NIST_P384', 'ECC_NIST_P521', 'ECC_SECG_P256K1',
            'HMAC_224', 'HMAC_256', 'HMAC_384', 'HMAC_512'
        ]
        if customer_master_key_spec not in valid_key_specs:
            raise KMSValidationError(f"customer_master_key_spec must be one of {valid_key_specs}")
        
        logger.info(f"Creating new KMS key with spec {customer_master_key_spec} for {key_usage}")
        
        # Prepare create key parameters
        create_key_params = {
            'Description': description,
            'KeyUsage': key_usage,
            'CustomerMasterKeySpec': customer_master_key_spec,
            'Origin': 'AWS_KMS'
        }
        
        # Add tags if provided
        if tags:
            create_key_params['Tags'] = tags
        
        def create_key_operation():
            response = self.kms_client.create_key(**create_key_params)
            return response.get('KeyMetadata', {})
        
        try:
            key_metadata = self._execute_with_retry("create_key", create_key_operation)
            key_id = key_metadata.get('KeyId')
            
            if not key_id:
                logger.error("Failed to get KeyId from create_key response")
                return None
            
            # Add key to store
            algorithm = SignatureAlgorithm.ECDSA_P256 if customer_master_key_spec == 'ECC_NIST_P256' else SignatureAlgorithm.HMAC_SHA256
            self.add_kms_key(key_id, algorithm=algorithm, key_type="customer")
            
            # Cache key metadata
            self.kms_key_cache[key_id] = key_metadata
            self.key_cache_expiry[key_id] = time.time() + self.key_cache_ttl
            
            logger.info(f"Successfully created KMS key {key_id}")
            return key_id
            
        except KMSError as e:
            logger.error(f"Failed to create KMS key: {e}")
            raise
    
    def create_key_alias(self, key_id: str, alias_name: str) -> bool:
        """
        Create an alias for a KMS key.
        
        Args:
            key_id: KMS key ID
            alias_name: Alias name (without 'alias/' prefix)
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            KMSValidationError: If input validation fails
            KMSError: If an error occurs during alias creation
        """
        # Validate inputs
        if not key_id:
            raise KMSValidationError("key_id cannot be empty")
        
        if not alias_name:
            raise KMSValidationError("alias_name cannot be empty")
        
        # Ensure alias name has 'alias/' prefix
        if not alias_name.startswith('alias/'):
            alias_name = f'alias/{alias_name}'
        
        logger.info(f"Creating alias {alias_name} for key {key_id}")
        
        def create_alias_operation():
            self.kms_client.create_alias(
                AliasName=alias_name,
                TargetKeyId=key_id
            )
            return True
        
        try:
            self._execute_with_retry("create_alias", create_alias_operation)
            
            # Store alias mapping
            clean_alias = alias_name.replace('alias/', '')
            self.key_aliases[clean_alias] = key_id
            
            logger.info(f"Successfully created alias {alias_name} for key {key_id}")
            return True
            
        except KMSError as e:
            logger.error(f"Failed to create alias {alias_name} for key {key_id}: {e}")
            raise
    
    def rotate_key(self, key_id: str, create_new_alias: bool = True) -> Optional[str]:
        """
        Rotate a KMS key by creating a new key and updating aliases.
        
        Args:
            key_id: KMS key ID to rotate
            create_new_alias: Whether to create a new alias for the new key
            
        Returns:
            New key ID if successful, None otherwise
            
        Raises:
            KMSValidationError: If input validation fails
            KMSError: If an error occurs during key rotation
        """
        # Validate inputs
        if not key_id:
            raise KMSValidationError("key_id cannot be empty")
        
        logger.info(f"Rotating KMS key {key_id}")
        
        try:
            # Get key metadata
            key_metadata = self.get_key_metadata(key_id)
            
            # Get key aliases
            def list_aliases_operation():
                response = self.kms_client.list_aliases(KeyId=key_id)
                return response.get('Aliases', [])
            
            aliases = self._execute_with_retry("list_aliases", list_aliases_operation)
            
            # Create new key with same specs
            description = f"Rotated from {key_id} on {datetime.datetime.now().isoformat()}"
            key_usage = key_metadata.get('KeyUsage', 'SIGN_VERIFY')
            key_spec = key_metadata.get('CustomerMasterKeySpec', 'ECC_NIST_P256')
            
            new_key_id = self.create_kms_key(
                description=description,
                key_usage=key_usage,
                customer_master_key_spec=key_spec
            )
            
            if not new_key_id:
                logger.error("Failed to create new key for rotation")
                return None
            
            # Update aliases if requested
            if create_new_alias and aliases:
                for alias in aliases:
                    alias_name = alias.get('AliasName')
                    if alias_name:
                        # Delete existing alias
                        def delete_alias_operation():
                            self.kms_client.delete_alias(AliasName=alias_name)
                            return True
                        
                        self._execute_with_retry("delete_alias", delete_alias_operation)
                        
                        # Create new alias pointing to new key
                        self.create_key_alias(new_key_id, alias_name)
            
            # Update metrics
            self.metrics["key_rotations"] += 1
            
            logger.info(f"Successfully rotated key {key_id} to {new_key_id}")
            return new_key_id
            
        except KMSError as e:
            logger.error(f"Failed to rotate key {key_id}: {e}")
            raise


class KMSSignatureVerifier:
    """Signature verifier that uses AWS KMS for signing and verification."""
    
    def __init__(self, key_store: KMSKeyStore):
        """
        Initialize KMS signature verifier.
        
        Args:
            key_store: KMS key store
            
        Raises:
            KMSValidationError: If key_store is None
        """
        if not key_store:
            raise KMSValidationError("key_store cannot be None")
        
        logger.info("Initializing KMS signature verifier")
        
        self.key_store = key_store
        self.kms_client = key_store.kms_client
        self.verification_results: List[Dict[str, Any]] = []
        
        # Metrics
        self.metrics = {
            "total_signatures": 0,
            "successful_signatures": 0,
            "failed_signatures": 0,
            "total_verifications": 0,
            "successful_verifications": 0,
            "failed_verifications": 0,
            "expired_signatures": 0
        }
        
        logger.info("KMS signature verifier initialized successfully")
    
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
            
        Raises:
            KMSValidationError: If input validation fails
            KMSError: If an error occurs during signing
        """
        # Validate inputs
        if not data:
            raise KMSValidationError("data cannot be empty")
        
        if not key_id:
            raise KMSValidationError("key_id cannot be empty")
        
        valid_message_types = ['RAW', 'DIGEST']
        if message_type not in valid_message_types:
            raise KMSValidationError(f"message_type must be one of {valid_message_types}")
        
        valid_signing_algorithms = [
            'RSASSA_PSS_SHA_256', 'RSASSA_PSS_SHA_384', 'RSASSA_PSS_SHA_512',
            'RSASSA_PKCS1_V1_5_SHA_256', 'RSASSA_PKCS1_V1_5_SHA_384', 'RSASSA_PKCS1_V1_5_SHA_512',
            'ECDSA_SHA_256', 'ECDSA_SHA_384', 'ECDSA_SHA_512',
            'HMAC_SHA_224', 'HMAC_SHA_256', 'HMAC_SHA_384', 'HMAC_SHA_512'
        ]
        if signing_algorithm not in valid_signing_algorithms:
            raise KMSValidationError(f"signing_algorithm must be one of {valid_signing_algorithms}")
        
        # Validate key permissions
        if not self.key_store._validate_key_permissions(key_id, ['Sign']):
            raise KMSPermissionError(f"Key {key_id} does not have permission for Sign operation")
        
        logger.info(f"Signing data with KMS key {key_id} using {signing_algorithm}")
        self.metrics["total_signatures"] += 1
        
        try:
            # If message type is DIGEST, hash the data first
            if message_type == 'DIGEST':
                data = hashlib.sha256(data).digest()
            
            def sign_operation():
                response = self.kms_client.sign(
                    KeyId=key_id,
                    Message=data,
                    MessageType=message_type,
                    SigningAlgorithm=signing_algorithm
                )
                return response.get('Signature')
            
            signature = self.key_store._execute_with_retry("sign", sign_operation)
            
            if not signature:
                logger.error(f"Failed to get signature from KMS for key {key_id}")
                self.metrics["failed_signatures"] += 1
                return None
            
            self.metrics["successful_signatures"] += 1
            logger.info(f"Successfully signed data with KMS key {key_id}")
            return signature
            
        except KMSError as e:
            logger.error(f"Failed to sign data with KMS key {key_id}: {e}")
            self.metrics["failed_signatures"] += 1
            raise
    
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
            
        Raises:
            KMSValidationError: If input validation fails
            KMSError: If an error occurs during verification
        """
        # Validate inputs
        if not data:
            raise KMSValidationError("data cannot be empty")
        
        if not signature:
            raise KMSValidationError("signature cannot be empty")
        
        if not key_id:
            raise KMSValidationError("key_id cannot be empty")
        
        valid_message_types = ['RAW', 'DIGEST']
        if message_type not in valid_message_types:
            raise KMSValidationError(f"message_type must be one of {valid_message_types}")
        
        valid_signing_algorithms = [
            'RSASSA_PSS_SHA_256', 'RSASSA_PSS_SHA_384', 'RSASSA_PSS_SHA_512',
            'RSASSA_PKCS1_V1_5_SHA_256', 'RSASSA_PKCS1_V1_5_SHA_384', 'RSASSA_PKCS1_V1_5_SHA_512',
            'ECDSA_SHA_256', 'ECDSA_SHA_384', 'ECDSA_SHA_512',
            'HMAC_SHA_224', 'HMAC_SHA_256', 'HMAC_SHA_384', 'HMAC_SHA_512'
        ]
        if signing_algorithm not in valid_signing_algorithms:
            raise KMSValidationError(f"signing_algorithm must be one of {valid_signing_algorithms}")
        
        # Validate key permissions
        if not self.key_store._validate_key_permissions(key_id, ['Verify']):
            raise KMSPermissionError(f"Key {key_id} does not have permission for Verify operation")
        
        logger.info(f"Verifying signature with KMS key {key_id} using {signing_algorithm}")
        self.metrics["total_verifications"] += 1
        
        try:
            # If message type is DIGEST, hash the data first
            if message_type == 'DIGEST':
                data = hashlib.sha256(data).digest()
            
            def verify_operation():
                response = self.kms_client.verify(
                    KeyId=key_id,
                    Message=data,
                    MessageType=message_type,
                    Signature=signature,
                    SigningAlgorithm=signing_algorithm
                )
                return response.get('SignatureValid', False)
            
            is_valid = self.key_store._execute_with_retry("verify", verify_operation)
            
            if is_valid:
                self.metrics["successful_verifications"] += 1
                logger.info(f"Signature verified successfully with KMS key {key_id}")
            else:
                self.metrics["failed_verifications"] += 1
                logger.warning(f"Signature verification failed with KMS key {key_id}")
            
            return is_valid
            
        except KMSError as e:
            logger.error(f"Failed to verify signature with KMS key {key_id}: {e}")
            self.metrics["failed_verifications"] += 1
            raise
    
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
            
        Raises:
            KMSValidationError: If input validation fails
            KMSError: If an error occurs during signature generation
        """
        # Validate inputs
        if not data:
            raise KMSValidationError("data cannot be empty")
        
        if not key_id:
            raise KMSValidationError("key_id cannot be empty")
        
        logger.info(f"Generating signature info for key {key_id}")
        
        # Get key metadata
        key_data = self.key_store.get_key(key_id)
        if not key_data:
            logger.error(f"Key {key_id} not found in key store")
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
        nonce = secrets.token_hex(16)
        timestamp = time.time()
        
        # Create message to sign
        message = data + f"{key_id}{timestamp}{nonce}".encode()
        
        try:
            # Sign with KMS
            signature = self.sign_data(
                message,
                kms_key_id,
                message_type='RAW',
                signing_algorithm=signing_algorithm
            )
            
            if not signature:
                logger.error(f"Failed to generate signature with KMS key {key_id}")
                return None
            
            # Create signature info
            signature_info = SignatureInfo(
                algorithm=algorithm,
                key_id=key_id,
                timestamp=timestamp,
                nonce=nonce,
                signature=signature,
                expiration=expiration
            )
            
            logger.info(f"Successfully generated signature info for key {key_id}")
            return signature_info
            
        except KMSError as e:
            logger.error(f"Failed to generate signature info: {e}")
            raise
    
    def verify_signature_info(self, data: bytes, signature_info: SignatureInfo) -> VerificationResult:
        """
        Verify signature info using KMS.
        
        Args:
            data: Data that was signed
            signature_info: Signature info
            
        Returns:
            VerificationResult
            
        Raises:
            KMSValidationError: If input validation fails
            KMSError: If an error occurs during verification
        """
        # Validate inputs
        if not data:
            raise KMSValidationError("data cannot be empty")
        
        if not signature_info:
            raise KMSValidationError("signature_info cannot be None")
        
        logger.info(f"Verifying signature info for key {signature_info.key_id}")
        
        # Record verification attempt
        self._record_verification_attempt(signature_info, data)
        
        # Check signature timestamp
        current_time = time.time()
        if signature_info.expiration and current_time > signature_info.expiration:
            logger.warning(f"Signature expired for key {signature_info.key_id}")
            self.metrics["expired_signatures"] += 1
            return VerificationResult.EXPIRED
        
        # Get key metadata
        key_data = self.key_store.get_key(signature_info.key_id)
        if not key_data:
            logger.error(f"Key {signature_info.key_id} not found in key store")
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
        
        try:
            # Verify with KMS
            is_valid = self.verify_signature(
                message,
                signature_info.signature,
                kms_key_id,
                message_type='RAW',
                signing_algorithm=signing_algorithm
            )
            
            result = VerificationResult.VALID if is_valid else VerificationResult.INVALID
            logger.info(f"Signature verification result for key {signature_info.key_id}: {result}")
            return result
            
        except KMSError as e:
            logger.error(f"Failed to verify signature info: {e}")
            raise
    
    def _record_verification_attempt(self, signature_info: SignatureInfo, data: bytes):
        """
        Record verification attempt for auditing.
        
        Args:
            signature_info: Signature info
            data: Data that was signed
        """
        verification_record = {
            "timestamp": time.time(),
            "iso_time": datetime.datetime.now().isoformat(),
            "key_id": signature_info.key_id,
            "algorithm": signature_info.algorithm.value,
            "data_hash": hashlib.sha256(data).hexdigest()[:16],
            "signature_timestamp": signature_info.timestamp,
            "signature_age": time.time() - signature_info.timestamp
        }
        
        # Add to history
        self.verification_results.append(verification_record)
        
        # Limit history size
        if len(self.verification_results) > 1000:
            self.verification_results = self.verification_results[-1000:]
        
        logger.debug(f"Recorded verification attempt for key {signature_info.key_id}")


# Helper functions for easy integration
def create_kms_verifier(region_name: str = "us-east-1",
                       profile_name: Optional[str] = None,
                       max_retries: int = 3,
                       key_cache_ttl: int = 3600) -> Tuple[KMSKeyStore, KMSSignatureVerifier]:
    """
    Create KMS key store and signature verifier.
    
    Args:
        region_name: AWS region name
        profile_name: AWS profile name (optional)
        max_retries: Maximum number of retries for transient errors
        key_cache_ttl: Time-to-live for key cache entries (seconds)
        
    Returns:
        Tuple of (KMSKeyStore, KMSSignatureVerifier)
        
    Raises:
        KMSConnectionError: If unable to initialize the KMS client
    """
    logger.info(f"Creating KMS verifier in region {region_name}")
    
    try:
        key_store = KMSKeyStore(
            region_name=region_name,
            profile_name=profile_name,
            max_retries=max_retries,
            key_cache_ttl=key_cache_ttl
        )
        
        verifier = KMSSignatureVerifier(key_store)
        
        logger.info("KMS verifier created successfully")
        return key_store, verifier
        
    except KMSError as e:
        logger.error(f"Failed to create KMS verifier: {e}", exc_info=True)
        raise


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
        
    Raises:
        KMSValidationError: If input validation fails
        KMSError: If an error occurs during signing
    """
    logger.info(f"Signing block data with KMS key {key_id}")
    
    if not verifier:
        raise KMSValidationError("verifier cannot be None")
    
    if not block_data:
        raise KMSValidationError("block_data cannot be empty")
    
    if not key_id:
        raise KMSValidationError("key_id cannot be empty")
    
    try:
        signature_info = verifier.generate_signature_info(
            data=block_data,
            key_id=key_id
        )
        
        if not signature_info:
            raise KMSError(f"Failed to generate signature with KMS key {key_id}")
        
        signature_dict = signature_info.to_dict()
        logger.info(f"Successfully signed block data with KMS key {key_id}")
        return signature_dict
        
    except KMSError as e:
        logger.error(f"Failed to sign block data with KMS key {key_id}: {e}")
        raise


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
        
    Raises:
        KMSValidationError: If input validation fails
        KMSError: If an error occurs during verification
    """
    if not verifier:
        raise KMSValidationError("verifier cannot be None")
    
    if not block_data:
        raise KMSValidationError("block_data cannot be empty")
    
    if not signature_metadata:
        raise KMSValidationError("signature_metadata cannot be empty")
    
    key_id = signature_metadata.get('key_id', 'unknown')
    logger.info(f"Verifying block signature with KMS key {key_id}")
    
    try:
        from .signature_verification import SignatureInfo
        
        signature_info = SignatureInfo.from_dict(signature_metadata)
        result = verifier.verify_signature_info(block_data, signature_info)
        
        is_valid = result == VerificationResult.VALID
        logger.info(f"Block signature verification result for key {key_id}: {result}")
        return is_valid
        
    except Exception as e:
        logger.error(f"Error verifying signature with KMS: {e}", exc_info=True)
        return False