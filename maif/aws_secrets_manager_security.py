"""
AWS Secrets Manager Integration for MAIF Security
=============================================

Integrates MAIF's security module with AWS Secrets Manager for secure storage and retrieval of sensitive information.
"""

import json
import time
import logging
import base64
import os
import uuid
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, ConnectionError, EndpointConnectionError

# Import centralized credential and config management
from .aws_config import get_aws_config, AWSConfig

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.serialization import load_pem_private_key, load_pem_public_key
from cryptography.hazmat.backends import default_backend

from .security import (
    SecurityManager, MAIFSigner, MAIFVerifier, AccessControlManager,
    ProvenanceEntry
)

# Configure logger
logger = logging.getLogger(__name__)


class SecretsManagerError(Exception):
    """Base exception for Secrets Manager errors."""
    pass


class SecretsManagerConnectionError(SecretsManagerError):
    """Exception for Secrets Manager connection errors."""
    pass


class SecretsManagerThrottlingError(SecretsManagerError):
    """Exception for Secrets Manager throttling errors."""
    pass


class SecretsManagerValidationError(SecretsManagerError):
    """Exception for Secrets Manager validation errors."""
    pass


class SecretsManagerClient:
    """Client for AWS Secrets Manager service with production-ready features."""
    
    # Set of known transient errors that can be retried
    RETRYABLE_ERRORS = {
        'ThrottlingException', 
        'Throttling', 
        'RequestLimitExceeded',
        'TooManyRequestsException',
        'ServiceUnavailable',
        'InternalServerError',
        'InternalFailure',
        'ServiceFailure',
        'LimitExceededException'
    }
    
    def __init__(self, region_name: str = "us-east-1", profile_name: Optional[str] = None,
                max_retries: int = 3, base_delay: float = 0.5, max_delay: float = 5.0):
        """
        Initialize Secrets Manager client.
        
        Args:
            region_name: AWS region name
            profile_name: AWS profile name (optional)
            max_retries: Maximum number of retries for transient errors
            base_delay: Base delay for exponential backoff (seconds)
            max_delay: Maximum delay for exponential backoff (seconds)
            
        Raises:
            SecretsManagerConnectionError: If unable to initialize the Secrets Manager client
        """
        # Validate inputs
        if not region_name:
            raise SecretsManagerValidationError("region_name cannot be empty")
        
        if max_retries < 0:
            raise SecretsManagerValidationError("max_retries must be non-negative")
        
        if base_delay <= 0 or max_delay <= 0:
            raise SecretsManagerValidationError("base_delay and max_delay must be positive")
        
        # Initialize AWS session
        try:
            logger.info(f"Initializing Secrets Manager client in region {region_name}")
            session = boto3.Session(profile_name=profile_name, region_name=region_name)
            
            # Initialize Secrets Manager client
            self.secrets_manager_client = session.client('secretsmanager')
            
            # Retry configuration
            self.max_retries = max_retries
            self.base_delay = base_delay
            self.max_delay = max_delay
            
            # Metrics for monitoring
            self.metrics = {
                "total_operations": 0,
                "successful_operations": 0,
                "failed_operations": 0,
                "retried_operations": 0,
                "operation_latencies": [],
                "throttling_count": 0,
                "secret_operations": {}
            }
            
            logger.info("Secrets Manager client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Secrets Manager client: {e}", exc_info=True)
            raise SecretsManagerConnectionError(f"Failed to initialize Secrets Manager client: {e}")
    
    def _execute_with_retry(self, operation_name: str, operation_func, secret_id: Optional[str] = None, *args, **kwargs):
        """
        Execute an operation with exponential backoff retry for transient errors.
        
        Args:
            operation_name: Name of the operation for logging
            operation_func: Function to execute
            secret_id: ID of the secret (for metrics)
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Result of the operation
            
        Raises:
            SecretsManagerError: For non-retryable errors or if max retries exceeded
        """
        start_time = time.time()
        self.metrics["total_operations"] += 1
        
        # Track secret-specific operations
        if secret_id:
            if secret_id not in self.metrics["secret_operations"]:
                self.metrics["secret_operations"][secret_id] = {
                    "total": 0,
                    "successful": 0,
                    "failed": 0,
                    "retried": 0,
                    "throttled": 0
                }
            self.metrics["secret_operations"][secret_id]["total"] += 1
        
        retries = 0
        last_exception = None
        
        while retries <= self.max_retries:
            try:
                logger.debug(f"Executing Secrets Manager {operation_name} (attempt {retries + 1}/{self.max_retries + 1})")
                result = operation_func(*args, **kwargs)
                
                # Record success metrics
                self.metrics["successful_operations"] += 1
                latency = time.time() - start_time
                self.metrics["operation_latencies"].append(latency)
                
                # Track secret-specific success
                if secret_id:
                    self.metrics["secret_operations"][secret_id]["successful"] += 1
                
                # Trim latencies list if it gets too large
                if len(self.metrics["operation_latencies"]) > 1000:
                    self.metrics["operation_latencies"] = self.metrics["operation_latencies"][-1000:]
                
                logger.debug(f"Secrets Manager {operation_name} completed successfully in {latency:.2f}s")
                return result
                
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')
                error_message = e.response.get('Error', {}).get('Message', '')
                
                # Check if this is a retryable error
                if error_code in self.RETRYABLE_ERRORS and retries < self.max_retries:
                    retries += 1
                    self.metrics["retried_operations"] += 1
                    
                    # Track throttling
                    if error_code == 'ThrottlingException' or error_code == 'Throttling':
                        self.metrics["throttling_count"] += 1
                        if secret_id:
                            self.metrics["secret_operations"][secret_id]["throttled"] += 1
                    
                    # Track secret-specific retries
                    if secret_id:
                        self.metrics["secret_operations"][secret_id]["retried"] += 1
                    
                    delay = min(self.max_delay, self.base_delay * (2 ** retries))
                    
                    logger.warning(
                        f"Secrets Manager {operation_name} failed with retryable error: {error_code} - {error_message}. "
                        f"Retrying in {delay:.2f}s (attempt {retries}/{self.max_retries})"
                    )
                    
                    time.sleep(delay)
                    last_exception = e
                else:
                    # Non-retryable error or max retries exceeded
                    self.metrics["failed_operations"] += 1
                    
                    # Track secret-specific failures
                    if secret_id:
                        self.metrics["secret_operations"][secret_id]["failed"] += 1
                    
                    if error_code in self.RETRYABLE_ERRORS:
                        logger.error(
                            f"Secrets Manager {operation_name} failed after {retries} retries: {error_code} - {error_message}",
                            exc_info=True
                        )
                        raise SecretsManagerThrottlingError(f"{error_code}: {error_message}. Max retries exceeded.")
                    elif error_code in ('AccessDeniedException', 'AccessDenied', 'ResourceNotFoundException'):
                        logger.error(
                            f"Secrets Manager {operation_name} failed due to permission or resource error: {error_code} - {error_message}",
                            exc_info=True
                        )
                        raise SecretsManagerValidationError(f"{error_code}: {error_message}")
                    else:
                        logger.error(
                            f"Secrets Manager {operation_name} failed with non-retryable error: {error_code} - {error_message}",
                            exc_info=True
                        )
                        raise SecretsManagerError(f"{error_code}: {error_message}")
                        
            except (ConnectionError, EndpointConnectionError) as e:
                # Network-related errors
                if retries < self.max_retries:
                    retries += 1
                    self.metrics["retried_operations"] += 1
                    
                    # Track secret-specific retries
                    if secret_id:
                        self.metrics["secret_operations"][secret_id]["retried"] += 1
                    
                    delay = min(self.max_delay, self.base_delay * (2 ** retries))
                    
                    logger.warning(
                        f"Secrets Manager {operation_name} failed with connection error: {str(e)}. "
                        f"Retrying in {delay:.2f}s (attempt {retries}/{self.max_retries})"
                    )
                    
                    time.sleep(delay)
                    last_exception = e
                else:
                    self.metrics["failed_operations"] += 1
                    
                    # Track secret-specific failures
                    if secret_id:
                        self.metrics["secret_operations"][secret_id]["failed"] += 1
                    
                    logger.error(f"Secrets Manager {operation_name} failed after {retries} retries due to connection error", exc_info=True)
                    raise SecretsManagerConnectionError(f"Connection error: {str(e)}. Max retries exceeded.")
            
            except Exception as e:
                # Unexpected errors
                self.metrics["failed_operations"] += 1
                
                # Track secret-specific failures
                if secret_id:
                    self.metrics["secret_operations"][secret_id]["failed"] += 1
                
                logger.error(f"Unexpected error in Secrets Manager {operation_name}: {str(e)}", exc_info=True)
                raise SecretsManagerError(f"Unexpected error: {str(e)}")
        
        # This should not be reached, but just in case
        if last_exception:
            raise SecretsManagerError(f"Operation failed after {retries} retries: {str(last_exception)}")
        else:
            raise SecretsManagerError(f"Operation failed after {retries} retries")
    
    def create_secret(self, secret_id: str, secret_value: Union[str, Dict, bytes], 
                     description: Optional[str] = None, tags: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Create a new secret.
        
        Args:
            secret_id: Secret ID or name
            secret_value: Secret value (string, dictionary, or binary)
            description: Secret description (optional)
            tags: Secret tags (optional)
            
        Returns:
            Secret creation response
            
        Raises:
            SecretsManagerValidationError: If input validation fails
            SecretsManagerError: If an error occurs while creating the secret
        """
        # Validate inputs
        if not secret_id:
            raise SecretsManagerValidationError("secret_id cannot be empty")
        
        if secret_value is None:
            raise SecretsManagerValidationError("secret_value cannot be None")
        
        logger.info(f"Creating secret {secret_id}")
        
        # Prepare create parameters
        create_params = {
            'Name': secret_id
        }
        
        # Handle different secret value types
        if isinstance(secret_value, dict):
            create_params['SecretString'] = json.dumps(secret_value)
        elif isinstance(secret_value, str):
            create_params['SecretString'] = secret_value
        elif isinstance(secret_value, bytes):
            create_params['SecretBinary'] = secret_value
        else:
            raise SecretsManagerValidationError(f"Unsupported secret_value type: {type(secret_value)}")
        
        if description:
            create_params['Description'] = description
        
        if tags:
            create_params['Tags'] = tags
        
        def create_secret_operation():
            return self.secrets_manager_client.create_secret(**create_params)
        
        response = self._execute_with_retry("create_secret", create_secret_operation, secret_id)
        
        return response
    
    def get_secret_value(self, secret_id: str, version_id: Optional[str] = None, 
                        version_stage: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the value of a secret.
        
        Args:
            secret_id: Secret ID or name
            version_id: Version ID (optional)
            version_stage: Version stage (optional)
            
        Returns:
            Secret value response
            
        Raises:
            SecretsManagerValidationError: If input validation fails
            SecretsManagerError: If an error occurs while getting the secret value
        """
        # Validate inputs
        if not secret_id:
            raise SecretsManagerValidationError("secret_id cannot be empty")
        
        logger.debug(f"Getting secret value for {secret_id}")
        
        # Prepare get parameters
        get_params = {
            'SecretId': secret_id
        }
        
        if version_id:
            get_params['VersionId'] = version_id
        
        if version_stage:
            get_params['VersionStage'] = version_stage
        
        def get_secret_value_operation():
            return self.secrets_manager_client.get_secret_value(**get_params)
        
        response = self._execute_with_retry("get_secret_value", get_secret_value_operation, secret_id)
        
        return response
    
    def update_secret(self, secret_id: str, secret_value: Union[str, Dict, bytes]) -> Dict[str, Any]:
        """
        Update the value of a secret.
        
        Args:
            secret_id: Secret ID or name
            secret_value: New secret value (string, dictionary, or binary)
            
        Returns:
            Secret update response
            
        Raises:
            SecretsManagerValidationError: If input validation fails
            SecretsManagerError: If an error occurs while updating the secret
        """
        # Validate inputs
        if not secret_id:
            raise SecretsManagerValidationError("secret_id cannot be empty")
        
        if secret_value is None:
            raise SecretsManagerValidationError("secret_value cannot be None")
        
        logger.info(f"Updating secret {secret_id}")
        
        # Prepare update parameters
        update_params = {
            'SecretId': secret_id
        }
        
        # Handle different secret value types
        if isinstance(secret_value, dict):
            update_params['SecretString'] = json.dumps(secret_value)
        elif isinstance(secret_value, str):
            update_params['SecretString'] = secret_value
        elif isinstance(secret_value, bytes):
            update_params['SecretBinary'] = secret_value
        else:
            raise SecretsManagerValidationError(f"Unsupported secret_value type: {type(secret_value)}")
        
        def update_secret_operation():
            return self.secrets_manager_client.update_secret(**update_params)
        
        response = self._execute_with_retry("update_secret", update_secret_operation, secret_id)
        
        return response
    
    def delete_secret(self, secret_id: str, recovery_window_in_days: int = 30, 
                     force_delete_without_recovery: bool = False) -> Dict[str, Any]:
        """
        Delete a secret.
        
        Args:
            secret_id: Secret ID or name
            recovery_window_in_days: Recovery window in days (default: 30)
            force_delete_without_recovery: Force delete without recovery (default: False)
            
        Returns:
            Secret deletion response
            
        Raises:
            SecretsManagerValidationError: If input validation fails
            SecretsManagerError: If an error occurs while deleting the secret
        """
        # Validate inputs
        if not secret_id:
            raise SecretsManagerValidationError("secret_id cannot be empty")
        
        logger.info(f"Deleting secret {secret_id}")
        
        # Prepare delete parameters
        delete_params = {
            'SecretId': secret_id
        }
        
        if force_delete_without_recovery:
            delete_params['ForceDeleteWithoutRecovery'] = True
        else:
            delete_params['RecoveryWindowInDays'] = recovery_window_in_days
        
        def delete_secret_operation():
            return self.secrets_manager_client.delete_secret(**delete_params)
        
        response = self._execute_with_retry("delete_secret", delete_secret_operation, secret_id)
        
        return response
    
    def list_secrets(self, max_results: int = 100) -> List[Dict[str, Any]]:
        """
        List secrets.
        
        Args:
            max_results: Maximum number of secrets to return
            
        Returns:
            List of secrets
            
        Raises:
            SecretsManagerError: If an error occurs while listing secrets
        """
        # Validate inputs
        if max_results <= 0:
            raise SecretsManagerValidationError("max_results must be positive")
        
        logger.info(f"Listing secrets (max_results: {max_results})")
        
        def list_secrets_operation():
            paginator = self.secrets_manager_client.get_paginator('list_secrets')
            
            secrets = []
            for page in paginator.paginate(PaginationConfig={'MaxItems': max_results}):
                secrets.extend(page.get('SecretList', []))
                
                # Stop if we've reached the limit
                if len(secrets) >= max_results:
                    secrets = secrets[:max_results]
                    break
            
            return secrets
        
        secrets = self._execute_with_retry("list_secrets", list_secrets_operation)
        logger.info(f"Found {len(secrets)} secrets")
        return secrets
    
    def rotate_secret(self, secret_id: str, rotation_lambda_arn: Optional[str] = None,
                     rotation_rules: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Configure rotation for a secret.
        
        Args:
            secret_id: Secret ID or name
            rotation_lambda_arn: ARN of the Lambda function for rotation (optional)
            rotation_rules: Rotation rules (optional)
            
        Returns:
            Secret rotation response
            
        Raises:
            SecretsManagerValidationError: If input validation fails
            SecretsManagerError: If an error occurs while configuring rotation
        """
        # Validate inputs
        if not secret_id:
            raise SecretsManagerValidationError("secret_id cannot be empty")
        
        logger.info(f"Configuring rotation for secret {secret_id}")
        
        # Prepare rotation parameters
        rotate_params = {
            'SecretId': secret_id
        }
        
        if rotation_lambda_arn:
            rotate_params['RotationLambdaARN'] = rotation_lambda_arn
        
        if rotation_rules:
            rotate_params['RotationRules'] = rotation_rules
        
        def rotate_secret_operation():
            return self.secrets_manager_client.rotate_secret(**rotate_params)
        
        response = self._execute_with_retry("rotate_secret", rotate_secret_operation, secret_id)
        
        return response
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics for Secrets Manager operations.
        
        Returns:
            Metrics for Secrets Manager operations
        """
        # Calculate average latency
        if self.metrics["operation_latencies"]:
            avg_latency = sum(self.metrics["operation_latencies"]) / len(self.metrics["operation_latencies"])
        else:
            avg_latency = 0
        
        # Calculate success rate
        total_ops = self.metrics["total_operations"]
        if total_ops > 0:
            success_rate = (self.metrics["successful_operations"] / total_ops) * 100
        else:
            success_rate = 0
        
        # Return metrics
        return {
            "total_operations": self.metrics["total_operations"],
            "successful_operations": self.metrics["successful_operations"],
            "failed_operations": self.metrics["failed_operations"],
            "retried_operations": self.metrics["retried_operations"],
            "success_rate": success_rate,
            "average_latency": avg_latency,
            "throttling_count": self.metrics["throttling_count"],
            "secret_operations": self.metrics["secret_operations"]
        }


class AWSSecretsManagerSecurity(SecurityManager):
    """Security manager that uses AWS Secrets Manager for secure storage."""
    
    def __init__(self, region_name: str = "us-east-1", secret_prefix: str = "maif/security/",
                agent_id: Optional[str] = None):
        """
        Initialize AWS Secrets Manager security manager.
        
        Args:
            region_name: AWS region name
            secret_prefix: Prefix for secret names
            agent_id: Agent ID (optional)
        """
        super().__init__()
        
        # Override the signer with a custom one if agent_id is provided
        if agent_id:
            self.signer = MAIFSigner(agent_id=agent_id)
        
        # Initialize Secrets Manager client
        self.secrets_manager_client = SecretsManagerClient(region_name=region_name)
        
        # Secret prefix
        self.secret_prefix = secret_prefix
        
        # Secret names
        self.encryption_key_secret = f"{secret_prefix}encryption_key"
        self.private_key_secret = f"{secret_prefix}private_key_{self.signer.agent_id}"
        self.access_control_secret = f"{secret_prefix}access_control"
        self.provenance_chain_secret = f"{secret_prefix}provenance_chain_{self.signer.agent_id}"
        
        # Initialize secrets
        self._initialize_secrets()
    
    def _initialize_secrets(self):
        """Initialize secrets in Secrets Manager."""
        try:
            # Check if encryption key secret exists
            try:
                self.secrets_manager_client.get_secret_value(self.encryption_key_secret)
                logger.info(f"Encryption key secret {self.encryption_key_secret} exists")
            except SecretsManagerError:
                # Create encryption key secret
                logger.info(f"Creating encryption key secret {self.encryption_key_secret}")
                
                # Generate a random encryption key
                encryption_key = os.urandom(32)  # 256-bit key for AES-256
                
                # Store in Secrets Manager
                self.secrets_manager_client.create_secret(
                    secret_id=self.encryption_key_secret,
                    secret_value=base64.b64encode(encryption_key).decode('ascii'),
                    description="MAIF encryption key",
                    tags=[
                        {'Key': 'application', 'Value': 'maif'},
                        {'Key': 'purpose', 'Value': 'encryption'}
                    ]
                )
                
                # Store locally for immediate use
                self.encryption_key = encryption_key
            
            # Check if private key secret exists
            try:
                self.secrets_manager_client.get_secret_value(self.private_key_secret)
                logger.info(f"Private key secret {self.private_key_secret} exists")
            except SecretsManagerError:
                # Create private key secret
                logger.info(f"Creating private key secret {self.private_key_secret}")
                
                # Get private key PEM
                private_key_pem = self.signer.private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ).decode('ascii')
                
                # Store in Secrets Manager
                self.secrets_manager_client.create_secret(
                    secret_id=self.private_key_secret,
                    secret_value=private_key_pem,
                    description=f"MAIF private key for agent {self.signer.agent_id}",
                    tags=[
                        {'Key': 'application', 'Value': 'maif'},
                        {'Key': 'purpose', 'Value': 'signing'},
                        {'Key': 'agent_id', 'Value': self.signer.agent_id}
                    ]
                )
            
            # Check if access control secret exists
            try:
                self.secrets_manager_client.get_secret_value(self.access_control_secret)
                logger.info(f"Access control secret {self.access_control_secret} exists")
            except SecretsManagerError:
                # Create access control secret
                logger.info(f"Creating access control secret {self.access_control_secret}")
                
                # Create empty access control policy
                access_control_policy = {
                    "permissions": {},
                    "version": "1.0",
                    "created_at": time.time(),
                    "created_by": self.signer.agent_id
                }
                
                # Store in Secrets Manager
                self.secrets_manager_client.create_secret(
                    secret_id=self.access_control_secret,
                    secret_value=access_control_policy,
                    description="MAIF access control policy",
                    tags=[
                        {'Key': 'application', 'Value': 'maif'},
                        {'Key': 'purpose', 'Value': 'access_control'}
                    ]
                )
            
            # Check if provenance chain secret exists
            try:
                self.secrets_manager_client.get_secret_value(self.provenance_chain_secret)
                logger.info(f"Provenance chain secret {self.provenance_chain_secret} exists")
            except SecretsManagerError:
                # Create provenance chain secret
                logger.info(f"Creating provenance chain secret {self.provenance_chain_secret}")
                
                # Get provenance chain
                provenance_chain = self.signer.get_provenance_chain()
                
                # Store in Secrets Manager
                self.secrets_manager_client.create_secret(
                    secret_id=self.provenance_chain_secret,
                    secret_value={
                        "chain": provenance_chain,
                        "agent_id": self.signer.agent_id,
                        "updated_at": time.time()
                    },
                    description=f"MAIF provenance chain for agent {self.signer.agent_id}",
                    tags=[
                        {'Key': 'application', 'Value': 'maif'},
                        {'Key': 'purpose', 'Value': 'provenance'},
                        {'Key': 'agent_id', 'Value': self.signer.agent_id}
                    ]
                )
        
        except Exception as e:
            logger.error(f"Error initializing secrets: {e}", exc_info=True)
            raise
    
    def _get_encryption_key(self) -> bytes:
        """Get encryption key from Secrets Manager."""
        try:
            # Check if we already have the key in memory
            if hasattr(self, 'encryption_key'):
                return self.encryption_key
            
            # Get from Secrets Manager
            response = self.secrets_manager_client.get_secret_value(self.encryption_key_secret)
            
            if 'SecretString' in response:
                # Decode base64 string
                encryption_key = base64.b64decode(response['SecretString'])
            elif 'SecretBinary' in response:
                # Use binary directly
                encryption_key = response['SecretBinary']
            else:
                raise SecretsManagerError("No secret value found")
            
            # Cache for future use
            self.encryption_key = encryption_key
            
            return encryption_key
            
        except Exception as e:
            logger.error(f"Error getting encryption key: {e}", exc_info=True)
            
            # Generate a temporary key if needed
            if not hasattr(self, 'encryption_key'):
                logger.warning("Generating temporary encryption key")
                self.encryption_key = os.urandom(32)
            
            return self.encryption_key
    
    def _get_private_key(self):
        """Get private key from Secrets Manager."""
        try:
            # Check if we already have the key in the signer
            if hasattr(self.signer, 'private_key') and self.signer.private_key:
                return self.signer.private_key
            
            # Get from Secrets Manager
            response = self.secrets_manager_client.get_secret_value(self.private_key_secret)
            
            if 'SecretString' in response:
                # Load private key from PEM
                private_key_pem = response['SecretString']
                private_key = load_pem_private_key(
                    private_key_pem.encode('ascii'),
                    password=None,
                    backend=default_backend()
                )
                
                # Update signer
                self.signer.private_key = private_key
                
                return private_key
            else:
                raise SecretsManagerError("No private key found")
            
        except Exception as e:
            logger.error(f"Error getting private key: {e}", exc_info=True)
            
            # Return existing key if available
            if hasattr(self.signer, 'private_key') and self.signer.private_key:
                return self.signer.private_key
            
            # Generate a new key if needed
            logger.warning("Generating new private key")
            self.signer.private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
            
            return self.signer.private_key
    
    def _get_access_control_policy(self) -> Dict[str, Any]:
        """Get access control policy from Secrets Manager."""
        try:
            # Get from Secrets Manager
            response = self.secrets_manager_client.get_secret_value(self.access_control_secret)
            
            if 'SecretString' in response:
                # Parse JSON policy
                if isinstance(response['SecretString'], str):
                    policy = json.loads(response['SecretString'])
                else:
                    policy = response['SecretString']
                
                return policy
            else:
                raise SecretsManagerError("No access control policy found")
            
        except Exception as e:
            logger.error(f"Error getting access control policy: {e}", exc_info=True)
            
            # Return default policy
            return {
                "permissions": {},
                "version": "1.0",
                "created_at": time.time(),
                "created_by": self.signer.agent_id
            }
    
    def _update_access_control_policy(self, policy: Dict[str, Any]):
        """Update access control policy in Secrets Manager."""
        try:
            # Update in Secrets Manager
            self.secrets_manager_client.update_secret(
                secret_id=self.access_control_secret,
                secret_value=policy
            )
            
            logger.info(f"Updated access control policy in {self.access_control_secret}")
            
        except Exception as e:
            logger.error(f"Error updating access control policy: {e}", exc_info=True)
            raise
    
    def _get_provenance_chain(self) -> List[Dict[str, Any]]:
        """Get provenance chain from Secrets Manager."""
        try:
            # Get from Secrets Manager
            response = self.secrets_manager_client.get_secret_value(self.provenance_chain_secret)
            
            if 'SecretString' in response:
                # Parse JSON
                if isinstance(response['SecretString'], str):
                    provenance_data = json.loads(response['SecretString'])
                else:
                    provenance_data = response['SecretString']
                
                return provenance_data.get('chain', [])
            else:
                raise SecretsManagerError("No provenance chain found")
            
        except Exception as e:
            logger.error(f"Error getting provenance chain: {e}", exc_info=True)
            
            # Return empty chain
            return []
    
    def _update_provenance_chain(self, chain: List[Dict[str, Any]]):
        """Update provenance chain in Secrets Manager."""
        try:
            # Update in Secrets Manager
            self.secrets_manager_client.update_secret(
                secret_id=self.provenance_chain_secret,
                secret_value={
                    "chain": chain,
                    "agent_id": self.signer.agent_id,
                    "updated_at": time.time()
                }
            )
            
            logger.info(f"Updated provenance chain in {self.provenance_chain_secret}")
            
        except Exception as e:
            logger.error(f"Error updating provenance chain: {e}", exc_info=True)
            raise
    
    def encrypt_data(self, data: bytes) -> bytes:
        """
        Encrypt data using AES-GCM with key from Secrets Manager.
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data with metadata header
        """
        # Log the encryption event
        self.log_security_event('encrypt', {'data_size': len(data)})
        
        try:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
            from cryptography.hazmat.backends import default_backend
            import os
            
            # Get encryption key from Secrets Manager
            encryption_key = self._get_encryption_key()
            
            # Generate a random IV (Initialization Vector)
            iv = os.urandom(12)  # 96 bits as recommended for GCM
            
            # Create an encryptor
            cipher = Cipher(
                algorithms.AES(encryption_key),
                modes.GCM(iv),
                backend=default_backend()
            )
            encryptor = cipher.encryptor()
            
            # Encrypt the data
            ciphertext = encryptor.update(data) + encryptor.finalize()
            
            # Get the authentication tag
            tag = encryptor.tag
            
            # Create a metadata structure with encryption parameters
            metadata = {
                'algorithm': 'AES-GCM',
                'iv': base64.b64encode(iv).decode('ascii'),
                'tag': base64.b64encode(tag).decode('ascii'),
                'key_version': self.encryption_key_secret,
                'encrypted_at': time.time()
            }
            
            # Prepend metadata as a JSON header to the ciphertext
            metadata_bytes = json.dumps(metadata).encode('utf-8')
            header_length = len(metadata_bytes).to_bytes(4, byteorder='big')
            
            # Return the complete encrypted package
            return header_length + metadata_bytes + ciphertext
            
        except Exception as e:
            logger.error(f"Error encrypting data: {e}", exc_info=True)
            raise
    
    def decrypt_data(self, data: bytes) -> bytes:
        """
        Decrypt data that was encrypted with AES-GCM.
        
        Args:
            data: Encrypted data with metadata header
            
        Returns:
            Decrypted data
        """
        # Log the decryption event
        self.log_security_event('decrypt', {'data_size': len(data)})
        
        try:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
            from cryptography.hazmat.backends import default_backend
            
            # Extract header length
            header_length = int.from_bytes(data[:4], byteorder='big')
            
            # Extract metadata and ciphertext
            metadata_bytes = data[4:4+header_length]
            ciphertext = data[4+header_length:]
            
            # Parse metadata
            metadata = json.loads(metadata_bytes.decode('utf-8'))
            
            # Verify algorithm
            if metadata['algorithm'] != 'AES-GCM':
                raise ValueError(f"Unsupported algorithm: {metadata['algorithm']}")
            
            # Get encryption key from Secrets Manager
            encryption_key = self._get_encryption_key()
            
            # Decode IV and tag
            iv = base64.b64decode(metadata['iv'])
            tag = base64.b64decode(metadata['tag'])
            
            # Create a decryptor
            cipher = Cipher(
                algorithms.AES(encryption_key),
                modes.GCM(iv, tag),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()
            
            # Decrypt the data
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            
            return plaintext
            
        except Exception as e:
            logger.error(f"Error decrypting data: {e}", exc_info=True)
            raise
    
    def rotate_encryption_key(self) -> str:
        """
        Rotate the encryption key in Secrets Manager.
        
        Returns:
            New key version identifier
        """
        try:
            logger.info("Rotating encryption key")
            
            # Generate a new encryption key
            new_encryption_key = os.urandom(32)  # 256-bit key for AES-256
            
            # Update in Secrets Manager (creates a new version)
            response = self.secrets_manager_client.update_secret(
                secret_id=self.encryption_key_secret,
                secret_value=base64.b64encode(new_encryption_key).decode('ascii')
            )
            
            # Clear cached key
            if hasattr(self, 'encryption_key'):
                del self.encryption_key
            
            # Log security event
            self.log_security_event('key_rotation', {
                'key_id': self.encryption_key_secret,
                'version_id': response.get('VersionId', 'unknown')
            })
            
            logger.info(f"Successfully rotated encryption key, new version: {response.get('VersionId')}")
            
            return response.get('VersionId', 'unknown')
            
        except Exception as e:
            logger.error(f"Error rotating encryption key: {e}", exc_info=True)
            raise
    
    def add_provenance_entry(self, entry: ProvenanceEntry):
        """
        Add a provenance entry to the chain in Secrets Manager.
        
        Args:
            entry: Provenance entry to add
        """
        try:
            # Get current chain
            chain = self._get_provenance_chain()
            
            # Add new entry
            chain.append(entry.to_dict())
            
            # Update in Secrets Manager
            self._update_provenance_chain(chain)
            
            # Update signer's provenance chain
            self.signer.provenance_chain.append(entry)
            
            logger.info(f"Added provenance entry for action: {entry.action}")
            
        except Exception as e:
            logger.error(f"Error adding provenance entry: {e}", exc_info=True)
            raise
    
    def grant_permission(self, agent_id: str, block_id: str, permission: str):
        """
        Grant permission to an agent for a block.
        
        Args:
            agent_id: Agent ID to grant permission to
            block_id: Block ID to grant permission for
            permission: Permission type (read, write, delete)
        """
        try:
            # Get current policy
            policy = self._get_access_control_policy()
            
            # Initialize permissions dict if needed
            if 'permissions' not in policy:
                policy['permissions'] = {}
            
            # Grant permission
            permission_key = f"{agent_id}:{block_id}"
            if permission_key not in policy['permissions']:
                policy['permissions'][permission_key] = []
            
            if permission not in policy['permissions'][permission_key]:
                policy['permissions'][permission_key].append(permission)
            
            # Update policy
            policy['updated_at'] = time.time()
            policy['updated_by'] = self.signer.agent_id
            
            # Save to Secrets Manager
            self._update_access_control_policy(policy)
            
            # Update local access control manager
            self.access_control.grant_permission(agent_id, block_id, permission)
            
            # Log security event
            self.log_security_event('permission_granted', {
                'agent_id': agent_id,
                'block_id': block_id,
                'permission': permission
            })
            
            logger.info(f"Granted {permission} permission to {agent_id} for {block_id}")
            
        except Exception as e:
            logger.error(f"Error granting permission: {e}", exc_info=True)
            raise
    
    def check_permission(self, agent_id: str, block_id: str, permission: str) -> bool:
        """
        Check if an agent has permission for a block.
        
        Args:
            agent_id: Agent ID to check
            block_id: Block ID to check
            permission: Permission type to check
            
        Returns:
            True if permission is granted, False otherwise
        """
        try:
            # Get current policy
            policy = self._get_access_control_policy()
            
            # Check permission
            permission_key = f"{agent_id}:{block_id}"
            permissions = policy.get('permissions', {}).get(permission_key, [])
            
            has_permission = permission in permissions
            
            # Log security event
            self.log_security_event('permission_check', {
                'agent_id': agent_id,
                'block_id': block_id,
                'permission': permission,
                'granted': has_permission
            })
            
            return has_permission
            
        except Exception as e:
            logger.error(f"Error checking permission: {e}", exc_info=True)
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics for Secrets Manager operations.
        
        Returns:
            Combined metrics from SecurityManager and Secrets Manager client
        """
        # Get base security metrics
        security_metrics = self.get_security_status()
        
        # Get Secrets Manager client metrics
        secrets_manager_metrics = self.secrets_manager_client.get_metrics()
        
        # Combine metrics
        return {
            **security_metrics,
            'secrets_manager': secrets_manager_metrics
        }


# Export all public classes and functions
__all__ = [
    'SecretsManagerError',
    'SecretsManagerConnectionError',
    'SecretsManagerThrottlingError',
    'SecretsManagerValidationError',
    'SecretsManagerClient',
    'AWSSecretsManagerSecurity'
]