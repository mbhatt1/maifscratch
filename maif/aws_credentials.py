"""
AWS Credentials Manager for MAIF
================================

Provides centralized credential management for all AWS service integrations
with support for multiple credential sources and security best practices.
"""

import os
import logging
import json
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
from enum import Enum
import boto3
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError
from botocore.session import Session
import threading
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CredentialSource(Enum):
    """Supported AWS credential sources in order of precedence."""
    ENVIRONMENT = "environment"
    IAM_ROLE = "iam_role"
    PROFILE = "profile"
    EXPLICIT = "explicit"
    CONTAINER = "container"
    INSTANCE_METADATA = "instance_metadata"


@dataclass
class AWSCredentials:
    """AWS credentials container."""
    access_key_id: str
    secret_access_key: str
    session_token: Optional[str] = None
    expiration: Optional[datetime] = None
    source: Optional[CredentialSource] = None
    
    def is_expired(self) -> bool:
        """Check if credentials are expired."""
        if self.expiration is None:
            return False
        return datetime.utcnow() >= self.expiration
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for boto3."""
        creds = {
            'aws_access_key_id': self.access_key_id,
            'aws_secret_access_key': self.secret_access_key
        }
        if self.session_token:
            creds['aws_session_token'] = self.session_token
        return creds


class AWSCredentialManager:
    """
    Centralized AWS credential manager for MAIF.
    
    Supports multiple credential sources:
    1. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    2. IAM roles (for EC2, ECS, Lambda)
    3. AWS profiles (~/.aws/credentials)
    4. Explicit credentials
    5. Container credentials (ECS)
    6. Instance metadata service (EC2)
    """
    
    def __init__(self, 
                 profile_name: Optional[str] = None,
                 access_key_id: Optional[str] = None,
                 secret_access_key: Optional[str] = None,
                 session_token: Optional[str] = None,
                 region_name: str = "us-east-1",
                 role_arn: Optional[str] = None,
                 role_session_name: Optional[str] = None,
                 validate_credentials: bool = True):
        """
        Initialize credential manager.
        
        Args:
            profile_name: AWS profile name
            access_key_id: Explicit AWS access key ID
            secret_access_key: Explicit AWS secret access key
            session_token: AWS session token for temporary credentials
            region_name: AWS region name
            role_arn: ARN of role to assume
            role_session_name: Session name for assumed role
            validate_credentials: Whether to validate credentials on initialization
        """
        self.profile_name = profile_name
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.session_token = session_token
        self.region_name = region_name
        self.role_arn = role_arn
        self.role_session_name = role_session_name or f"maif-session-{os.getpid()}"
        
        self._credentials_cache: Optional[AWSCredentials] = None
        self._session_cache: Optional[boto3.Session] = None
        self._lock = threading.RLock()
        
        # Detect credential source
        self.credential_source = self._detect_credential_source()
        logger.info(f"Detected credential source: {self.credential_source.value}")
        
        # Validate credentials if requested
        if validate_credentials:
            self.validate_credentials()
    
    def _detect_credential_source(self) -> CredentialSource:
        """Detect which credential source is being used."""
        # Check explicit credentials
        if self.access_key_id and self.secret_access_key:
            return CredentialSource.EXPLICIT
            
        # Check environment variables
        if os.environ.get('AWS_ACCESS_KEY_ID') and os.environ.get('AWS_SECRET_ACCESS_KEY'):
            return CredentialSource.ENVIRONMENT
            
        # Check for profile
        if self.profile_name:
            return CredentialSource.PROFILE
            
        # Check for container credentials
        if os.environ.get('AWS_CONTAINER_CREDENTIALS_RELATIVE_URI'):
            return CredentialSource.CONTAINER
            
        # Check for IAM role
        try:
            # Try to get instance metadata
            session = Session()
            credentials = session.get_credentials()
            if credentials and hasattr(credentials, 'method'):
                if 'iam-role' in credentials.method:
                    return CredentialSource.IAM_ROLE
                elif 'instance-metadata' in credentials.method:
                    return CredentialSource.INSTANCE_METADATA
        except Exception:
            pass
            
        # Default to environment
        return CredentialSource.ENVIRONMENT
    
    def get_session(self, refresh: bool = False) -> boto3.Session:
        """
        Get boto3 session with credentials.
        
        Args:
            refresh: Force refresh of credentials
            
        Returns:
            boto3.Session configured with credentials
            
        Raises:
            NoCredentialsError: If no credentials are available
        """
        with self._lock:
            if not refresh and self._session_cache:
                # Check if cached credentials are still valid
                if self._credentials_cache and not self._credentials_cache.is_expired():
                    return self._session_cache
                    
            # Create new session
            logger.info("Creating new AWS session")
            
            # Start with basic session parameters
            session_params = {'region_name': self.region_name}
            
            # Add profile if specified
            if self.profile_name:
                session_params['profile_name'] = self.profile_name
                
            # Add explicit credentials if provided
            if self.access_key_id and self.secret_access_key:
                session_params['aws_access_key_id'] = self.access_key_id
                session_params['aws_secret_access_key'] = self.secret_access_key
                if self.session_token:
                    session_params['aws_session_token'] = self.session_token
                    
            try:
                session = boto3.Session(**session_params)
                
                # If role ARN is specified, assume the role
                if self.role_arn:
                    session = self._assume_role(session)
                    
                self._session_cache = session
                
                # Cache credentials for expiration checking
                self._cache_credentials(session)
                
                return session
                
            except Exception as e:
                logger.error(f"Failed to create AWS session: {e}")
                raise NoCredentialsError(f"Failed to create AWS session: {e}")
    
    def _assume_role(self, session: boto3.Session) -> boto3.Session:
        """Assume an IAM role and return new session."""
        logger.info(f"Assuming role: {self.role_arn}")
        
        sts_client = session.client('sts')
        
        try:
            response = sts_client.assume_role(
                RoleArn=self.role_arn,
                RoleSessionName=self.role_session_name,
                DurationSeconds=3600  # 1 hour
            )
            
            credentials = response['Credentials']
            
            # Create new session with assumed role credentials
            assumed_session = boto3.Session(
                aws_access_key_id=credentials['AccessKeyId'],
                aws_secret_access_key=credentials['SecretAccessKey'],
                aws_session_token=credentials['SessionToken'],
                region_name=self.region_name
            )
            
            # Update credentials cache
            self._credentials_cache = AWSCredentials(
                access_key_id=credentials['AccessKeyId'],
                secret_access_key=credentials['SecretAccessKey'],
                session_token=credentials['SessionToken'],
                expiration=credentials['Expiration'],
                source=CredentialSource.IAM_ROLE
            )
            
            logger.info(f"Successfully assumed role: {self.role_arn}")
            return assumed_session
            
        except ClientError as e:
            logger.error(f"Failed to assume role {self.role_arn}: {e}")
            raise
    
    def _cache_credentials(self, session: boto3.Session):
        """Cache credentials from session for expiration checking."""
        try:
            credentials = session.get_credentials()
            if credentials:
                frozen_credentials = credentials.get_frozen_credentials()
                
                expiration = None
                if hasattr(credentials, '_expiry_time'):
                    expiration = credentials._expiry_time
                    
                self._credentials_cache = AWSCredentials(
                    access_key_id=frozen_credentials.access_key,
                    secret_access_key=frozen_credentials.secret_key,
                    session_token=frozen_credentials.token,
                    expiration=expiration,
                    source=self.credential_source
                )
        except Exception as e:
            logger.warning(f"Failed to cache credentials: {e}")
    
    def get_client(self, service_name: str, **kwargs) -> Any:
        """
        Get AWS service client with managed credentials.
        
        Args:
            service_name: AWS service name (e.g., 's3', 'dynamodb')
            **kwargs: Additional client configuration
            
        Returns:
            AWS service client
        """
        session = self.get_session()
        
        # Merge kwargs with defaults
        client_config = {
            'service_name': service_name,
            'region_name': self.region_name
        }
        client_config.update(kwargs)
        
        return session.client(**client_config)
    
    def get_resource(self, service_name: str, **kwargs) -> Any:
        """
        Get AWS service resource with managed credentials.
        
        Args:
            service_name: AWS service name (e.g., 's3', 'dynamodb')
            **kwargs: Additional resource configuration
            
        Returns:
            AWS service resource
        """
        session = self.get_session()
        
        # Merge kwargs with defaults
        resource_config = {
            'service_name': service_name,
            'region_name': self.region_name
        }
        resource_config.update(kwargs)
        
        return session.resource(**resource_config)
    
    def validate_credentials(self) -> bool:
        """
        Validate AWS credentials by making a simple API call.
        
        Returns:
            True if credentials are valid
            
        Raises:
            NoCredentialsError: If no credentials are available
            ClientError: If credentials are invalid
        """
        logger.info("Validating AWS credentials")
        
        try:
            # Use STS get_caller_identity as it works with all credential types
            sts_client = self.get_client('sts')
            response = sts_client.get_caller_identity()
            
            account_id = response.get('Account')
            user_arn = response.get('Arn')
            
            logger.info(f"Credentials validated successfully - Account: {account_id}, ARN: {user_arn}")
            return True
            
        except NoCredentialsError:
            logger.error("No AWS credentials found")
            raise
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            logger.error(f"Invalid AWS credentials: {error_code}")
            raise
        except Exception as e:
            logger.error(f"Failed to validate credentials: {e}")
            raise
    
    def get_credential_info(self) -> Dict[str, Any]:
        """Get information about current credentials."""
        info = {
            'source': self.credential_source.value,
            'region': self.region_name,
            'profile': self.profile_name,
            'role_arn': self.role_arn,
            'is_temporary': False,
            'is_expired': False,
            'expiration': None
        }
        
        if self._credentials_cache:
            info['is_temporary'] = self._credentials_cache.session_token is not None
            info['is_expired'] = self._credentials_cache.is_expired()
            if self._credentials_cache.expiration:
                info['expiration'] = self._credentials_cache.expiration.isoformat()
                
        return info
    
    def refresh_credentials(self) -> bool:
        """
        Refresh temporary credentials if needed.
        
        Returns:
            True if credentials were refreshed
        """
        with self._lock:
            if self._credentials_cache and self._credentials_cache.is_expired():
                logger.info("Refreshing expired credentials")
                self.get_session(refresh=True)
                return True
            return False
    
    @classmethod
    def from_environment(cls, region_name: str = "us-east-1") -> 'AWSCredentialManager':
        """Create credential manager from environment variables."""
        return cls(
            access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
            session_token=os.environ.get('AWS_SESSION_TOKEN'),
            region_name=os.environ.get('AWS_DEFAULT_REGION', region_name)
        )
    
    @classmethod
    def from_profile(cls, profile_name: str, region_name: str = "us-east-1") -> 'AWSCredentialManager':
        """Create credential manager from AWS profile."""
        return cls(
            profile_name=profile_name,
            region_name=region_name
        )
    
    @classmethod
    def from_role(cls, role_arn: str, session_name: Optional[str] = None,
                  region_name: str = "us-east-1") -> 'AWSCredentialManager':
        """Create credential manager that assumes an IAM role."""
        return cls(
            role_arn=role_arn,
            role_session_name=session_name,
            region_name=region_name
        )


# Global credential manager instance
_credential_manager: Optional[AWSCredentialManager] = None
_manager_lock = threading.Lock()


def get_credential_manager() -> AWSCredentialManager:
    """
    Get the global credential manager instance.
    
    Returns:
        AWSCredentialManager instance
    """
    global _credential_manager
    
    with _manager_lock:
        if _credential_manager is None:
            # Create default manager from environment
            _credential_manager = AWSCredentialManager.from_environment()
            
        return _credential_manager


def set_credential_manager(manager: AWSCredentialManager):
    """
    Set the global credential manager instance.
    
    Args:
        manager: AWSCredentialManager instance to use globally
    """
    global _credential_manager
    
    with _manager_lock:
        _credential_manager = manager
        logger.info(f"Global credential manager updated - Source: {manager.credential_source.value}")


def configure_aws_credentials(
        profile_name: Optional[str] = None,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        session_token: Optional[str] = None,
        region_name: str = "us-east-1",
        role_arn: Optional[str] = None,
        role_session_name: Optional[str] = None) -> AWSCredentialManager:
    """
    Configure and set global AWS credentials.
    
    This is a convenience function to configure AWS credentials globally
    for all MAIF AWS integrations.
    
    Args:
        profile_name: AWS profile name
        access_key_id: Explicit AWS access key ID
        secret_access_key: Explicit AWS secret access key
        session_token: AWS session token for temporary credentials
        region_name: AWS region name
        role_arn: ARN of role to assume
        role_session_name: Session name for assumed role
        
    Returns:
        Configured AWSCredentialManager instance
    """
    manager = AWSCredentialManager(
        profile_name=profile_name,
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        session_token=session_token,
        region_name=region_name,
        role_arn=role_arn,
        role_session_name=role_session_name
    )
    
    set_credential_manager(manager)
    return manager