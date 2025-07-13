"""
AWS Configuration Module for MAIF
=================================

Provides centralized configuration for all AWS service integrations
including credential management, retry policies, and service defaults.
"""

import os
import logging
import threading
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from enum import Enum
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from .aws_credentials import (
    AWSCredentialManager, 
    get_credential_manager,
    configure_aws_credentials
)

logger = logging.getLogger(__name__)


class AWSEnvironment(Enum):
    """AWS environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    GOVCLOUD = "govcloud"


@dataclass
class RetryConfig:
    """Retry configuration for AWS operations."""
    max_attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 5.0
    exponential_base: float = 2.0
    jitter: bool = True
    
    def to_boto_config(self) -> Dict[str, Any]:
        """Convert to boto3 retry configuration."""
        return {
            'max_attempts': self.max_attempts,
            'mode': 'adaptive'  # Use adaptive retry mode
        }


@dataclass
class ServiceConfig:
    """Configuration for a specific AWS service."""
    service_name: str
    region_name: Optional[str] = None
    endpoint_url: Optional[str] = None
    max_pool_connections: int = 10
    connect_timeout: int = 60
    read_timeout: int = 60
    retry_config: RetryConfig = field(default_factory=RetryConfig)
    extra_config: Dict[str, Any] = field(default_factory=dict)
    
    def to_boto_config(self) -> Config:
        """Convert to boto3 Config object."""
        config_dict = {
            'region_name': self.region_name,
            'max_pool_connections': self.max_pool_connections,
            'connect_timeout': self.connect_timeout,
            'read_timeout': self.read_timeout,
            'retries': self.retry_config.to_boto_config()
        }
        
        # Add any extra configuration
        config_dict.update(self.extra_config)
        
        return Config(**config_dict)


@dataclass
class AWSConfig:
    """
    Centralized AWS configuration for MAIF.
    
    This class manages all AWS-related configuration including:
    - Credential management
    - Service-specific configurations
    - Retry policies
    - Environment settings
    """
    
    # Credential manager
    credential_manager: Optional[AWSCredentialManager] = None
    
    # Environment
    environment: AWSEnvironment = AWSEnvironment.PRODUCTION
    
    # Default region
    default_region: str = "us-east-1"
    
    # Service configurations
    service_configs: Dict[str, ServiceConfig] = field(default_factory=dict)
    
    # Global retry configuration
    default_retry_config: RetryConfig = field(default_factory=RetryConfig)
    
    # Enable enhanced monitoring
    enable_monitoring: bool = True
    
    # Enable request/response logging (be careful with sensitive data)
    enable_debug_logging: bool = False
    
    # Custom user agent suffix
    user_agent_suffix: str = "MAIF/1.0"
    
    def __post_init__(self):
        """Initialize configuration."""
        # Use global credential manager if not provided
        if self.credential_manager is None:
            self.credential_manager = get_credential_manager()
            
        # Set up default service configurations
        self._setup_default_service_configs()
        
        # Configure logging based on environment
        self._configure_logging()
    
    def _setup_default_service_configs(self):
        """Set up default configurations for common AWS services."""
        # S3 configuration
        if 's3' not in self.service_configs:
            self.service_configs['s3'] = ServiceConfig(
                service_name='s3',
                max_pool_connections=50,  # Higher for S3 due to parallel operations
                retry_config=RetryConfig(
                    max_attempts=5,  # More retries for S3
                    base_delay=0.3
                )
            )
        
        # DynamoDB configuration
        if 'dynamodb' not in self.service_configs:
            self.service_configs['dynamodb'] = ServiceConfig(
                service_name='dynamodb',
                retry_config=RetryConfig(
                    max_attempts=3,
                    base_delay=0.1  # Faster retries for DynamoDB
                )
            )
        
        # KMS configuration
        if 'kms' not in self.service_configs:
            self.service_configs['kms'] = ServiceConfig(
                service_name='kms',
                max_pool_connections=20,
                retry_config=RetryConfig(
                    max_attempts=3,
                    base_delay=0.5
                )
            )
        
        # Lambda configuration
        if 'lambda' not in self.service_configs:
            self.service_configs['lambda'] = ServiceConfig(
                service_name='lambda',
                connect_timeout=300,  # 5 minutes for long-running operations
                read_timeout=300
            )
        
        # CloudWatch configuration
        if 'logs' not in self.service_configs:
            self.service_configs['logs'] = ServiceConfig(
                service_name='logs',
                retry_config=RetryConfig(
                    max_attempts=5,  # More retries for logging
                    base_delay=0.2
                )
            )
    
    def _configure_logging(self):
        """Configure logging based on environment."""
        if self.environment == AWSEnvironment.DEVELOPMENT:
            logging.getLogger('boto3').setLevel(logging.DEBUG if self.enable_debug_logging else logging.INFO)
            logging.getLogger('botocore').setLevel(logging.DEBUG if self.enable_debug_logging else logging.INFO)
        elif self.environment == AWSEnvironment.PRODUCTION:
            logging.getLogger('boto3').setLevel(logging.WARNING)
            logging.getLogger('botocore').setLevel(logging.WARNING)
    
    def get_client(self, service_name: str, **kwargs) -> Any:
        """
        Get AWS service client with configuration.
        
        Args:
            service_name: AWS service name
            **kwargs: Additional client parameters
            
        Returns:
            Configured AWS service client
        """
        # Get service configuration
        service_config = self.service_configs.get(
            service_name,
            ServiceConfig(service_name=service_name)
        )
        
        # Merge configurations
        config = service_config.to_boto_config()
        
        # Add user agent
        if self.user_agent_suffix:
            config.user_agent_extra = self.user_agent_suffix
        
        # Override region if specified
        region_name = kwargs.pop('region_name', None) or service_config.region_name or self.default_region
        
        # Get client from credential manager
        return self.credential_manager.get_client(
            service_name,
            region_name=region_name,
            config=config,
            **kwargs
        )
    
    def get_resource(self, service_name: str, **kwargs) -> Any:
        """
        Get AWS service resource with configuration.
        
        Args:
            service_name: AWS service name
            **kwargs: Additional resource parameters
            
        Returns:
            Configured AWS service resource
        """
        # Get service configuration
        service_config = self.service_configs.get(
            service_name,
            ServiceConfig(service_name=service_name)
        )
        
        # Merge configurations
        config = service_config.to_boto_config()
        
        # Add user agent
        if self.user_agent_suffix:
            config.user_agent_extra = self.user_agent_suffix
        
        # Override region if specified
        region_name = kwargs.pop('region_name', None) or service_config.region_name or self.default_region
        
        # Get resource from credential manager
        return self.credential_manager.get_resource(
            service_name,
            region_name=region_name,
            config=config,
            **kwargs
        )
    
    def update_service_config(self, service_name: str, config: ServiceConfig):
        """Update configuration for a specific service."""
        self.service_configs[service_name] = config
        logger.info(f"Updated configuration for service: {service_name}")
    
    def get_service_config(self, service_name: str) -> ServiceConfig:
        """Get configuration for a specific service."""
        return self.service_configs.get(
            service_name,
            ServiceConfig(service_name=service_name)
        )
    
    @classmethod
    def from_environment(cls, environment: Optional[str] = None) -> 'AWSConfig':
        """
        Create AWS configuration from environment variables.
        
        Environment variables:
        - AWS_ENVIRONMENT: development, staging, production, govcloud
        - AWS_DEFAULT_REGION: Default AWS region
        - AWS_ENABLE_MONITORING: Enable monitoring (true/false)
        - AWS_ENABLE_DEBUG: Enable debug logging (true/false)
        """
        env_name = environment or os.environ.get('AWS_ENVIRONMENT', 'production')
        
        try:
            env = AWSEnvironment(env_name.lower())
        except ValueError:
            logger.warning(f"Unknown environment '{env_name}', defaulting to production")
            env = AWSEnvironment.PRODUCTION
        
        return cls(
            environment=env,
            default_region=os.environ.get('AWS_DEFAULT_REGION', 'us-east-1'),
            enable_monitoring=os.environ.get('AWS_ENABLE_MONITORING', 'true').lower() == 'true',
            enable_debug_logging=os.environ.get('AWS_ENABLE_DEBUG', 'false').lower() == 'true'
        )
    
    @classmethod
    def for_govcloud(cls) -> 'AWSConfig':
        """Create configuration for AWS GovCloud."""
        config = cls(
            environment=AWSEnvironment.GOVCLOUD,
            default_region='us-gov-east-1'
        )
        
        # Update service endpoints for GovCloud
        for service_config in config.service_configs.values():
            if service_config.region_name and 'us-east-1' in service_config.region_name:
                service_config.region_name = 'us-gov-east-1'
        
        return config
    
    def validate_configuration(self) -> bool:
        """
        Validate AWS configuration.
        
        Returns:
            True if configuration is valid
            
        Raises:
            Exception if configuration is invalid
        """
        # Validate credentials
        try:
            self.credential_manager.validate_credentials()
        except Exception as e:
            logger.error(f"Credential validation failed: {e}")
            raise
        
        # Test connectivity to key services
        test_services = ['s3', 'sts']
        for service in test_services:
            try:
                client = self.get_client(service)
                if service == 's3':
                    client.list_buckets(MaxBuckets=1)
                elif service == 'sts':
                    client.get_caller_identity()
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                if error_code not in ['AccessDenied', 'UnauthorizedOperation']:
                    logger.error(f"Failed to connect to {service}: {e}")
                    raise
        
        logger.info("AWS configuration validated successfully")
        return True


# Global configuration instance
_aws_config: Optional[AWSConfig] = None
_config_lock = threading.Lock()


def get_aws_config() -> AWSConfig:
    """
    Get the global AWS configuration instance.
    
    Returns:
        AWSConfig instance
    """
    global _aws_config
    
    with _config_lock:
        if _aws_config is None:
            # Create default configuration from environment
            _aws_config = AWSConfig.from_environment()
            logger.info(f"Created AWS configuration for environment: {_aws_config.environment.value}")
        
        return _aws_config


def set_aws_config(config: AWSConfig):
    """
    Set the global AWS configuration instance.
    
    Args:
        config: AWSConfig instance to use globally
    """
    global _aws_config
    
    with _config_lock:
        _aws_config = config
        logger.info(f"Global AWS configuration updated for environment: {config.environment.value}")


def configure_aws(
        environment: Optional[str] = None,
        region: Optional[str] = None,
        profile: Optional[str] = None,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        role_arn: Optional[str] = None,
        enable_monitoring: bool = True,
        enable_debug: bool = False) -> AWSConfig:
    """
    Configure AWS settings globally for MAIF.
    
    This is a convenience function to configure all AWS settings at once.
    
    Args:
        environment: Environment name (development, staging, production, govcloud)
        region: Default AWS region
        profile: AWS profile name
        access_key_id: AWS access key ID
        secret_access_key: AWS secret access key
        role_arn: IAM role ARN to assume
        enable_monitoring: Enable CloudWatch monitoring
        enable_debug: Enable debug logging
        
    Returns:
        Configured AWSConfig instance
    """
    # Configure credentials
    if any([profile, access_key_id, role_arn]):
        credential_manager = configure_aws_credentials(
            profile_name=profile,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            region_name=region or 'us-east-1',
            role_arn=role_arn
        )
    else:
        credential_manager = get_credential_manager()
    
    # Determine environment
    if environment:
        try:
            env = AWSEnvironment(environment.lower())
        except ValueError:
            logger.warning(f"Unknown environment '{environment}', defaulting to production")
            env = AWSEnvironment.PRODUCTION
    else:
        env = AWSEnvironment.PRODUCTION
    
    # Create configuration
    config = AWSConfig(
        credential_manager=credential_manager,
        environment=env,
        default_region=region or credential_manager.region_name,
        enable_monitoring=enable_monitoring,
        enable_debug_logging=enable_debug
    )
    
    # Set as global configuration
    set_aws_config(config)
    
    # Validate configuration
    try:
        config.validate_configuration()
    except Exception as e:
        logger.error(f"AWS configuration validation failed: {e}")
        raise
    
    return config


# Import commonly used functions at module level
__all__ = [
    'AWSConfig',
    'ServiceConfig',
    'RetryConfig',
    'AWSEnvironment',
    'get_aws_config',
    'set_aws_config',
    'configure_aws',
    # Re-export credential functions
    'get_credential_manager',
    'configure_aws_credentials'
]