"""
Production Configuration for MAIF
=================================

Centralized configuration management for production deployments.
"""

import os
from typing import Optional, Dict, Any
from pydantic import BaseSettings, Field, validator
import structlog

logger = structlog.get_logger(__name__)


class ProductionConfig(BaseSettings):
    """Production configuration with environment variable support."""
    
    # Application settings
    app_name: str = Field(default="MAIF", env="MAIF_APP_NAME")
    environment: str = Field(default="production", env="MAIF_ENVIRONMENT")
    debug: bool = Field(default=False, env="MAIF_DEBUG")
    log_level: str = Field(default="INFO", env="MAIF_LOG_LEVEL")
    
    # AWS Configuration
    aws_region: str = Field(default="us-east-1", env="AWS_DEFAULT_REGION")
    aws_profile: Optional[str] = Field(default=None, env="AWS_PROFILE")
    s3_bucket: str = Field(default="maif-artifacts", env="MAIF_S3_BUCKET")
    dynamodb_table: str = Field(default="maif-metadata", env="MAIF_DYNAMODB_TABLE")
    kms_key_id: Optional[str] = Field(default=None, env="MAIF_KMS_KEY_ID")
    
    # Performance settings
    max_workers: int = Field(default=10, env="MAIF_MAX_WORKERS")
    batch_size: int = Field(default=100, env="MAIF_BATCH_SIZE")
    connection_pool_size: int = Field(default=10, env="MAIF_CONNECTION_POOL_SIZE")
    request_timeout: int = Field(default=30, env="MAIF_REQUEST_TIMEOUT")
    
    # Rate limiting
    rate_limit_enabled: bool = Field(default=True, env="MAIF_RATE_LIMIT_ENABLED")
    requests_per_second: float = Field(default=100.0, env="MAIF_REQUESTS_PER_SECOND")
    burst_size: int = Field(default=200, env="MAIF_BURST_SIZE")
    
    # Cost tracking
    cost_tracking_enabled: bool = Field(default=True, env="MAIF_COST_TRACKING_ENABLED")
    budget_limit: float = Field(default=1000.0, env="MAIF_BUDGET_LIMIT")
    budget_period: str = Field(default="monthly", env="MAIF_BUDGET_PERIOD")
    
    # Monitoring
    metrics_enabled: bool = Field(default=True, env="MAIF_METRICS_ENABLED")
    metrics_namespace: str = Field(default="MAIF/Production", env="MAIF_METRICS_NAMESPACE")
    health_check_interval: int = Field(default=30, env="MAIF_HEALTH_CHECK_INTERVAL")
    
    # Security
    enable_encryption: bool = Field(default=True, env="MAIF_ENABLE_ENCRYPTION")
    enable_signing: bool = Field(default=True, env="MAIF_ENABLE_SIGNING")
    enable_access_control: bool = Field(default=True, env="MAIF_ENABLE_ACCESS_CONTROL")
    
    # Retry configuration
    max_retries: int = Field(default=3, env="MAIF_MAX_RETRIES")
    retry_base_delay: float = Field(default=1.0, env="MAIF_RETRY_BASE_DELAY")
    retry_max_delay: float = Field(default=60.0, env="MAIF_RETRY_MAX_DELAY")
    
    # Cache settings
    cache_enabled: bool = Field(default=True, env="MAIF_CACHE_ENABLED")
    cache_ttl: int = Field(default=3600, env="MAIF_CACHE_TTL")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.upper()
        
    @validator("budget_period")
    def validate_budget_period(cls, v):
        """Validate budget period."""
        valid_periods = ["daily", "weekly", "monthly"]
        if v.lower() not in valid_periods:
            raise ValueError(f"Invalid budget period: {v}. Must be one of {valid_periods}")
        return v.lower()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.dict(exclude_none=True)
        
    def log_config(self, redact_sensitive: bool = True):
        """Log configuration settings."""
        config_dict = self.to_dict()
        
        if redact_sensitive:
            # Redact sensitive values
            sensitive_keys = ["aws_profile", "kms_key_id"]
            for key in sensitive_keys:
                if key in config_dict and config_dict[key]:
                    config_dict[key] = "***REDACTED***"
                    
        logger.info("Production configuration loaded", config=config_dict)


# Global configuration instance
_config: Optional[ProductionConfig] = None


def get_config() -> ProductionConfig:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = ProductionConfig()
        _config.log_config()
    return _config


def reset_config():
    """Reset configuration (mainly for testing)."""
    global _config
    _config = None


# Configuration validation
def validate_production_config():
    """Validate production configuration is properly set."""
    config = get_config()
    
    issues = []
    
    # Check critical AWS settings
    if config.environment == "production":
        if not config.kms_key_id:
            issues.append("KMS key ID not set for production")
        if config.debug:
            issues.append("Debug mode enabled in production")
        if config.log_level == "DEBUG":
            issues.append("Debug logging enabled in production")
        if not config.enable_encryption:
            issues.append("Encryption disabled in production")
        if not config.rate_limit_enabled:
            issues.append("Rate limiting disabled in production")
            
    if issues:
        logger.warning("Production configuration issues found", issues=issues)
        return False
        
    logger.info("Production configuration validated successfully")
    return True