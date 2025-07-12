"""
Production Error Handling for MAIF
==================================

Comprehensive error handling with categorization, retry logic, and reporting.
"""

import sys
import traceback
from typing import Optional, Dict, Any, Callable, Type, Union, List
from functools import wraps
import time
from enum import Enum
from dataclasses import dataclass
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

from .logging_config import get_logger

logger = get_logger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    NETWORK = "network"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    RESOURCE_NOT_FOUND = "resource_not_found"
    CONFLICT = "conflict"
    DATA_CORRUPTION = "data_corruption"
    CONFIGURATION = "configuration"
    INTERNAL = "internal"
    EXTERNAL_SERVICE = "external_service"


@dataclass
class ErrorContext:
    """Context information for errors."""
    operation: str
    component: str
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    resource_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class MAIFError(Exception):
    """Base exception for all MAIF errors."""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.INTERNAL,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
        retry_allowed: bool = False
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context
        self.cause = cause
        self.retry_allowed = retry_allowed
        self.timestamp = time.time()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/reporting."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "retry_allowed": self.retry_allowed,
            "timestamp": self.timestamp,
            "context": {
                "operation": self.context.operation if self.context else None,
                "component": self.context.component if self.context else None,
                "user_id": self.context.user_id if self.context else None,
                "request_id": self.context.request_id if self.context else None,
                "resource_id": self.context.resource_id if self.context else None,
                "metadata": self.context.metadata if self.context else None,
            } if self.context else None,
            "cause": str(self.cause) if self.cause else None,
            "traceback": traceback.format_exc() if self.cause else None,
        }


# Specific error types
class ValidationError(MAIFError):
    """Validation error."""
    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            **kwargs
        )
        self.field = field


class AuthenticationError(MAIFError):
    """Authentication error."""
    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class AuthorizationError(MAIFError):
    """Authorization error."""
    def __init__(self, message: str = "Access denied", **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.AUTHORIZATION,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class NetworkError(MAIFError):
    """Network-related error."""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.MEDIUM,
            retry_allowed=True,
            **kwargs
        )


class TimeoutError(MAIFError):
    """Timeout error."""
    def __init__(self, message: str, timeout_seconds: float, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.TIMEOUT,
            severity=ErrorSeverity.MEDIUM,
            retry_allowed=True,
            **kwargs
        )
        self.timeout_seconds = timeout_seconds


class RateLimitError(MAIFError):
    """Rate limit exceeded error."""
    def __init__(self, message: str, retry_after: Optional[float] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.RATE_LIMIT,
            severity=ErrorSeverity.LOW,
            retry_allowed=True,
            **kwargs
        )
        self.retry_after = retry_after


class ResourceNotFoundError(MAIFError):
    """Resource not found error."""
    def __init__(self, resource_type: str, resource_id: str, **kwargs):
        message = f"{resource_type} with id '{resource_id}' not found"
        super().__init__(
            message,
            category=ErrorCategory.RESOURCE_NOT_FOUND,
            severity=ErrorSeverity.LOW,
            **kwargs
        )
        self.resource_type = resource_type
        self.resource_id = resource_id


class ConflictError(MAIFError):
    """Resource conflict error."""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.CONFLICT,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )


class DataCorruptionError(MAIFError):
    """Data corruption error."""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.DATA_CORRUPTION,
            severity=ErrorSeverity.CRITICAL,
            **kwargs
        )


class ConfigurationError(MAIFError):
    """Configuration error."""
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        self.config_key = config_key


class ExternalServiceError(MAIFError):
    """External service error."""
    def __init__(self, service_name: str, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.EXTERNAL_SERVICE,
            severity=ErrorSeverity.MEDIUM,
            retry_allowed=True,
            **kwargs
        )
        self.service_name = service_name


def handle_error(
    error: Exception,
    context: Optional[ErrorContext] = None,
    reraise: bool = True,
    log_level: str = "error"
) -> Optional[MAIFError]:
    """
    Handle an error with proper logging and conversion.
    
    Args:
        error: The error to handle
        context: Error context information
        reraise: Whether to re-raise the error
        log_level: Logging level to use
        
    Returns:
        MAIFError instance if not re-raised
    """
    # Convert to MAIFError if needed
    if isinstance(error, MAIFError):
        maif_error = error
    else:
        maif_error = MAIFError(
            message=str(error),
            context=context,
            cause=error
        )
    
    # Log the error
    log_func = getattr(logger, log_level)
    log_func("Error occurred", **maif_error.to_dict())
    
    # Send to monitoring/alerting if critical
    if maif_error.severity == ErrorSeverity.CRITICAL:
        _send_critical_alert(maif_error)
    
    if reraise:
        raise maif_error
    return maif_error


def _send_critical_alert(error: MAIFError):
    """Send critical error alert (implement based on your alerting system)."""
    # TODO: Implement integration with PagerDuty, SNS, etc.
    logger.critical("CRITICAL ERROR ALERT", error=error.to_dict())


def error_handler(
    operation: str,
    component: str,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    reraise: bool = True
):
    """
    Decorator for automatic error handling.
    
    Args:
        operation: Name of the operation
        component: Component name
        severity: Default error severity
        reraise: Whether to re-raise exceptions
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            context = ErrorContext(
                operation=operation,
                component=component,
                metadata={"args": str(args), "kwargs": str(kwargs)}
            )
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                handle_error(e, context=context, reraise=reraise)
                
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            context = ErrorContext(
                operation=operation,
                component=component,
                metadata={"args": str(args), "kwargs": str(kwargs)}
            )
            
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                handle_error(e, context=context, reraise=reraise)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    return decorator


def create_retry_decorator(
    max_attempts: int = 3,
    wait_multiplier: float = 1.0,
    wait_max: float = 60.0,
    retry_on: Optional[List[Type[Exception]]] = None
):
    """
    Create a retry decorator with customizable parameters.
    
    Args:
        max_attempts: Maximum number of retry attempts
        wait_multiplier: Exponential backoff multiplier
        wait_max: Maximum wait time between retries
        retry_on: List of exception types to retry on
    """
    if retry_on is None:
        retry_on = [NetworkError, TimeoutError, RateLimitError, ExternalServiceError]
    
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=wait_multiplier, max=wait_max),
        retry=retry_if_exception_type(tuple(retry_on)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )


# Production retry decorator
production_retry = create_retry_decorator()


def validate_input(
    data: Any,
    schema: Dict[str, Any],
    raise_on_error: bool = True
) -> Union[bool, List[str]]:
    """
    Validate input data against a schema.
    
    Args:
        data: Data to validate
        schema: Validation schema
        raise_on_error: Whether to raise ValidationError on failure
        
    Returns:
        True if valid, list of errors if not raising
    """
    errors = []
    
    # TODO: Implement actual schema validation
    # For now, this is a placeholder
    
    if errors and raise_on_error:
        raise ValidationError(
            f"Validation failed: {', '.join(errors)}",
            context=ErrorContext(
                operation="validate_input",
                component="validation"
            )
        )
    
    return True if not errors else errors


# Circuit breaker implementation
class CircuitBreaker:
    """Circuit breaker for handling repeated failures."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
            else:
                raise ExternalServiceError(
                    "Circuit breaker is open",
                    "Service temporarily unavailable"
                )
        
        try:
            result = func(*args, **kwargs)
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
            return result
        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                logger.error(
                    "Circuit breaker opened",
                    failure_count=self.failure_count,
                    service=func.__name__
                )
            
            raise


import asyncio