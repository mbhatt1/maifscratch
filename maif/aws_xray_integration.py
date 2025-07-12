"""
AWS X-Ray Integration for MAIF
==============================

Provides distributed tracing capabilities for MAIF agents using AWS X-Ray.
Enables monitoring, debugging, and performance analysis of agent operations.
"""

import json
import time
import functools
import asyncio
from typing import Dict, Any, Optional, Callable, List, Union
from datetime import datetime
import logging

import boto3
from botocore.exceptions import ClientError
from aws_xray_sdk.core import xray_recorder
from aws_xray_sdk.core import patch_all
from aws_xray_sdk.ext.flask.middleware import XRayMiddleware

logger = logging.getLogger(__name__)


class MAIFXRayIntegration:
    """
    AWS X-Ray integration for MAIF agents with production-ready features.
    
    Features:
    - Automatic tracing of agent operations
    - Custom segments and subsegments
    - Metadata and annotation support
    - Error tracking and alerting
    - Performance metrics collection
    - Service map generation
    """
    
    def __init__(self, service_name: str = "MAIF-Agent", 
                 region_name: str = "us-east-1",
                 sampling_rate: float = 0.1,
                 daemon_address: Optional[str] = None):
        """
        Initialize X-Ray integration.
        
        Args:
            service_name: Name of the service for X-Ray
            region_name: AWS region
            sampling_rate: Percentage of requests to trace (0.0-1.0)
            daemon_address: X-Ray daemon address (default: 127.0.0.1:2000)
        """
        self.service_name = service_name
        self.region_name = region_name
        self.sampling_rate = sampling_rate
        
        # Configure X-Ray recorder
        xray_recorder.configure(
            service=service_name,
            sampling=sampling_rate,
            context_missing='LOG_ERROR',
            daemon_address=daemon_address or '127.0.0.1:2000',
            plugins=('EC2Plugin', 'ECSPlugin')  # Auto-detect AWS environment
        )
        
        # Patch AWS SDK and other libraries
        patch_all()
        
        # Metrics storage
        self.metrics = {
            'traces_started': 0,
            'traces_completed': 0,
            'errors_traced': 0
        }
        
        logger.info(f"X-Ray integration initialized for service: {service_name}")
    
    def trace_agent_operation(self, operation_name: str):
        """
        Decorator to trace agent operations.
        
        Args:
            operation_name: Name of the operation being traced
        """
        def decorator(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                segment = xray_recorder.begin_segment(
                    name=f"{self.service_name}.{operation_name}"
                )
                
                try:
                    # Add metadata
                    segment.put_metadata('operation', operation_name)
                    segment.put_metadata('timestamp', datetime.utcnow().isoformat())
                    
                    # Add agent context if available
                    if args and hasattr(args[0], 'agent_id'):
                        segment.put_annotation('agent_id', args[0].agent_id)
                        segment.put_metadata('agent_type', type(args[0]).__name__)
                    
                    self.metrics['traces_started'] += 1
                    
                    # Execute the operation
                    result = await func(*args, **kwargs)
                    
                    # Add result metadata
                    if result:
                        segment.put_metadata('result_type', type(result).__name__)
                    
                    segment.put_annotation('status', 'success')
                    self.metrics['traces_completed'] += 1
                    
                    return result
                    
                except Exception as e:
                    # Record error
                    segment.add_exception(e)
                    segment.put_annotation('status', 'error')
                    segment.put_metadata('error_type', type(e).__name__)
                    segment.put_metadata('error_message', str(e))
                    
                    self.metrics['errors_traced'] += 1
                    logger.error(f"Error in traced operation {operation_name}: {e}")
                    
                    raise
                    
                finally:
                    xray_recorder.end_segment()
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                segment = xray_recorder.begin_segment(
                    name=f"{self.service_name}.{operation_name}"
                )
                
                try:
                    # Add metadata
                    segment.put_metadata('operation', operation_name)
                    segment.put_metadata('timestamp', datetime.utcnow().isoformat())
                    
                    # Add agent context if available
                    if args and hasattr(args[0], 'agent_id'):
                        segment.put_annotation('agent_id', args[0].agent_id)
                        segment.put_metadata('agent_type', type(args[0]).__name__)
                    
                    self.metrics['traces_started'] += 1
                    
                    # Execute the operation
                    result = func(*args, **kwargs)
                    
                    # Add result metadata
                    if result:
                        segment.put_metadata('result_type', type(result).__name__)
                    
                    segment.put_annotation('status', 'success')
                    self.metrics['traces_completed'] += 1
                    
                    return result
                    
                except Exception as e:
                    # Record error
                    segment.add_exception(e)
                    segment.put_annotation('status', 'error')
                    segment.put_metadata('error_type', type(e).__name__)
                    segment.put_metadata('error_message', str(e))
                    
                    self.metrics['errors_traced'] += 1
                    logger.error(f"Error in traced operation {operation_name}: {e}")
                    
                    raise
                    
                finally:
                    xray_recorder.end_segment()
            
            # Return appropriate wrapper based on function type
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def trace_subsegment(self, name: str):
        """
        Create a subsegment for detailed tracing.
        
        Args:
            name: Name of the subsegment
        """
        def decorator(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                subsegment = xray_recorder.begin_subsegment(name)
                
                try:
                    result = await func(*args, **kwargs)
                    subsegment.put_annotation('status', 'success')
                    return result
                    
                except Exception as e:
                    subsegment.add_exception(e)
                    subsegment.put_annotation('status', 'error')
                    raise
                    
                finally:
                    xray_recorder.end_subsegment()
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                subsegment = xray_recorder.begin_subsegment(name)
                
                try:
                    result = func(*args, **kwargs)
                    subsegment.put_annotation('status', 'success')
                    return result
                    
                except Exception as e:
                    subsegment.add_exception(e)
                    subsegment.put_annotation('status', 'error')
                    raise
                    
                finally:
                    xray_recorder.end_subsegment()
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def add_annotation(self, key: str, value: Union[str, int, float, bool]):
        """Add annotation to current segment."""
        try:
            segment = xray_recorder.current_segment()
            if segment:
                segment.put_annotation(key, value)
        except Exception as e:
            logger.warning(f"Failed to add annotation: {e}")
    
    def add_metadata(self, key: str, value: Any, namespace: str = 'default'):
        """Add metadata to current segment."""
        try:
            segment = xray_recorder.current_segment()
            if segment:
                segment.put_metadata(key, value, namespace)
        except Exception as e:
            logger.warning(f"Failed to add metadata: {e}")
    
    def trace_aws_call(self, service_name: str):
        """
        Trace AWS service calls.
        
        Args:
            service_name: Name of the AWS service being called
        """
        def decorator(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                subsegment = xray_recorder.begin_subsegment(f"aws.{service_name}")
                
                try:
                    subsegment.put_annotation('aws_service', service_name)
                    result = await func(*args, **kwargs)
                    subsegment.put_annotation('status', 'success')
                    return result
                    
                except ClientError as e:
                    subsegment.add_exception(e)
                    subsegment.put_annotation('status', 'aws_error')
                    subsegment.put_metadata('error_code', e.response['Error']['Code'])
                    raise
                    
                except Exception as e:
                    subsegment.add_exception(e)
                    subsegment.put_annotation('status', 'error')
                    raise
                    
                finally:
                    xray_recorder.end_subsegment()
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                subsegment = xray_recorder.begin_subsegment(f"aws.{service_name}")
                
                try:
                    subsegment.put_annotation('aws_service', service_name)
                    result = func(*args, **kwargs)
                    subsegment.put_annotation('status', 'success')
                    return result
                    
                except ClientError as e:
                    subsegment.add_exception(e)
                    subsegment.put_annotation('status', 'aws_error')
                    subsegment.put_metadata('error_code', e.response['Error']['Code'])
                    raise
                    
                except Exception as e:
                    subsegment.add_exception(e)
                    subsegment.put_annotation('status', 'error')
                    raise
                    
                finally:
                    xray_recorder.end_subsegment()
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def get_trace_header(self) -> Optional[str]:
        """Get current trace header for propagation."""
        try:
            entity = xray_recorder.current_segment() or xray_recorder.current_subsegment()
            if entity:
                return entity.trace_id
        except Exception:
            pass
        return None
    
    def inject_trace_header(self, headers: Dict[str, str]):
        """Inject trace header into outgoing requests."""
        trace_header = self.get_trace_header()
        if trace_header:
            headers['X-Amzn-Trace-Id'] = trace_header
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get X-Ray integration metrics."""
        return {
            **self.metrics,
            'service_name': self.service_name,
            'sampling_rate': self.sampling_rate
        }


# Global X-Ray recorder for convenience
xray = xray_recorder


# Decorator for easy X-Ray tracing
def xray_trace(name: Optional[str] = None):
    """
    Decorator to add X-Ray tracing to any function.
    
    Args:
        name: Optional custom name for the trace
    
    Example:
        @xray_trace("process_data")
        async def process_data(data):
            # Your code here
            pass
    """
    def decorator(func):
        trace_name = name or func.__name__
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            segment = xray_recorder.begin_segment(trace_name)
            
            try:
                segment.put_metadata('function', func.__name__)
                segment.put_metadata('module', func.__module__)
                
                result = await func(*args, **kwargs)
                
                segment.put_annotation('status', 'success')
                return result
                
            except Exception as e:
                segment.add_exception(e)
                segment.put_annotation('status', 'error')
                raise
                
            finally:
                xray_recorder.end_segment()
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            segment = xray_recorder.begin_segment(trace_name)
            
            try:
                segment.put_metadata('function', func.__name__)
                segment.put_metadata('module', func.__module__)
                
                result = func(*args, **kwargs)
                
                segment.put_annotation('status', 'success')
                return result
                
            except Exception as e:
                segment.add_exception(e)
                segment.put_annotation('status', 'error')
                raise
                
            finally:
                xray_recorder.end_segment()
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# X-Ray subsegment decorator
def xray_subsegment(name: Optional[str] = None):
    """
    Decorator to create an X-Ray subsegment.
    
    Args:
        name: Optional custom name for the subsegment
    """
    def decorator(func):
        subsegment_name = name or func.__name__
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            subsegment = xray_recorder.begin_subsegment(subsegment_name)
            
            try:
                result = await func(*args, **kwargs)
                subsegment.put_annotation('status', 'success')
                return result
                
            except Exception as e:
                subsegment.add_exception(e)
                subsegment.put_annotation('status', 'error')
                raise
                
            finally:
                xray_recorder.end_subsegment()
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            subsegment = xray_recorder.begin_subsegment(subsegment_name)
            
            try:
                result = func(*args, **kwargs)
                subsegment.put_annotation('status', 'success')
                return result
                
            except Exception as e:
                subsegment.add_exception(e)
                subsegment.put_annotation('status', 'error')
                raise
                
            finally:
                xray_recorder.end_subsegment()
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator