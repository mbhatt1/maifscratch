"""
AWS CloudWatch Logs Integration for MAIF Compliance Logging
===========================================================

Integrates MAIF's compliance logging with AWS CloudWatch Logs for centralized
logging, monitoring, and alerting of compliance events.
"""

import json
import time
import logging
import hashlib
import gzip
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from queue import Queue, Empty
import threading
from enum import Enum

import boto3
from botocore.exceptions import ClientError, ConnectionError, EndpointConnectionError

# Import centralized credential and config management
from .aws_config import get_aws_config, AWSConfig

from .compliance_logging import (
    ComplianceLogger, LogEntry, LogLevel, LogCategory
)

# Configure logger
logger = logging.getLogger(__name__)


class CloudWatchError(Exception):
    """Base exception for CloudWatch errors."""
    pass


class CloudWatchConnectionError(CloudWatchError):
    """Exception for CloudWatch connection errors."""
    pass


class CloudWatchThrottlingError(CloudWatchError):
    """Exception for CloudWatch throttling errors."""
    pass


class CloudWatchValidationError(CloudWatchError):
    """Exception for CloudWatch validation errors."""
    pass


@dataclass
class CloudWatchLogEvent:
    """Represents a CloudWatch log event."""
    timestamp: int  # milliseconds since epoch
    message: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to CloudWatch format."""
        return {
            'timestamp': self.timestamp,
            'message': self.message
        }


class CloudWatchLogsClient:
    """Client for AWS CloudWatch Logs with production-ready features."""
    
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
        'DataAlreadyAcceptedException'
    }
    
    # CloudWatch Logs limits
    MAX_BATCH_SIZE = 1048576  # 1 MB
    MAX_BATCH_COUNT = 10000   # Max events per batch
    MAX_EVENT_SIZE = 262144   # 256 KB per event
    
    def __init__(self, region_name: str = "us-east-1", profile_name: Optional[str] = None,
                 max_retries: int = 3, base_delay: float = 0.5, max_delay: float = 5.0):
        """
        Initialize CloudWatch Logs client.
        
        Args:
            region_name: AWS region name
            profile_name: AWS profile name (optional)
            max_retries: Maximum number of retries for transient errors
            base_delay: Base delay for exponential backoff (seconds)
            max_delay: Maximum delay for exponential backoff (seconds)
            
        Raises:
            CloudWatchConnectionError: If unable to initialize the CloudWatch client
        """
        # Validate inputs
        if not region_name:
            raise CloudWatchValidationError("region_name cannot be empty")
        
        if max_retries < 0:
            raise CloudWatchValidationError("max_retries must be non-negative")
        
        if base_delay <= 0 or max_delay <= 0:
            raise CloudWatchValidationError("base_delay and max_delay must be positive")
        
        # Initialize AWS session
        try:
            logger.info(f"Initializing CloudWatch Logs client in region {region_name}")
            session = boto3.Session(profile_name=profile_name, region_name=region_name)
            
            # Initialize CloudWatch Logs client
            self.cloudwatch_client = session.client('logs')
            
            # Retry configuration
            self.max_retries = max_retries
            self.base_delay = base_delay
            self.max_delay = max_delay
            
            # Sequence tokens for log streams
            self.sequence_tokens: Dict[str, Optional[str]] = {}
            self.sequence_token_lock = threading.Lock()
            
            # Metrics for monitoring
            self.metrics = {
                "total_operations": 0,
                "successful_operations": 0,
                "failed_operations": 0,
                "retried_operations": 0,
                "operation_latencies": [],
                "throttling_count": 0,
                "events_sent": 0,
                "bytes_sent": 0,
                "batches_sent": 0
            }
            
            logger.info("CloudWatch Logs client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize CloudWatch Logs client: {e}", exc_info=True)
            raise CloudWatchConnectionError(f"Failed to initialize CloudWatch Logs client: {e}")
    
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
            CloudWatchError: For non-retryable errors or if max retries exceeded
        """
        start_time = time.time()
        self.metrics["total_operations"] += 1
        
        retries = 0
        last_exception = None
        
        while retries <= self.max_retries:
            try:
                logger.debug(f"Executing CloudWatch {operation_name} (attempt {retries + 1}/{self.max_retries + 1})")
                result = operation_func(*args, **kwargs)
                
                # Record success metrics
                self.metrics["successful_operations"] += 1
                latency = time.time() - start_time
                self.metrics["operation_latencies"].append(latency)
                
                # Trim latencies list if it gets too large
                if len(self.metrics["operation_latencies"]) > 1000:
                    self.metrics["operation_latencies"] = self.metrics["operation_latencies"][-1000:]
                
                logger.debug(f"CloudWatch {operation_name} completed successfully in {latency:.2f}s")
                return result
                
            except ClientError as e:
                error_response = e.response or {}
                error_code = error_response.get('Error', {}).get('Code', '')
                error_message = error_response.get('Error', {}).get('Message', '')
                
                # Check if this is a retryable error
                if error_code in self.RETRYABLE_ERRORS and retries < self.max_retries:
                    retries += 1
                    self.metrics["retried_operations"] += 1
                    
                    # Track throttling
                    if error_code == 'ThrottlingException' or error_code == 'Throttling':
                        self.metrics["throttling_count"] += 1
                    
                    delay = min(self.max_delay, self.base_delay * (2 ** retries))
                    
                    logger.warning(
                        f"CloudWatch {operation_name} failed with retryable error: {error_code} - {error_message}. "
                        f"Retrying in {delay:.2f}s (attempt {retries}/{self.max_retries})"
                    )
                    
                    time.sleep(delay)
                    last_exception = e
                else:
                    # Non-retryable error or max retries exceeded
                    self.metrics["failed_operations"] += 1
                    
                    if error_code in self.RETRYABLE_ERRORS:
                        logger.error(
                            f"CloudWatch {operation_name} failed after {retries} retries: {error_code} - {error_message}",
                            exc_info=True
                        )
                        raise CloudWatchThrottlingError(f"{error_code}: {error_message}. Max retries exceeded.")
                    elif error_code in ('AccessDeniedException', 'AccessDenied', 'ResourceNotFoundException'):
                        logger.error(
                            f"CloudWatch {operation_name} failed due to permission or resource error: {error_code} - {error_message}",
                            exc_info=True
                        )
                        raise CloudWatchValidationError(f"{error_code}: {error_message}")
                    else:
                        logger.error(
                            f"CloudWatch {operation_name} failed with non-retryable error: {error_code} - {error_message}",
                            exc_info=True
                        )
                        raise CloudWatchError(f"{error_code}: {error_message}")
                        
            except (ConnectionError, EndpointConnectionError) as e:
                # Network-related errors
                if retries < self.max_retries:
                    retries += 1
                    self.metrics["retried_operations"] += 1
                    
                    delay = min(self.max_delay, self.base_delay * (2 ** retries))
                    
                    logger.warning(
                        f"CloudWatch {operation_name} failed with connection error: {str(e)}. "
                        f"Retrying in {delay:.2f}s (attempt {retries}/{self.max_retries})"
                    )
                    
                    time.sleep(delay)
                    last_exception = e
                else:
                    self.metrics["failed_operations"] += 1
                    logger.error(f"CloudWatch {operation_name} failed after {retries} retries due to connection error", exc_info=True)
                    raise CloudWatchConnectionError(f"Connection error: {str(e)}. Max retries exceeded.")
            
            except Exception as e:
                # Unexpected errors
                self.metrics["failed_operations"] += 1
                logger.error(f"Unexpected error in CloudWatch {operation_name}: {str(e)}", exc_info=True)
                raise CloudWatchError(f"Unexpected error: {str(e)}")
        
        # This should not be reached, but just in case
        if last_exception:
            raise CloudWatchError(f"Operation failed after {retries} retries: {str(last_exception)}")
        else:
            raise CloudWatchError(f"Operation failed after {retries} retries")
    
    def create_log_group(self, log_group_name: str, retention_days: int = 30,
                        kms_key_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a log group.
        
        Args:
            log_group_name: Name of the log group
            retention_days: Log retention period in days
            kms_key_id: KMS key ID for encryption (optional)
            
        Returns:
            Response from CloudWatch
        """
        logger.info(f"Creating log group: {log_group_name}")
        
        def create_operation():
            # Create log group
            create_params = {'logGroupName': log_group_name}
            if kms_key_id:
                create_params['kmsKeyId'] = kms_key_id
            
            try:
                self.cloudwatch_client.create_log_group(**create_params)
            except ClientError as e:
                if e.response['Error']['Code'] == 'ResourceAlreadyExistsException':
                    logger.info(f"Log group {log_group_name} already exists")
                else:
                    raise
            
            # Set retention policy
            if retention_days > 0:
                self.cloudwatch_client.put_retention_policy(
                    logGroupName=log_group_name,
                    retentionInDays=retention_days
                )
            
            return {'logGroupName': log_group_name, 'retentionDays': retention_days}
        
        return self._execute_with_retry("create_log_group", create_operation)
    
    def create_log_stream(self, log_group_name: str, log_stream_name: str) -> Dict[str, Any]:
        """
        Create a log stream.
        
        Args:
            log_group_name: Name of the log group
            log_stream_name: Name of the log stream
            
        Returns:
            Response from CloudWatch
        """
        logger.info(f"Creating log stream: {log_group_name}/{log_stream_name}")
        
        def create_operation():
            try:
                self.cloudwatch_client.create_log_stream(
                    logGroupName=log_group_name,
                    logStreamName=log_stream_name
                )
            except ClientError as e:
                if e.response['Error']['Code'] == 'ResourceAlreadyExistsException':
                    logger.info(f"Log stream {log_stream_name} already exists")
                else:
                    raise
            
            return {'logGroupName': log_group_name, 'logStreamName': log_stream_name}
        
        return self._execute_with_retry("create_log_stream", create_operation)
    
    def put_log_events(self, log_group_name: str, log_stream_name: str,
                      events: List[CloudWatchLogEvent]) -> Dict[str, Any]:
        """
        Put log events to CloudWatch.
        
        Args:
            log_group_name: Name of the log group
            log_stream_name: Name of the log stream
            events: List of log events
            
        Returns:
            Response from CloudWatch
        """
        if not events:
            return {'eventsAccepted': 0}
        
        # Sort events by timestamp (CloudWatch requirement)
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        
        # Get sequence token
        stream_key = f"{log_group_name}/{log_stream_name}"
        
        def put_operation():
            # Prepare parameters
            put_params = {
                'logGroupName': log_group_name,
                'logStreamName': log_stream_name,
                'logEvents': [event.to_dict() for event in sorted_events]
            }
            
            # Add sequence token if available
            with self.sequence_token_lock:
                if stream_key in self.sequence_tokens:
                    put_params['sequenceToken'] = self.sequence_tokens[stream_key]
            
            try:
                response = self.cloudwatch_client.put_log_events(**put_params)
                
                # Update sequence token
                with self.sequence_token_lock:
                    self.sequence_tokens[stream_key] = response.get('nextSequenceToken')
                
                # Update metrics
                self.metrics["events_sent"] += len(sorted_events)
                self.metrics["batches_sent"] += 1
                
                # Calculate approximate size
                batch_size = sum(len(event.message) + 26 for event in sorted_events)  # 26 bytes overhead per event
                self.metrics["bytes_sent"] += batch_size
                
                return response
                
            except ClientError as e:
                error_code = e.response['Error']['Code']
                
                if error_code == 'InvalidSequenceTokenException':
                    # Extract correct sequence token from error message
                    import re
                    match = re.search(r'sequenceToken is: (\S+)', e.response['Error']['Message'])
                    if match:
                        with self.sequence_token_lock:
                            self.sequence_tokens[stream_key] = match.group(1)
                        # Retry with correct token
                        raise e  # Will be retried
                    else:
                        # Clear token and retry
                        with self.sequence_token_lock:
                            self.sequence_tokens.pop(stream_key, None)
                        raise e
                else:
                    raise
        
        return self._execute_with_retry("put_log_events", put_operation)
    
    def create_metric_filter(self, log_group_name: str, filter_name: str,
                           filter_pattern: str, metric_name: str,
                           metric_namespace: str, metric_value: str = "1") -> Dict[str, Any]:
        """
        Create a metric filter for compliance monitoring.
        
        Args:
            log_group_name: Name of the log group
            filter_name: Name of the filter
            filter_pattern: CloudWatch filter pattern
            metric_name: Name of the metric
            metric_namespace: Namespace for the metric
            metric_value: Value to emit (default: "1")
            
        Returns:
            Response from CloudWatch
        """
        logger.info(f"Creating metric filter: {filter_name}")
        
        def create_operation():
            self.cloudwatch_client.put_metric_filter(
                logGroupName=log_group_name,
                filterName=filter_name,
                filterPattern=filter_pattern,
                metricTransformations=[
                    {
                        'metricName': metric_name,
                        'metricNamespace': metric_namespace,
                        'metricValue': metric_value,
                        'defaultValue': 0.0
                    }
                ]
            )
            
            return {
                'filterName': filter_name,
                'metricName': metric_name,
                'metricNamespace': metric_namespace
            }
        
        return self._execute_with_retry("create_metric_filter", create_operation)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics for CloudWatch operations.
        
        Returns:
            Metrics for CloudWatch operations
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
            "events_sent": self.metrics["events_sent"],
            "bytes_sent": self.metrics["bytes_sent"],
            "batches_sent": self.metrics["batches_sent"],
            "average_batch_size": self.metrics["bytes_sent"] / max(1, self.metrics["batches_sent"])
        }


class AWSCloudWatchComplianceLogger(ComplianceLogger):
    """Compliance logger that streams to AWS CloudWatch Logs."""
    
    def __init__(self, db_path: Optional[str] = None, maif_path: Optional[str] = None,
                 region_name: str = "us-east-1", log_group_name: str = "maif-compliance",
                 retention_days: int = 90, enable_metrics: bool = True,
                 batch_size: int = 100, batch_interval: float = 5.0):
        """
        Initialize CloudWatch compliance logger.
        
        Args:
            db_path: Path to SQLite database file (optional)
            maif_path: Path to MAIF file (optional)
            region_name: AWS region name
            log_group_name: CloudWatch log group name
            retention_days: Log retention period in days
            enable_metrics: Enable CloudWatch metrics
            batch_size: Number of events to batch before sending
            batch_interval: Maximum time to wait before sending batch (seconds)
        """
        # Initialize parent class
        super().__init__(db_path, maif_path)
        
        # Initialize CloudWatch client
        self.cloudwatch_client = CloudWatchLogsClient(region_name=region_name)
        
        # Configuration
        self.log_group_name = log_group_name
        self.retention_days = retention_days
        self.enable_metrics = enable_metrics
        self.batch_size = batch_size
        self.batch_interval = batch_interval
        
        # Create log group
        self.cloudwatch_client.create_log_group(
            log_group_name=log_group_name,
            retention_days=retention_days
        )
        
        # Create log streams for each category
        self.log_streams = {}
        for category in LogCategory:
            stream_name = f"{category.value}-{datetime.utcnow().strftime('%Y%m%d')}"
            self.cloudwatch_client.create_log_stream(log_group_name, stream_name)
            self.log_streams[category] = stream_name
        
        # Batch processing
        self.event_queue = Queue()
        self.batch_thread = None
        self.stop_batch_processing = threading.Event()
        
        # Start batch processing thread
        self._start_batch_processing()
        
        # Create metric filters if enabled
        if enable_metrics:
            self._create_metric_filters()
    
    def _start_batch_processing(self):
        """Start background thread for batch processing."""
        self.batch_thread = threading.Thread(target=self._batch_processor, daemon=True)
        self.batch_thread.start()
        logger.info("Started CloudWatch batch processing thread")
    
    def _batch_processor(self):
        """Background thread that processes log events in batches."""
        batch = []
        last_send_time = time.time()
        
        while not self.stop_batch_processing.is_set():
            try:
                # Wait for event with timeout
                timeout = max(0.1, self.batch_interval - (time.time() - last_send_time))
                event = self.event_queue.get(timeout=timeout)
                batch.append(event)
                
                # Check if batch is ready to send
                if len(batch) >= self.batch_size or (time.time() - last_send_time) >= self.batch_interval:
                    if batch:
                        self._send_batch(batch)
                        batch = []
                        last_send_time = time.time()
                        
            except Empty:
                # Timeout - check if we need to send partial batch
                if batch and (time.time() - last_send_time) >= self.batch_interval:
                    self._send_batch(batch)
                    batch = []
                    last_send_time = time.time()
            
            except Exception as e:
                logger.error(f"Error in batch processor: {e}", exc_info=True)
        
        # Send any remaining events
        if batch:
            self._send_batch(batch)
    
    def _send_batch(self, batch: List[Tuple[LogCategory, CloudWatchLogEvent]]):
        """Send a batch of events to CloudWatch."""
        # Group events by category
        events_by_category = {}
        for category, event in batch:
            if category not in events_by_category:
                events_by_category[category] = []
            events_by_category[category].append(event)
        
        # Send events for each category
        for category, events in events_by_category.items():
            try:
                stream_name = self.log_streams[category]
                self.cloudwatch_client.put_log_events(
                    log_group_name=self.log_group_name,
                    log_stream_name=stream_name,
                    events=events
                )
                logger.debug(f"Sent {len(events)} events to CloudWatch stream {stream_name}")
            except Exception as e:
                logger.error(f"Failed to send events to CloudWatch: {e}", exc_info=True)
    
    def _create_metric_filters(self):
        """Create CloudWatch metric filters for compliance monitoring."""
        metric_namespace = "MAIF/Compliance"
        
        # Filter for critical security events
        self.cloudwatch_client.create_metric_filter(
            log_group_name=self.log_group_name,
            filter_name=f"{self.log_group_name}-critical-events",
            filter_pattern='{ $.level = 50 }',  # CRITICAL level
            metric_name="CriticalEvents",
            metric_namespace=metric_namespace
        )
        
        # Filter for security violations
        self.cloudwatch_client.create_metric_filter(
            log_group_name=self.log_group_name,
            filter_name=f"{self.log_group_name}-security-violations",
            filter_pattern='{ $.category = "security" && $.level >= 40 }',  # ERROR or CRITICAL
            metric_name="SecurityViolations",
            metric_namespace=metric_namespace
        )
        
        # Filter for access denied events
        self.cloudwatch_client.create_metric_filter(
            log_group_name=self.log_group_name,
            filter_name=f"{self.log_group_name}-access-denied",
            filter_pattern='{ $.action = "access_denied" }',
            metric_name="AccessDeniedEvents",
            metric_namespace=metric_namespace
        )
        
        # Filter for data modifications
        self.cloudwatch_client.create_metric_filter(
            log_group_name=self.log_group_name,
            filter_name=f"{self.log_group_name}-data-modifications",
            filter_pattern='{ $.category = "data" && ($.action = "update" || $.action = "delete") }',
            metric_name="DataModifications",
            metric_namespace=metric_namespace
        )
        
        logger.info("Created CloudWatch metric filters for compliance monitoring")
    
    def log(self, level: LogLevel, category: LogCategory, user_id: str, 
           action: str, resource_id: str, details: Dict[str, Any]) -> str:
        """
        Log compliance event to both SQLite and CloudWatch.
        
        Args:
            level: Log level
            category: Log category
            user_id: User ID
            action: Action performed
            resource_id: Resource ID
            details: Additional details
            
        Returns:
            Log entry ID
        """
        # Log to SQLite first
        entry_id = super().log(level, category, user_id, action, resource_id, details)
        
        # Create CloudWatch log event
        log_data = {
            "entry_id": entry_id,
            "timestamp": time.time(),
            "level": level.value,
            "level_name": level.name,
            "category": category.value,
            "user_id": user_id,
            "action": action,
            "resource_id": resource_id,
            "details": details,
            "environment": {
                "maif_path": self.maif_path,
                "region": self.cloudwatch_client.cloudwatch_client.meta.region_name
            }
        }
        
        # Add hash for tamper detection
        log_data["hash"] = hashlib.sha256(
            json.dumps(log_data, sort_keys=True).encode()
        ).hexdigest()
        
        # Create CloudWatch event
        event = CloudWatchLogEvent(
            timestamp=int(time.time() * 1000),  # Convert to milliseconds
            message=json.dumps(log_data)
        )
        
        # Add to queue for batch processing
        self.event_queue.put((category, event))
        
        return entry_id
    
    def query_logs(self, start_time: Optional[float] = None, end_time: Optional[float] = None,
                  filter_pattern: Optional[str] = None, categories: Optional[List[LogCategory]] = None,
                  limit: int = 1000) -> List[LogEntry]:
        """
        Query logs from CloudWatch using CloudWatch Insights.
        
        Args:
            start_time: Start timestamp (seconds since epoch)
            end_time: End timestamp (seconds since epoch)
            filter_pattern: CloudWatch filter pattern
            categories: List of categories to filter
            limit: Maximum number of results
            
        Returns:
            List of log entries
        """
        # Use local SQLite query for now (CloudWatch Insights requires additional setup)
        return super().query(
            start_time=start_time,
            end_time=end_time,
            categories=categories,
            limit=limit
        )
    
    def create_compliance_dashboard(self) -> Dict[str, Any]:
        """
        Create CloudWatch dashboard for compliance monitoring.
        
        Returns:
            Dashboard configuration
        """
        dashboard_name = f"{self.log_group_name}-dashboard"
        metric_namespace = "MAIF/Compliance"
        
        dashboard_body = {
            "widgets": [
                {
                    "type": "metric",
                    "properties": {
                        "metrics": [
                            [metric_namespace, "CriticalEvents", {"stat": "Sum"}],
                            [metric_namespace, "SecurityViolations", {"stat": "Sum"}],
                            [metric_namespace, "AccessDeniedEvents", {"stat": "Sum"}],
                            [metric_namespace, "DataModifications", {"stat": "Sum"}]
                        ],
                        "period": 300,
                        "stat": "Sum",
                        "region": self.cloudwatch_client.cloudwatch_client.meta.region_name,
                        "title": "Compliance Events"
                    }
                },
                {
                    "type": "log",
                    "properties": {
                        "query": f"SOURCE '{self.log_group_name}' | fields @timestamp, level_name, category, user_id, action, resource_id | sort @timestamp desc",
                        "region": self.cloudwatch_client.cloudwatch_client.meta.region_name,
                        "title": "Recent Compliance Events"
                    }
                }
            ]
        }
        
        # Create dashboard using CloudWatch API
        cloudwatch = boto3.client('cloudwatch', region_name=self.cloudwatch_client.cloudwatch_client.meta.region_name)
        cloudwatch.put_dashboard(
            DashboardName=dashboard_name,
            DashboardBody=json.dumps(dashboard_body)
        )
        
        return {
            "dashboard_name": dashboard_name,
            "dashboard_url": f"https://console.aws.amazon.com/cloudwatch/home?region={self.cloudwatch_client.cloudwatch_client.meta.region_name}#dashboards:name={dashboard_name}"
        }
    
    def enable_real_time_alerts(self, sns_topic_arn: str,
                              critical_threshold: int = 1,
                              security_threshold: int = 5) -> List[Dict[str, Any]]:
        """
        Enable real-time alerts using CloudWatch alarms.
        
        Args:
            sns_topic_arn: SNS topic ARN for notifications
            critical_threshold: Threshold for critical events
            security_threshold: Threshold for security violations
            
        Returns:
            List of created alarms
        """
        cloudwatch = boto3.client('cloudwatch', region_name=self.cloudwatch_client.cloudwatch_client.meta.region_name)
        metric_namespace = "MAIF/Compliance"
        alarms = []
        
        # Alarm for critical events
        alarm_name = f"{self.log_group_name}-critical-events-alarm"
        cloudwatch.put_metric_alarm(
            AlarmName=alarm_name,
            ComparisonOperator='GreaterThanOrEqualToThreshold',
            EvaluationPeriods=1,
            MetricName='CriticalEvents',
            Namespace=metric_namespace,
            Period=300,
            Statistic='Sum',
            Threshold=critical_threshold,
            ActionsEnabled=True,
            AlarmActions=[sns_topic_arn],
            AlarmDescription='Alert on critical compliance events'
        )
        alarms.append({"name": alarm_name, "type": "critical_events"})
        
        # Alarm for security violations
        alarm_name = f"{self.log_group_name}-security-violations-alarm"
        cloudwatch.put_metric_alarm(
            AlarmName=alarm_name,
            ComparisonOperator='GreaterThanOrEqualToThreshold',
            EvaluationPeriods=1,
            MetricName='SecurityViolations',
            Namespace=metric_namespace,
            Period=300,
            Statistic='Sum',
            Threshold=security_threshold,
            ActionsEnabled=True,
            AlarmActions=[sns_topic_arn],
            AlarmDescription='Alert on security violations'
        )
        alarms.append({"name": alarm_name, "type": "security_violations"})
        
        logger.info(f"Created {len(alarms)} CloudWatch alarms for compliance monitoring")
        return alarms
    
    def export_to_s3(self, s3_bucket: str, s3_prefix: str = "compliance-logs/") -> Dict[str, Any]:
        """
        Export logs to S3 for long-term storage.
        
        Args:
            s3_bucket: S3 bucket name
            s3_prefix: S3 key prefix
            
        Returns:
            Export task information
        """
        # Create export task
        response = self.cloudwatch_client.cloudwatch_client.create_export_task(
            logGroupName=self.log_group_name,
            fromTime=int((datetime.utcnow() - timedelta(days=1)).timestamp() * 1000),
            to=int(datetime.utcnow().timestamp() * 1000),
            destination=s3_bucket,
            destinationPrefix=s3_prefix
        )
        
        return {
            "taskId": response['taskId'],
            "s3_location": f"s3://{s3_bucket}/{s3_prefix}"
        }
    
    def get_compliance_summary(self) -> Dict[str, Any]:
        """
        Get compliance summary including CloudWatch metrics.
        
        Returns:
            Compliance summary with statistics
        """
        # Get base summary from parent class
        summary = super().get_compliance_report()
        
        # Add CloudWatch metrics
        cloudwatch_metrics = self.cloudwatch_client.get_metrics()
        summary['cloudwatch'] = {
            'events_sent': cloudwatch_metrics['events_sent'],
            'batches_sent': cloudwatch_metrics['batches_sent'],
            'bytes_sent': cloudwatch_metrics['bytes_sent'],
            'success_rate': cloudwatch_metrics['success_rate'],
            'log_group': self.log_group_name,
            'retention_days': self.retention_days,
            'streams': len(self.log_streams)
        }
        
        return summary
    
    def close(self):
        """Close the logger and clean up resources."""
        # Stop batch processing
        self.stop_batch_processing.set()
        if self.batch_thread:
            self.batch_thread.join(timeout=5.0)
        
        # Process any remaining events
        remaining_events = []
        while not self.event_queue.empty():
            try:
                remaining_events.append(self.event_queue.get_nowait())
            except Empty:
                break
        
        if remaining_events:
            self._send_batch(remaining_events)
        
        # Close parent resources
        super().close()
        
        logger.info("CloudWatch compliance logger closed")


# Export all public classes and functions
__all__ = [
    'CloudWatchError',
    'CloudWatchConnectionError',
    'CloudWatchThrottlingError',
    'CloudWatchValidationError',
    'CloudWatchLogsClient',
    'CloudWatchLogEvent',
    'AWSCloudWatchComplianceLogger'
]