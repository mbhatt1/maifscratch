"""
AWS Kinesis Integration for MAIF Streaming
=======================================

Integrates MAIF's streaming modules with AWS Kinesis for scalable, real-time data streaming.
"""

import json
import time
import logging
import uuid
import base64
import threading
from typing import Iterator, Tuple, Optional, Dict, Any, List, Union, BinaryIO
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import io
import os

import boto3
from botocore.exceptions import ClientError, ConnectionError, EndpointConnectionError

from .streaming import (
    StreamingConfig, MAIFStreamReader, MAIFStreamWriter, 
    PerformanceProfiler, StreamingMAIFProcessor
)
from .streaming_ultra import UltraStreamingConfig, UltraHighThroughputReader
from .core import MAIFEncoder, MAIFDecoder, MAIFBlock

# Configure logger
logger = logging.getLogger(__name__)


class KinesisStreamingError(Exception):
    """Base exception for Kinesis streaming errors."""
    pass


class KinesisConnectionError(KinesisStreamingError):
    """Exception for Kinesis connection errors."""
    pass


class KinesisThrottlingError(KinesisStreamingError):
    """Exception for Kinesis throttling errors."""
    pass


class KinesisValidationError(KinesisStreamingError):
    """Exception for Kinesis validation errors."""
    pass


@dataclass
class KinesisStreamingConfig(StreamingConfig):
    """Configuration for Kinesis streaming operations."""
    region_name: str = "us-east-1"
    max_retries: int = 3
    base_delay: float = 0.5
    max_delay: float = 5.0
    batch_size: int = 500  # Maximum records per batch for Kinesis
    max_batch_size_bytes: int = 5 * 1024 * 1024  # 5MB maximum batch size for Kinesis
    max_record_size_bytes: int = 1024 * 1024  # 1MB maximum record size for Kinesis
    use_enhanced_fan_out: bool = False
    consumer_name: Optional[str] = None
    shard_iterator_type: str = "TRIM_HORIZON"
    enable_metrics: bool = True
    metrics_namespace: str = "MAIF/Kinesis"


class KinesisClient:
    """Client for AWS Kinesis service with production-ready features."""
    
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
        'ProvisionedThroughputExceededException',
        'LimitExceededException'
    }
    
    def __init__(self, region_name: str = "us-east-1", profile_name: Optional[str] = None,
                max_retries: int = 3, base_delay: float = 0.5, max_delay: float = 5.0):
        """
        Initialize Kinesis client.
        
        Args:
            region_name: AWS region name
            profile_name: AWS profile name (optional)
            max_retries: Maximum number of retries for transient errors
            base_delay: Base delay for exponential backoff (seconds)
            max_delay: Maximum delay for exponential backoff (seconds)
            
        Raises:
            KinesisConnectionError: If unable to initialize the Kinesis client
        """
        # Validate inputs
        if not region_name:
            raise KinesisValidationError("region_name cannot be empty")
        
        if max_retries < 0:
            raise KinesisValidationError("max_retries must be non-negative")
        
        if base_delay <= 0 or max_delay <= 0:
            raise KinesisValidationError("base_delay and max_delay must be positive")
        
        # Initialize AWS session
        try:
            logger.info(f"Initializing Kinesis client in region {region_name}")
            session = boto3.Session(profile_name=profile_name, region_name=region_name)
            
            # Initialize Kinesis client
            self.kinesis_client = session.client('kinesis')
            
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
                "bytes_sent": 0,
                "bytes_received": 0,
                "records_sent": 0,
                "records_received": 0,
                "stream_operations": {}
            }
            
            logger.info("Kinesis client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Kinesis client: {e}", exc_info=True)
            raise KinesisConnectionError(f"Failed to initialize Kinesis client: {e}")
    
    def _execute_with_retry(self, operation_name: str, operation_func, stream_name: Optional[str] = None, *args, **kwargs):
        """
        Execute an operation with exponential backoff retry for transient errors.
        
        Args:
            operation_name: Name of the operation for logging
            operation_func: Function to execute
            stream_name: Name of the stream (for metrics)
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Result of the operation
            
        Raises:
            KinesisStreamingError: For non-retryable errors or if max retries exceeded
        """
        start_time = time.time()
        self.metrics["total_operations"] += 1
        
        # Track stream-specific operations
        if stream_name:
            if stream_name not in self.metrics["stream_operations"]:
                self.metrics["stream_operations"][stream_name] = {
                    "total": 0,
                    "successful": 0,
                    "failed": 0,
                    "retried": 0,
                    "throttled": 0,
                    "bytes_sent": 0,
                    "bytes_received": 0,
                    "records_sent": 0,
                    "records_received": 0
                }
            self.metrics["stream_operations"][stream_name]["total"] += 1
        
        retries = 0
        last_exception = None
        
        while retries <= self.max_retries:
            try:
                logger.debug(f"Executing Kinesis {operation_name} (attempt {retries + 1}/{self.max_retries + 1})")
                result = operation_func(*args, **kwargs)
                
                # Record success metrics
                self.metrics["successful_operations"] += 1
                latency = time.time() - start_time
                self.metrics["operation_latencies"].append(latency)
                
                # Track stream-specific success
                if stream_name:
                    self.metrics["stream_operations"][stream_name]["successful"] += 1
                
                # Trim latencies list if it gets too large
                if len(self.metrics["operation_latencies"]) > 1000:
                    self.metrics["operation_latencies"] = self.metrics["operation_latencies"][-1000:]
                
                logger.debug(f"Kinesis {operation_name} completed successfully in {latency:.2f}s")
                return result
                
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')
                error_message = e.response.get('Error', {}).get('Message', '')
                
                # Check if this is a retryable error
                if error_code in self.RETRYABLE_ERRORS and retries < self.max_retries:
                    retries += 1
                    self.metrics["retried_operations"] += 1
                    
                    # Track throttling
                    if error_code == 'ProvisionedThroughputExceededException' or error_code == 'ThrottlingException':
                        self.metrics["throttling_count"] += 1
                        if stream_name:
                            self.metrics["stream_operations"][stream_name]["throttled"] += 1
                    
                    # Track stream-specific retries
                    if stream_name:
                        self.metrics["stream_operations"][stream_name]["retried"] += 1
                    
                    delay = min(self.max_delay, self.base_delay * (2 ** retries))
                    
                    logger.warning(
                        f"Kinesis {operation_name} failed with retryable error: {error_code} - {error_message}. "
                        f"Retrying in {delay:.2f}s (attempt {retries}/{self.max_retries})"
                    )
                    
                    time.sleep(delay)
                    last_exception = e
                else:
                    # Non-retryable error or max retries exceeded
                    self.metrics["failed_operations"] += 1
                    
                    # Track stream-specific failures
                    if stream_name:
                        self.metrics["stream_operations"][stream_name]["failed"] += 1
                    
                    if error_code in self.RETRYABLE_ERRORS:
                        logger.error(
                            f"Kinesis {operation_name} failed after {retries} retries: {error_code} - {error_message}",
                            exc_info=True
                        )
                        raise KinesisThrottlingError(f"{error_code}: {error_message}. Max retries exceeded.")
                    elif error_code in ('AccessDeniedException', 'AccessDenied', 'ResourceNotFoundException'):
                        logger.error(
                            f"Kinesis {operation_name} failed due to permission or resource error: {error_code} - {error_message}",
                            exc_info=True
                        )
                        raise KinesisValidationError(f"{error_code}: {error_message}")
                    else:
                        logger.error(
                            f"Kinesis {operation_name} failed with non-retryable error: {error_code} - {error_message}",
                            exc_info=True
                        )
                        raise KinesisStreamingError(f"{error_code}: {error_message}")
                        
            except (ConnectionError, EndpointConnectionError) as e:
                # Network-related errors
                if retries < self.max_retries:
                    retries += 1
                    self.metrics["retried_operations"] += 1
                    
                    # Track stream-specific retries
                    if stream_name:
                        self.metrics["stream_operations"][stream_name]["retried"] += 1
                    
                    delay = min(self.max_delay, self.base_delay * (2 ** retries))
                    
                    logger.warning(
                        f"Kinesis {operation_name} failed with connection error: {str(e)}. "
                        f"Retrying in {delay:.2f}s (attempt {retries}/{self.max_retries})"
                    )
                    
                    time.sleep(delay)
                    last_exception = e
                else:
                    self.metrics["failed_operations"] += 1
                    
                    # Track stream-specific failures
                    if stream_name:
                        self.metrics["stream_operations"][stream_name]["failed"] += 1
                    
                    logger.error(f"Kinesis {operation_name} failed after {retries} retries due to connection error", exc_info=True)
                    raise KinesisConnectionError(f"Connection error: {str(e)}. Max retries exceeded.")
            
            except Exception as e:
                # Unexpected errors
                self.metrics["failed_operations"] += 1
                
                # Track stream-specific failures
                if stream_name:
                    self.metrics["stream_operations"][stream_name]["failed"] += 1
                
                logger.error(f"Unexpected error in Kinesis {operation_name}: {str(e)}", exc_info=True)
                raise KinesisStreamingError(f"Unexpected error: {str(e)}")
        
        # This should not be reached, but just in case
        if last_exception:
            raise KinesisStreamingError(f"Operation failed after {retries} retries: {str(last_exception)}")
        else:
            raise KinesisStreamingError(f"Operation failed after {retries} retries")
    
    def list_streams(self, limit: int = 100) -> List[str]:
        """
        List Kinesis streams.
        
        Args:
            limit: Maximum number of streams to return
            
        Returns:
            List of stream names
            
        Raises:
            KinesisStreamingError: If an error occurs while listing streams
        """
        # Validate inputs
        if limit <= 0:
            raise KinesisValidationError("limit must be positive")
        
        logger.info(f"Listing Kinesis streams (limit: {limit})")
        
        def list_streams_operation():
            paginator = self.kinesis_client.get_paginator('list_streams')
            
            streams = []
            for page in paginator.paginate(PaginationConfig={'MaxItems': limit}):
                streams.extend(page.get('StreamNames', []))
                
                # Stop if we've reached the limit
                if len(streams) >= limit:
                    streams = streams[:limit]
                    break
            
            return streams
        
        streams = self._execute_with_retry("list_streams", list_streams_operation)
        logger.info(f"Found {len(streams)} Kinesis streams")
        return streams
    
    def describe_stream(self, stream_name: str) -> Dict[str, Any]:
        """
        Describe a Kinesis stream.
        
        Args:
            stream_name: Name of the stream
            
        Returns:
            Stream description
            
        Raises:
            KinesisValidationError: If input validation fails
            KinesisStreamingError: If an error occurs while describing the stream
        """
        # Validate inputs
        if not stream_name:
            raise KinesisValidationError("stream_name cannot be empty")
        
        logger.info(f"Describing Kinesis stream {stream_name}")
        
        def describe_stream_operation():
            paginator = self.kinesis_client.get_paginator('describe_stream')
            
            stream_description = None
            for page in paginator.paginate(StreamName=stream_name):
                if 'StreamDescription' in page:
                    stream_description = page['StreamDescription']
                    break
            
            return stream_description
        
        stream_description = self._execute_with_retry("describe_stream", describe_stream_operation, stream_name)
        
        if not stream_description:
            raise KinesisStreamingError(f"Stream {stream_name} not found")
        
        return stream_description
    
    def create_stream(self, stream_name: str, shard_count: int = 1) -> Dict[str, Any]:
        """
        Create a Kinesis stream.
        
        Args:
            stream_name: Name of the stream
            shard_count: Number of shards
            
        Returns:
            Stream creation response
            
        Raises:
            KinesisValidationError: If input validation fails
            KinesisStreamingError: If an error occurs while creating the stream
        """
        # Validate inputs
        if not stream_name:
            raise KinesisValidationError("stream_name cannot be empty")
        
        if shard_count <= 0:
            raise KinesisValidationError("shard_count must be positive")
        
        logger.info(f"Creating Kinesis stream {stream_name} with {shard_count} shards")
        
        def create_stream_operation():
            return self.kinesis_client.create_stream(
                StreamName=stream_name,
                ShardCount=shard_count
            )
        
        response = self._execute_with_retry("create_stream", create_stream_operation, stream_name)
        
        # Wait for stream to become active
        logger.info(f"Waiting for stream {stream_name} to become active")
        
        def wait_for_stream_active():
            waiter = self.kinesis_client.get_waiter('stream_exists')
            waiter.wait(StreamName=stream_name)
            return {"status": "active"}
        
        wait_response = self._execute_with_retry("wait_for_stream_active", wait_for_stream_active, stream_name)
        
        return {
            "stream_name": stream_name,
            "shard_count": shard_count,
            "status": wait_response.get("status", "unknown")
        }
    
    def delete_stream(self, stream_name: str) -> Dict[str, Any]:
        """
        Delete a Kinesis stream.
        
        Args:
            stream_name: Name of the stream
            
        Returns:
            Stream deletion response
            
        Raises:
            KinesisValidationError: If input validation fails
            KinesisStreamingError: If an error occurs while deleting the stream
        """
        # Validate inputs
        if not stream_name:
            raise KinesisValidationError("stream_name cannot be empty")
        
        logger.info(f"Deleting Kinesis stream {stream_name}")
        
        def delete_stream_operation():
            return self.kinesis_client.delete_stream(
                StreamName=stream_name,
                EnforceConsumerDeletion=True
            )
        
        response = self._execute_with_retry("delete_stream", delete_stream_operation, stream_name)
        
        return {
            "stream_name": stream_name,
            "status": "deleting"
        }
    
    def put_record(self, stream_name: str, data: bytes, partition_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Put a record into a Kinesis stream.
        
        Args:
            stream_name: Name of the stream
            data: Record data
            partition_key: Partition key (optional)
            
        Returns:
            Put record response
            
        Raises:
            KinesisValidationError: If input validation fails
            KinesisStreamingError: If an error occurs while putting the record
        """
        # Validate inputs
        if not stream_name:
            raise KinesisValidationError("stream_name cannot be empty")
        
        if not data:
            raise KinesisValidationError("data cannot be empty")
        
        if len(data) > 1024 * 1024:  # 1MB limit
            raise KinesisValidationError("data exceeds 1MB limit")
        
        # Generate partition key if not provided
        if not partition_key:
            partition_key = str(uuid.uuid4())
        
        logger.debug(f"Putting record into Kinesis stream {stream_name}")
        
        def put_record_operation():
            response = self.kinesis_client.put_record(
                StreamName=stream_name,
                Data=data,
                PartitionKey=partition_key
            )
            
            # Update metrics
            self.metrics["bytes_sent"] += len(data)
            self.metrics["records_sent"] += 1
            
            if stream_name in self.metrics["stream_operations"]:
                self.metrics["stream_operations"][stream_name]["bytes_sent"] += len(data)
                self.metrics["stream_operations"][stream_name]["records_sent"] += 1
            
            return response
        
        response = self._execute_with_retry("put_record", put_record_operation, stream_name)
        
        return response
    
    def put_records(self, stream_name: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Put multiple records into a Kinesis stream.
        
        Args:
            stream_name: Name of the stream
            records: List of records, each with 'Data' and 'PartitionKey'
            
        Returns:
            Put records response
            
        Raises:
            KinesisValidationError: If input validation fails
            KinesisStreamingError: If an error occurs while putting the records
        """
        # Validate inputs
        if not stream_name:
            raise KinesisValidationError("stream_name cannot be empty")
        
        if not records:
            raise KinesisValidationError("records cannot be empty")
        
        if len(records) > 500:  # 500 records limit
            raise KinesisValidationError("records exceeds 500 limit")
        
        # Validate record size
        total_size = 0
        for record in records:
            if 'Data' not in record:
                raise KinesisValidationError("record missing 'Data'")
            
            if 'PartitionKey' not in record:
                record['PartitionKey'] = str(uuid.uuid4())
            
            data_size = len(record['Data'])
            if data_size > 1024 * 1024:  # 1MB limit
                raise KinesisValidationError(f"record data exceeds 1MB limit: {data_size} bytes")
            
            total_size += data_size
        
        if total_size > 5 * 1024 * 1024:  # 5MB total limit
            raise KinesisValidationError(f"total records size exceeds 5MB limit: {total_size} bytes")
        
        logger.debug(f"Putting {len(records)} records into Kinesis stream {stream_name}")
        
        def put_records_operation():
            response = self.kinesis_client.put_records(
                StreamName=stream_name,
                Records=records
            )
            
            # Update metrics
            self.metrics["bytes_sent"] += total_size
            self.metrics["records_sent"] += len(records)
            
            if stream_name in self.metrics["stream_operations"]:
                self.metrics["stream_operations"][stream_name]["bytes_sent"] += total_size
                self.metrics["stream_operations"][stream_name]["records_sent"] += len(records)
            
            # Check for failed records
            failed_count = response.get('FailedRecordCount', 0)
            if failed_count > 0:
                logger.warning(f"{failed_count} records failed to be put into stream {stream_name}")
            
            return response
        
        response = self._execute_with_retry("put_records", put_records_operation, stream_name)
        
        return response
    
    def get_shard_iterator(self, stream_name: str, shard_id: str, iterator_type: str = 'TRIM_HORIZON',
                          starting_sequence_number: Optional[str] = None,
                          timestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        Get a shard iterator for a Kinesis stream.
        
        Args:
            stream_name: Name of the stream
            shard_id: Shard ID
            iterator_type: Iterator type ('TRIM_HORIZON', 'LATEST', 'AT_SEQUENCE_NUMBER', 'AFTER_SEQUENCE_NUMBER', 'AT_TIMESTAMP')
            starting_sequence_number: Starting sequence number (required for 'AT_SEQUENCE_NUMBER' and 'AFTER_SEQUENCE_NUMBER')
            timestamp: Timestamp (required for 'AT_TIMESTAMP')
            
        Returns:
            Shard iterator response
            
        Raises:
            KinesisValidationError: If input validation fails
            KinesisStreamingError: If an error occurs while getting the shard iterator
        """
        # Validate inputs
        if not stream_name:
            raise KinesisValidationError("stream_name cannot be empty")
        
        if not shard_id:
            raise KinesisValidationError("shard_id cannot be empty")
        
        if iterator_type not in ('TRIM_HORIZON', 'LATEST', 'AT_SEQUENCE_NUMBER', 'AFTER_SEQUENCE_NUMBER', 'AT_TIMESTAMP'):
            raise KinesisValidationError("invalid iterator_type")
        
        if iterator_type in ('AT_SEQUENCE_NUMBER', 'AFTER_SEQUENCE_NUMBER') and not starting_sequence_number:
            raise KinesisValidationError("starting_sequence_number required for iterator_type")
        
        if iterator_type == 'AT_TIMESTAMP' and timestamp is None:
            raise KinesisValidationError("timestamp required for iterator_type 'AT_TIMESTAMP'")
        
        logger.debug(f"Getting shard iterator for stream {stream_name}, shard {shard_id}, type {iterator_type}")
        
        def get_shard_iterator_operation():
            params = {
                'StreamName': stream_name,
                'ShardId': shard_id,
                'ShardIteratorType': iterator_type
            }
            
            if iterator_type in ('AT_SEQUENCE_NUMBER', 'AFTER_SEQUENCE_NUMBER'):
                params['StartingSequenceNumber'] = starting_sequence_number
            
            if iterator_type == 'AT_TIMESTAMP':
                params['Timestamp'] = timestamp
            
            return self.kinesis_client.get_shard_iterator(**params)
        
        response = self._execute_with_retry("get_shard_iterator", get_shard_iterator_operation, stream_name)
        
        return response
    
    def get_records(self, shard_iterator: str, limit: int = 1000) -> Dict[str, Any]:
        """
        Get records from a Kinesis stream.
        
        Args:
            shard_iterator: Shard iterator
            limit: Maximum number of records to return
            
        Returns:
            Get records response
            
        Raises:
            KinesisValidationError: If input validation fails
            KinesisStreamingError: If an error occurs while getting the records
        """
        # Validate inputs
        if not shard_iterator:
            raise KinesisValidationError("shard_iterator cannot be empty")
        
        if limit <= 0 or limit > 10000:
            raise KinesisValidationError("limit must be between 1 and 10000")
        
        logger.debug(f"Getting records with limit {limit}")
        
        def get_records_operation():
            response = self.kinesis_client.get_records(
                ShardIterator=shard_iterator,
                Limit=limit
            )
            
            # Update metrics
            records = response.get('Records', [])
            bytes_received = sum(len(record.get('Data', b'')) for record in records)
            
            self.metrics["bytes_received"] += bytes_received
            self.metrics["records_received"] += len(records)
            
            # We don't know the stream name here, so we can't update stream-specific metrics
            
            return response
        
        response = self._execute_with_retry("get_records", get_records_operation)
        
        return response
    
    def register_stream_consumer(self, stream_name: str, consumer_name: str) -> Dict[str, Any]:
        """
        Register a consumer for enhanced fan-out.
        
        Args:
            stream_name: Name of the stream
            consumer_name: Name of the consumer
            
        Returns:
            Register consumer response
            
        Raises:
            KinesisValidationError: If input validation fails
            KinesisStreamingError: If an error occurs while registering the consumer
        """
        # Validate inputs
        if not stream_name:
            raise KinesisValidationError("stream_name cannot be empty")
        
        if not consumer_name:
            raise KinesisValidationError("consumer_name cannot be empty")
        
        logger.info(f"Registering consumer {consumer_name} for stream {stream_name}")
        
        # Get stream ARN
        stream_description = self.describe_stream(stream_name)
        stream_arn = stream_description.get('StreamARN')
        
        if not stream_arn:
            raise KinesisStreamingError(f"Could not get ARN for stream {stream_name}")
        
        def register_consumer_operation():
            return self.kinesis_client.register_stream_consumer(
                StreamARN=stream_arn,
                ConsumerName=consumer_name
            )
        
        response = self._execute_with_retry("register_stream_consumer", register_consumer_operation, stream_name)
        
        return response.get('Consumer', {})
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics for Kinesis operations.
        
        Returns:
            Metrics for Kinesis operations
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
            "bytes_sent": self.metrics["bytes_sent"],
            "bytes_received": self.metrics["bytes_received"],
            "records_sent": self.metrics["records_sent"],
            "records_received": self.metrics["records_received"],
            "stream_operations": self.metrics["stream_operations"]
        }
    
    def reset_metrics(self) -> None:
        """Reset metrics for Kinesis operations."""
        self.metrics = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "retried_operations": 0,
            "operation_latencies": [],
            "throttling_count": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
            "records_sent": 0,
            "records_received": 0,
            "stream_operations": {}
        }
        
        logger.info("Kinesis metrics reset")


class KinesisStreamReader:
    """High-performance streaming reader from Kinesis streams."""
    
    def __init__(self, stream_name: str, config: Optional[KinesisStreamingConfig] = None):
        """
        Initialize Kinesis stream reader.
        
        Args:
            stream_name: Name of the Kinesis stream
            config: Streaming configuration
        """
        self.stream_name = stream_name
        self.config = config or KinesisStreamingConfig()
        
        # Initialize Kinesis client
        self.kinesis_client = KinesisClient(
            region_name=self.config.region_name,
            max_retries=self.config.max_retries,
            base_delay=self.config.base_delay,
            max_delay=self.config.max_delay
        )
        
        # Performance profiler
        self.profiler = PerformanceProfiler()
        
        # Stream state
        self.shards = []
        self.shard_iterators = {}
        self.last_sequence_numbers = {}
        self._total_bytes_read = 0
        self._total_records_read = 0
        self._start_time = None
        
        # Enhanced fan-out consumer
        self.consumer_arn = None
        if self.config.use_enhanced_fan_out and self.config.consumer_name:
            self._register_consumer()
    
    def __enter__(self):
        """Context manager entry."""
        self._start_time = time.time()
        self._initialize_shards()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass
    
    def _initialize_shards(self):
        """Initialize shards and shard iterators."""
        try:
            # Get stream description
            stream_description = self.kinesis_client.describe_stream(self.stream_name)
            
            # Get shards
            self.shards = stream_description.get('Shards', [])
            
            logger.info(f"Initialized {len(self.shards)} shards for stream {self.stream_name}")
            
            # Get shard iterators
            for shard in self.shards:
                shard_id = shard['ShardId']
                
                # Get shard iterator
                iterator_response = self.kinesis_client.get_shard_iterator(
                    stream_name=self.stream_name,
                    shard_id=shard_id,
                    iterator_type=self.config.shard_iterator_type
                )
                
                shard_iterator = iterator_response.get('ShardIterator')
                if shard_iterator:
                    self.shard_iterators[shard_id] = shard_iterator
            
            logger.info(f"Initialized {len(self.shard_iterators)} shard iterators for stream {self.stream_name}")
            
        except Exception as e:
            logger.error(f"Error initializing shards: {e}", exc_info=True)
            raise KinesisStreamingError(f"Failed to initialize shards: {e}")
    
    def _register_consumer(self):
        """Register enhanced fan-out consumer."""
        try:
            consumer = self.kinesis_client.register_stream_consumer(
                stream_name=self.stream_name,
                consumer_name=self.config.consumer_name
            )
            
            self.consumer_arn = consumer.get('ConsumerARN')
            logger.info(f"Registered consumer {self.config.consumer_name} for stream {self.stream_name}")
            
        except Exception as e:
            logger.warning(f"Error registering consumer: {e}")
            self.consumer_arn = None
    
    def stream_records(self) -> Iterator[Tuple[Dict[str, Any], bytes]]:
        """
        Stream records from Kinesis.
        
        Yields:
            Tuple of (record metadata, record data)
        """
        if not self.shards or not self.shard_iterators:
            logger.warning("No shards or shard iterators available")
            return
        
        # Process each shard
        for shard_id, shard_iterator in list(self.shard_iterators.items()):
            if not shard_iterator:
                continue
            
            # Get records from shard
            try:
                with self.profiler.context_timer("get_records"):
                    response = self.kinesis_client.get_records(
                        shard_iterator=shard_iterator,
                        limit=self.config.batch_size
                    )
                
                # Process records
                records = response.get('Records', [])
                self._total_records_read += len(records)
                
                for record in records:
                    # Extract data and metadata
                    data = record.get('Data', b'')
                    self._total_bytes_read += len(data)
                    
                    # Create metadata
                    metadata = {
                        'sequence_number': record.get('SequenceNumber'),
                        'partition_key': record.get('PartitionKey'),
                        'timestamp': record.get('ApproximateArrivalTimestamp'),
                        'shard_id': shard_id
                    }
                    
                    # Update last sequence number
                    if 'SequenceNumber' in record:
                        self.last_sequence_numbers[shard_id] = record['SequenceNumber']
                    
                    yield metadata, data
                
                # Update shard iterator
                next_shard_iterator = response.get('NextShardIterator')
                if next_shard_iterator:
                    self.shard_iterators[shard_id] = next_shard_iterator
                else:
                    # Shard is closed
                    logger.info(f"Shard {shard_id} is closed")
                    del self.shard_iterators[shard_id]
                
            except Exception as e:
                logger.error(f"Error streaming records from shard {shard_id}: {e}", exc_info=True)
                # Continue with next shard
    
    def stream_records_continuous(self, max_iterations: Optional[int] = None,
                                 sleep_time: float = 1.0) -> Iterator[Tuple[Dict[str, Any], bytes]]:
        """
        Stream records continuously from Kinesis.
        
        Args:
            max_iterations: Maximum number of iterations (None for infinite)
            sleep_time: Time to sleep between iterations when no records are available
            
        Yields:
            Tuple of (record metadata, record data)
        """
        iteration = 0
        
        while max_iterations is None or iteration < max_iterations:
            # Check if we have any active shards
            if not self.shard_iterators:
                logger.info("No active shards remaining")
                break
            
            # Stream records
            records_found = False
            for metadata, data in self.stream_records():
                records_found = True
                yield metadata, data
            
            # Sleep if no records were found
            if not records_found:
                time.sleep(sleep_time)
            
            iteration += 1
    
    def stream_to_maif(self, output_path: str, max_iterations: Optional[int] = None) -> Dict[str, Any]:
        """
        Stream records from Kinesis to a MAIF file.
        
        Args:
            output_path: Path to output MAIF file
            max_iterations: Maximum number of iterations (None for infinite)
            
        Returns:
            Streaming statistics
        """
        manifest_path = output_path.replace('.maif', '_manifest.json')
        
        with MAIFStreamWriter(output_path, self.config) as writer:
            # Stream records
            for metadata, data in self.stream_records_continuous(max_iterations):
                # Determine block type based on metadata
                block_type = "binary_data"  # Default
                
                # Try to detect content type
                if data.startswith(b'{') and data.endswith(b'}'):
                    block_type = "json_data"
                elif data.startswith(b'<') and data.endswith(b'>'):
                    block_type = "xml_data"
                
                # Write block
                writer.write_block(data, block_type, metadata)
            
            # Finalize MAIF file
            writer.finalize(manifest_path)
        
        return self.get_stats()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get streaming statistics."""
        elapsed = time.time() - self._start_time if self._start_time else 0.001
        throughput_mbps = (self._total_bytes_read / (1024 * 1024)) / elapsed if elapsed > 0 else 0
        
        return {
            "stream_name": self.stream_name,
            "total_records_read": self._total_records_read,
            "total_bytes_read": self._total_bytes_read,
            "elapsed_seconds": elapsed,
            "throughput_mbps": throughput_mbps,
            "active_shards": len(self.shard_iterators),
            "total_shards": len(self.shards),
            "enhanced_fan_out": self.config.use_enhanced_fan_out,
            "consumer_name": self.config.consumer_name,
            "consumer_arn": self.consumer_arn,
            "profiler_stats": self.profiler.get_stats()
        }


class KinesisStreamWriter:
    """High-performance streaming writer to Kinesis streams."""
    
    def __init__(self, stream_name: str, config: Optional[KinesisStreamingConfig] = None):
        """
        Initialize Kinesis stream writer.
        
        Args:
            stream_name: Name of the Kinesis stream
            config: Streaming configuration
        """
        self.stream_name = stream_name
        self.config = config or KinesisStreamingConfig()
        
        # Initialize Kinesis client
        self.kinesis_client = KinesisClient(
            region_name=self.config.region_name,
            max_retries=self.config.max_retries,
            base_delay=self.config.base_delay,
            max_delay=self.config.max_delay
        )
        
        # Performance profiler
        self.profiler = PerformanceProfiler()
        
        # Stream state
        self._total_bytes_written = 0
        self._total_records_written = 0
        self._start_time = None
        
        # Batch buffer
        self._batch_buffer = []
        self._batch_size_bytes = 0
    
    def __enter__(self):
        """Context manager entry."""
        self._start_time = time.time()
        self._ensure_stream_exists()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Flush any remaining records
        self._flush_batch()
    
    def _ensure_stream_exists(self):
        """Ensure the Kinesis stream exists."""
        try:
            # Check if stream exists
            self.kinesis_client.describe_stream(self.stream_name)
            logger.info(f"Stream {self.stream_name} exists")
        except KinesisStreamingError:
            # Create stream
            logger.info(f"Creating stream {self.stream_name}")
            self.kinesis_client.create_stream(self.stream_name)
    
    def write_record(self, data: bytes, partition_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Write a record to the Kinesis stream.
        
        Args:
            data: Record data
            partition_key: Partition key (optional)
            
        Returns:
            Write response
        """
        # Validate data size
        if len(data) > self.config.max_record_size_bytes:
            raise KinesisValidationError(f"Record size {len(data)} exceeds maximum {self.config.max_record_size_bytes}")
        
        # Generate partition key if not provided
        if not partition_key:
            partition_key = str(uuid.uuid4())
        
        # Check if we should batch
        if len(self._batch_buffer) < self.config.batch_size and \
           self._batch_size_bytes + len(data) < self.config.max_batch_size_bytes:
            # Add to batch
            self._batch_buffer.append({
                'Data': data,
                'PartitionKey': partition_key
            })
            self._batch_size_bytes += len(data)
            
            # Check if batch is full
            if len(self._batch_buffer) >= self.config.batch_size or \
               self._batch_size_bytes >= self.config.max_batch_size_bytes:
                return self._flush_batch()
            
            # Return empty response for now
            return {}
        else:
            # Flush current batch
            self._flush_batch()
            
            # Write single record
            with self.profiler.context_timer("put_record"):
                response = self.kinesis_client.put_record(
                    stream_name=self.stream_name,
                    data=data,
                    partition_key=partition_key
                )
            
            # Update metrics
            self._total_bytes_written += len(data)
            self._total_records_written += 1
            
            return response
    
    def _flush_batch(self) -> Dict[str, Any]:
        """Flush the current batch of records."""
        if not self._batch_buffer:
            return {}
        
        try:
            with self.profiler.context_timer("put_records"):
                response = self.kinesis_client.put_records(
                    stream_name=self.stream_name,
                    records=self._batch_buffer
                )
            
            # Update metrics
            self._total_bytes_written += self._batch_size_bytes
            self._total_records_written += len(self._batch_buffer)
            
            # Check for failed records
            failed_count = response.get('FailedRecordCount', 0)
            if failed_count > 0:
                logger.warning(f"{failed_count} records failed to be written")
                
                # Retry failed records
                failed_records = []
                for i, record in enumerate(response.get('Records', [])):
                    if 'ErrorCode' in record:
                        failed_records.append(self._batch_buffer[i])
                
                if failed_records:
                    logger.info(f"Retrying {len(failed_records)} failed records")
                    self._batch_buffer = failed_records
                    self._batch_size_bytes = sum(len(record['Data']) for record in failed_records)
                    return self._flush_batch()
            
            # Clear batch
            self._batch_buffer = []
            self._batch_size_bytes = 0
            
            return response
            
        except Exception as e:
            logger.error(f"Error flushing batch: {e}", exc_info=True)
            return {"error": str(e)}
    
    def stream_from_maif(self, maif_path: str) -> Dict[str, Any]:
        """
        Stream blocks from a MAIF file to Kinesis.
        
        Args:
            maif_path: Path to MAIF file
            
        Returns:
            Streaming statistics
        """
        with MAIFStreamReader(maif_path, self.config) as reader:
            # Stream blocks
            for block_type, block_data in reader.stream_blocks():
                # Create partition key based on block type
                partition_key = f"{block_type}_{uuid.uuid4().hex[:8]}"
                
                # Write record
                self.write_record(block_data, partition_key)
        
        # Flush any remaining records
        self._flush_batch()
        
        return self.get_stats()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get streaming statistics."""
        elapsed = time.time() - self._start_time if self._start_time else 0.001
        throughput_mbps = (self._total_bytes_written / (1024 * 1024)) / elapsed if elapsed > 0 else 0
        
        return {
            "stream_name": self.stream_name,
            "total_records_written": self._total_records_written,
            "total_bytes_written": self._total_bytes_written,
            "elapsed_seconds": elapsed,
            "throughput_mbps": throughput_mbps,
            "batch_buffer_size": len(self._batch_buffer),
            "batch_buffer_bytes": self._batch_size_bytes,
            "profiler_stats": self.profiler.get_stats(),
            "kinesis_metrics": self.kinesis_client.get_metrics()
        }


class KinesisMAIFProcessor:
    """High-level processor for streaming MAIF data to/from Kinesis."""
    
    def __init__(self, config: Optional[KinesisStreamingConfig] = None):
        """
        Initialize Kinesis MAIF processor.
        
        Args:
            config: Streaming configuration
        """
        self.config = config or KinesisStreamingConfig()
        self.profiler = PerformanceProfiler()
    
    def maif_to_kinesis(self, maif_path: str, stream_name: str) -> Dict[str, Any]:
        """
        Stream MAIF file to Kinesis.
        
        Args:
            maif_path: Path to MAIF file
            stream_name: Name of Kinesis stream
            
        Returns:
            Processing statistics
        """
        with self.profiler.context_timer("maif_to_kinesis"):
            with KinesisStreamWriter(stream_name, self.config) as writer:
                stats = writer.stream_from_maif(maif_path)
        
        return {
            "operation": "maif_to_kinesis",
            "maif_path": maif_path,
            "stream_name": stream_name,
            "stats": stats,
            "profiler_stats": self.profiler.get_stats()
        }
    
    def kinesis_to_maif(self, stream_name: str, output_path: str,
                       max_iterations: Optional[int] = None) -> Dict[str, Any]:
        """
        Stream Kinesis to MAIF file.
        
        Args:
            stream_name: Name of Kinesis stream
            output_path: Path to output MAIF file
            max_iterations: Maximum number of iterations (None for infinite)
            
        Returns:
            Processing statistics
        """
        with self.profiler.context_timer("kinesis_to_maif"):
            with KinesisStreamReader(stream_name, self.config) as reader:
                stats = reader.stream_to_maif(output_path, max_iterations)
        
        return {
            "operation": "kinesis_to_maif",
            "stream_name": stream_name,
            "output_path": output_path,
            "stats": stats,
            "profiler_stats": self.profiler.get_stats()
        }
    
    def process_kinesis_stream(self, stream_name: str, processor_func,
                              max_iterations: Optional[int] = None) -> Dict[str, Any]:
        """
        Process Kinesis stream with custom function.
        
        Args:
            stream_name: Name of Kinesis stream
            processor_func: Function to process each record (metadata, data)
            max_iterations: Maximum number of iterations (None for infinite)
            
        Returns:
            Processing statistics
        """
        results = []
        
        with self.profiler.context_timer("process_kinesis_stream"):
            with KinesisStreamReader(stream_name, self.config) as reader:
                # Process records
                for metadata, data in reader.stream_records_continuous(max_iterations):
                    with self.profiler.context_timer("process_record"):
                        result = processor_func(metadata, data)
                        results.append(result)
        
        return {
            "operation": "process_kinesis_stream",
            "stream_name": stream_name,
            "records_processed": len(results),
            "stats": reader.get_stats(),
            "profiler_stats": self.profiler.get_stats()
        }


# Export classes
__all__ = [
    'KinesisStreamingConfig',
    'KinesisClient',
    'KinesisStreamReader',
    'KinesisStreamWriter',
    'KinesisMAIFProcessor',
    'KinesisStreamingError',
    'KinesisConnectionError',
    'KinesisThrottlingError',
    'KinesisValidationError'
]