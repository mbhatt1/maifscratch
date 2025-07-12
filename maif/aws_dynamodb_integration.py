"""
AWS DynamoDB Integration for MAIF
================================

Provides integration with AWS DynamoDB for NoSQL database operations.
"""

import json
import time
import logging
import datetime
import decimal
import hashlib
import threading
from typing import Dict, List, Optional, Any, Union, Tuple, Set
import boto3
from botocore.exceptions import ClientError, ConnectionError, EndpointConnectionError

# Configure logger
logger = logging.getLogger(__name__)


# Helper class to handle Decimal types in DynamoDB
class DecimalEncoder(json.JSONEncoder):
    """Helper class to convert Decimal to float for JSON serialization."""
    def default(self, o):
        if isinstance(o, decimal.Decimal):
            # Handle special cases
            if o.is_nan():
                return None
            elif o.is_infinite():
                return None
            # Convert to int if it's a whole number, else float
            return int(o) if o % 1 == 0 else float(o)
        return super(DecimalEncoder, self).default(o)


class DynamoDBError(Exception):
    """Base exception for DynamoDB integration errors."""
    pass


class DynamoDBConnectionError(DynamoDBError):
    """Exception for DynamoDB connection errors."""
    pass


class DynamoDBThrottlingError(DynamoDBError):
    """Exception for DynamoDB throttling errors."""
    pass


class DynamoDBValidationError(DynamoDBError):
    """Exception for DynamoDB validation errors."""
    pass


class DynamoDBPermissionError(DynamoDBError):
    """Exception for DynamoDB permission errors."""
    pass


class DynamoDBClient:
    """Client for AWS DynamoDB service with production-ready features."""
    
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
                max_retries: int = 3, base_delay: float = 0.5, max_delay: float = 5.0,
                table_cache_ttl: int = 3600):
        """
        Initialize DynamoDB client.
        
        Args:
            region_name: AWS region name
            profile_name: AWS profile name (optional)
            max_retries: Maximum number of retries for transient errors
            base_delay: Base delay for exponential backoff (seconds)
            max_delay: Maximum delay for exponential backoff (seconds)
            table_cache_ttl: Time-to-live for table cache entries (seconds)
            
        Raises:
            DynamoDBConnectionError: If unable to initialize the DynamoDB client
        """
        # Validate inputs
        if not region_name:
            raise DynamoDBValidationError("region_name cannot be empty")
        
        if max_retries < 0:
            raise DynamoDBValidationError("max_retries must be non-negative")
        
        if base_delay <= 0 or max_delay <= 0:
            raise DynamoDBValidationError("base_delay and max_delay must be positive")
        
        if table_cache_ttl <= 0:
            raise DynamoDBValidationError("table_cache_ttl must be positive")
        
        # Initialize AWS session
        try:
            logger.info(f"Initializing DynamoDB client in region {region_name}")
            session = boto3.Session(profile_name=profile_name, region_name=region_name)
            
            # Initialize DynamoDB clients
            self.dynamodb_client = session.client('dynamodb')
            self.dynamodb_resource = session.resource('dynamodb')
            
            # Retry configuration
            self.max_retries = max_retries
            self.base_delay = base_delay
            self.max_delay = max_delay
            
            # Cache for table info
            self.table_cache: Dict[str, Any] = {}
            self.table_cache_expiry: Dict[str, float] = {}
            self.table_cache_ttl = table_cache_ttl
            
            # Metrics for monitoring
            self.metrics = {
                "total_operations": 0,
                "successful_operations": 0,
                "failed_operations": 0,
                "retried_operations": 0,
                "operation_latencies": [],
                "read_capacity_consumed": 0,
                "write_capacity_consumed": 0,
                "throttling_count": 0,
                "table_operations": {}
            }
            
            # Thread safety
            self._lock = threading.RLock()
            
            logger.info("DynamoDB client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize DynamoDB client: {e}", exc_info=True)
            raise DynamoDBConnectionError(f"Failed to initialize DynamoDB client: {e}")
    
    def _execute_with_retry(self, operation_name: str, operation_func, table_name: Optional[str] = None, *args, **kwargs):
        """
        Execute an operation with exponential backoff retry for transient errors.
        
        Args:
            operation_name: Name of the operation for logging
            operation_func: Function to execute
            table_name: Name of the table (for metrics)
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Result of the operation
            
        Raises:
            DynamoDBError: For non-retryable errors or if max retries exceeded
        """
        start_time = time.time()
        with self._lock:
            self.metrics["total_operations"] += 1
            
            # Track table-specific operations
            if table_name:
                if table_name not in self.metrics["table_operations"]:
                    self.metrics["table_operations"][table_name] = {
                        "total": 0,
                        "successful": 0,
                        "failed": 0,
                        "retried": 0,
                        "throttled": 0
                    }
                self.metrics["table_operations"][table_name]["total"] += 1
        
        retries = 0
        last_exception = None
        
        while retries <= self.max_retries:
            try:
                logger.debug(f"Executing DynamoDB {operation_name} (attempt {retries + 1}/{self.max_retries + 1})")
                result = operation_func(*args, **kwargs)
                
                # Record success metrics
                with self._lock:
                    self.metrics["successful_operations"] += 1
                    latency = time.time() - start_time
                    self.metrics["operation_latencies"].append(latency)
                    
                    # Track table-specific success
                    if table_name:
                        self.metrics["table_operations"][table_name]["successful"] += 1
                    
                    # Track consumed capacity
                    if isinstance(result, dict) and 'ConsumedCapacity' in result:
                        capacity = result['ConsumedCapacity']
                        if isinstance(capacity, list):
                            for cap in capacity:
                                self._track_consumed_capacity(cap)
                        else:
                            self._track_consumed_capacity(capacity)
                    
                    # Trim latencies list if it gets too large
                    if len(self.metrics["operation_latencies"]) > 1000:
                        self.metrics["operation_latencies"] = self.metrics["operation_latencies"][-1000:]
                
                logger.debug(f"DynamoDB {operation_name} completed successfully in {latency:.2f}s")
                return result
                
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')
                error_message = e.response.get('Error', {}).get('Message', '')
                
                # Check if this is a retryable error
                if error_code in self.RETRYABLE_ERRORS and retries < self.max_retries:
                    retries += 1
                    with self._lock:
                        self.metrics["retried_operations"] += 1
                        
                        # Track throttling
                        if error_code == 'ProvisionedThroughputExceededException' or error_code == 'ThrottlingException':
                            self.metrics["throttling_count"] += 1
                            if table_name:
                                self.metrics["table_operations"][table_name]["throttled"] += 1
                        
                        # Track table-specific retries
                        if table_name:
                            self.metrics["table_operations"][table_name]["retried"] += 1
                    
                    delay = min(self.max_delay, self.base_delay * (2 ** retries))
                    
                    logger.warning(
                        f"DynamoDB {operation_name} failed with retryable error: {error_code} - {error_message}. "
                        f"Retrying in {delay:.2f}s (attempt {retries}/{self.max_retries})"
                    )
                    
                    time.sleep(delay)
                    last_exception = e
                else:
                    # Non-retryable error or max retries exceeded
                    with self._lock:
                        self.metrics["failed_operations"] += 1
                        
                        # Track table-specific failures
                        if table_name:
                            self.metrics["table_operations"][table_name]["failed"] += 1
                    
                    if error_code in self.RETRYABLE_ERRORS:
                        logger.error(
                            f"DynamoDB {operation_name} failed after {retries} retries: {error_code} - {error_message}",
                            exc_info=True
                        )
                        raise DynamoDBThrottlingError(f"{error_code}: {error_message}. Max retries exceeded.")
                    elif error_code in ('AccessDeniedException', 'AccessDenied', 'ResourceNotFoundException'):
                        logger.error(
                            f"DynamoDB {operation_name} failed due to permission or resource error: {error_code} - {error_message}",
                            exc_info=True
                        )
                        raise DynamoDBPermissionError(f"{error_code}: {error_message}")
                    else:
                        logger.error(
                            f"DynamoDB {operation_name} failed with non-retryable error: {error_code} - {error_message}",
                            exc_info=True
                        )
                        raise DynamoDBError(f"{error_code}: {error_message}")
                        
            except (ConnectionError, EndpointConnectionError) as e:
                # Network-related errors
                if retries < self.max_retries:
                    retries += 1
                    with self._lock:
                        self.metrics["retried_operations"] += 1
                        
                        # Track table-specific retries
                        if table_name:
                            self.metrics["table_operations"][table_name]["retried"] += 1
                    
                    delay = min(self.max_delay, self.base_delay * (2 ** retries))
                    
                    logger.warning(
                        f"DynamoDB {operation_name} failed with connection error: {str(e)}. "
                        f"Retrying in {delay:.2f}s (attempt {retries}/{self.max_retries})"
                    )
                    
                    time.sleep(delay)
                    last_exception = e
                else:
                    with self._lock:
                        self.metrics["failed_operations"] += 1
                        
                        # Track table-specific failures
                        if table_name:
                            self.metrics["table_operations"][table_name]["failed"] += 1
                    
                    logger.error(f"DynamoDB {operation_name} failed after {retries} retries due to connection error", exc_info=True)
                    raise DynamoDBConnectionError(f"Connection error: {str(e)}. Max retries exceeded.")
            
            except Exception as e:
                # Unexpected errors
                with self._lock:
                    self.metrics["failed_operations"] += 1
                    
                    # Track table-specific failures
                    if table_name:
                        self.metrics["table_operations"][table_name]["failed"] += 1
                
                logger.error(f"Unexpected error in DynamoDB {operation_name}: {str(e)}", exc_info=True)
                raise DynamoDBError(f"Unexpected error: {str(e)}")
        
        # This should not be reached, but just in case
        if last_exception:
            raise DynamoDBError(f"Operation failed after {retries} retries: {str(last_exception)}")
        else:
            raise DynamoDBError(f"Operation failed after {retries} retries")
    
    def _track_consumed_capacity(self, capacity: Dict[str, Any]):
        """Track consumed capacity for metrics."""
        if 'ReadCapacityUnits' in capacity:
            self.metrics["read_capacity_consumed"] += capacity['ReadCapacityUnits']
        
        if 'WriteCapacityUnits' in capacity:
            self.metrics["write_capacity_consumed"] += capacity['WriteCapacityUnits']
    
    def _clean_table_cache(self):
        """Clean expired entries from table cache."""
        with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, expiry in self.table_cache_expiry.items()
                if current_time > expiry
            ]
            
            for key in expired_keys:
                if key in self.table_cache:
                    del self.table_cache[key]
                if key in self.table_cache_expiry:
                    del self.table_cache_expiry[key]
            
            if expired_keys:
                logger.debug(f"Cleaned table cache: {len(expired_keys)} expired entries removed")
    
    def list_tables(self, limit: int = 100) -> List[str]:
        """
        List DynamoDB tables.
        
        Args:
            limit: Maximum number of tables to return
            
        Returns:
            List of table names
            
        Raises:
            DynamoDBError: If an error occurs while listing tables
        """
        # Validate inputs
        if limit <= 0:
            raise DynamoDBValidationError("limit must be positive")
        
        logger.info(f"Listing DynamoDB tables (limit: {limit})")
        
        def list_tables_operation():
            paginator = self.dynamodb_client.get_paginator('list_tables')
            
            tables = []
            for page in paginator.paginate(PaginationConfig={'MaxItems': limit}):
                tables.extend(page.get('TableNames', []))
                
                # Stop if we've reached the limit
                if len(tables) >= limit:
                    tables = tables[:limit]
                    break
            
            return tables
        
        tables = self._execute_with_retry("list_tables", list_tables_operation)
        logger.info(f"Found {len(tables)} DynamoDB tables")
        return tables
    
    def describe_table(self, table_name: str) -> Dict[str, Any]:
        """
        Describe a DynamoDB table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Table description
            
        Raises:
            DynamoDBValidationError: If input validation fails
            DynamoDBError: If an error occurs while describing the table
        """
        # Validate inputs
        if not table_name:
            raise DynamoDBValidationError("table_name cannot be empty")
        
        # Clean cache periodically
        self._clean_table_cache()
        
        # Check cache
        with self._lock:
            if table_name in self.table_cache:
                logger.debug(f"Returning cached table info for {table_name}")
                return self.table_cache[table_name]
        
        logger.info(f"Describing DynamoDB table {table_name}")
        
        def describe_table_operation():
            response = self.dynamodb_client.describe_table(TableName=table_name)
            return response.get('Table', {})
        
        table_info = self._execute_with_retry("describe_table", describe_table_operation, table_name)
        
        # Cache table info
        with self._lock:
            self.table_cache[table_name] = table_info
            self.table_cache_expiry[table_name] = time.time() + self.table_cache_ttl
        
        return table_info
    
    def create_table(self, table_name: str, key_schema: List[Dict[str, str]],
                    attribute_definitions: List[Dict[str, str]],
                    provisioned_throughput: Dict[str, int],
                    global_secondary_indexes: Optional[List[Dict[str, Any]]] = None,
                    local_secondary_indexes: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Create a DynamoDB table.
        
        Args:
            table_name: Name of the table
            key_schema: Key schema for the table
            attribute_definitions: Attribute definitions
            provisioned_throughput: Provisioned throughput
            global_secondary_indexes: Global secondary indexes (optional)
            local_secondary_indexes: Local secondary indexes (optional)
            
        Returns:
            Table description
            
        Raises:
            DynamoDBValidationError: If input validation fails
            DynamoDBError: If an error occurs while creating the table
        """
        # Validate inputs
        if not table_name:
            raise DynamoDBValidationError("table_name cannot be empty")
        
        if not key_schema:
            raise DynamoDBValidationError("key_schema cannot be empty")
        
        if not attribute_definitions:
            raise DynamoDBValidationError("attribute_definitions cannot be empty")
        
        if not provisioned_throughput:
            raise DynamoDBValidationError("provisioned_throughput cannot be empty")
        
        logger.info(f"Creating DynamoDB table {table_name}")
        
        # Prepare create parameters
        create_params = {
            'TableName': table_name,
            'KeySchema': key_schema,
            'AttributeDefinitions': attribute_definitions,
            'ProvisionedThroughput': provisioned_throughput
        }
        
        if global_secondary_indexes:
            create_params['GlobalSecondaryIndexes'] = global_secondary_indexes
        
        if local_secondary_indexes:
            create_params['LocalSecondaryIndexes'] = local_secondary_indexes
        
        def create_table_operation():
            response = self.dynamodb_client.create_table(**create_params)
            return response.get('TableDescription', {})
        
        table_info = self._execute_with_retry("create_table", create_table_operation, table_name)
        
        # Cache table info
        with self._lock:
            self.table_cache[table_name] = table_info
            self.table_cache_expiry[table_name] = time.time() + self.table_cache_ttl
        
        return table_info
    
    def delete_table(self, table_name: str) -> Dict[str, Any]:
        """
        Delete a DynamoDB table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Table description
            
        Raises:
            DynamoDBValidationError: If input validation fails
            DynamoDBError: If an error occurs while deleting the table
        """
        # Validate inputs
        if not table_name:
            raise DynamoDBValidationError("table_name cannot be empty")
        
        logger.info(f"Deleting DynamoDB table {table_name}")
        
        def delete_table_operation():
            response = self.dynamodb_client.delete_table(TableName=table_name)
            return response.get('TableDescription', {})
        
        table_info = self._execute_with_retry("delete_table", delete_table_operation, table_name)
        
        # Remove from cache
        with self._lock:
            if table_name in self.table_cache:
                del self.table_cache[table_name]
            
            if table_name in self.table_cache_expiry:
                del self.table_cache_expiry[table_name]
        
        return table_info
    
    def put_item(self, table_name: str, item: Dict[str, Any],
                condition_expression: Optional[str] = None,
                expression_attribute_names: Optional[Dict[str, str]] = None,
                expression_attribute_values: Optional[Dict[str, Any]] = None,
                return_values: str = 'NONE',
                return_consumed_capacity: str = 'TOTAL') -> Dict[str, Any]:
        """
        Put an item in a DynamoDB table.
        
        Args:
            table_name: Name of the table
            item: Item to put
            condition_expression: Condition expression (optional)
            expression_attribute_names: Expression attribute names (optional)
            expression_attribute_values: Expression attribute values (optional)
            return_values: Return values option
            return_consumed_capacity: Return consumed capacity option
            
        Returns:
            Response from DynamoDB
            
        Raises:
            DynamoDBValidationError: If input validation fails
            DynamoDBError: If an error occurs while putting the item
        """
        # Validate inputs
        if not table_name:
            raise DynamoDBValidationError("table_name cannot be empty")
        
        if not item:
            raise DynamoDBValidationError("item cannot be empty")
        
        logger.info(f"Putting item in DynamoDB table {table_name}")
        
        # Prepare put parameters
        put_params = {
            'TableName': table_name,
            'Item': item,
            'ReturnValues': return_values,
            'ReturnConsumedCapacity': return_consumed_capacity
        }
        
        if condition_expression:
            put_params['ConditionExpression'] = condition_expression
        
        if expression_attribute_names:
            put_params['ExpressionAttributeNames'] = expression_attribute_names
        
        if expression_attribute_values:
            put_params['ExpressionAttributeValues'] = expression_attribute_values
        
        def put_item_operation():
            return self.dynamodb_client.put_item(**put_params)
        
        return self._execute_with_retry("put_item", put_item_operation, table_name)
    
    def get_item(self, table_name: str, key: Dict[str, Any],
                projection_expression: Optional[str] = None,
                expression_attribute_names: Optional[Dict[str, str]] = None,
                consistent_read: bool = False,
                return_consumed_capacity: str = 'TOTAL') -> Dict[str, Any]:
        """
        Get an item from a DynamoDB table.
        
        Args:
            table_name: Name of the table
            key: Key of the item to get
            projection_expression: Projection expression (optional)
            expression_attribute_names: Expression attribute names (optional)
            consistent_read: Whether to use consistent read
            return_consumed_capacity: Return consumed capacity option
            
        Returns:
            Response from DynamoDB
            
        Raises:
            DynamoDBValidationError: If input validation fails
            DynamoDBError: If an error occurs while getting the item
        """
        # Validate inputs
        if not table_name:
            raise DynamoDBValidationError("table_name cannot be empty")
        
        if not key:
            raise DynamoDBValidationError("key cannot be empty")
        
        logger.info(f"Getting item from DynamoDB table {table_name}")
        
        # Prepare get parameters
        get_params = {
            'TableName': table_name,
            'Key': key,
            'ConsistentRead': consistent_read,
            'ReturnConsumedCapacity': return_consumed_capacity
        }
        
        if projection_expression:
            get_params['ProjectionExpression'] = projection_expression
        
        if expression_attribute_names:
            get_params['ExpressionAttributeNames'] = expression_attribute_names
        
        def get_item_operation():
            return self.dynamodb_client.get_item(**get_params)
        
        return self._execute_with_retry("get_item", get_item_operation, table_name)
    
    def update_item(self, table_name: str, key: Dict[str, Any],
                   update_expression: str,
                   condition_expression: Optional[str] = None,
                   expression_attribute_names: Optional[Dict[str, str]] = None,
                   expression_attribute_values: Optional[Dict[str, Any]] = None,
                   return_values: str = 'NONE',
                   return_consumed_capacity: str = 'TOTAL') -> Dict[str, Any]:
        """
        Update an item in a DynamoDB table.
        
        Args:
            table_name: Name of the table
            key: Key of the item to update
            update_expression: Update expression
            condition_expression: Condition expression (optional)
            expression_attribute_names: Expression attribute names (optional)
            expression_attribute_values: Expression attribute values (optional)
            return_values: Return values option
            return_consumed_capacity: Return consumed capacity option
            
        Returns:
            Response from DynamoDB
            
        Raises:
            DynamoDBValidationError: If input validation fails
            DynamoDBError: If an error occurs while updating the item
        """
        # Validate inputs
        if not table_name:
            raise DynamoDBValidationError("table_name cannot be empty")
        
        if not key:
            raise DynamoDBValidationError("key cannot be empty")
        
        if not update_expression:
            raise DynamoDBValidationError("update_expression cannot be empty")
        
        logger.info(f"Updating item in DynamoDB table {table_name}")
        
        # Prepare update parameters
        update_params = {
            'TableName': table_name,
            'Key': key,
            'UpdateExpression': update_expression,
            'ReturnValues': return_values,
            'ReturnConsumedCapacity': return_consumed_capacity
        }
        
        if condition_expression:
            update_params['ConditionExpression'] = condition_expression
        
        if expression_attribute_names:
            update_params['ExpressionAttributeNames'] = expression_attribute_names
        
        if expression_attribute_values:
            update_params['ExpressionAttributeValues'] = expression_attribute_values
        
        def update_item_operation():
            return self.dynamodb_client.update_item(**update_params)
        
        return self._execute_with_retry("update_item", update_item_operation, table_name)
    
    def delete_item(self, table_name: str, key: Dict[str, Any],
                   condition_expression: Optional[str] = None,
                   expression_attribute_names: Optional[Dict[str, str]] = None,
                   expression_attribute_values: Optional[Dict[str, Any]] = None,
                   return_values: str = 'NONE',
                   return_consumed_capacity: str = 'TOTAL') -> Dict[str, Any]:
        """
        Delete an item from a DynamoDB table.
        
        Args:
            table_name: Name of the table
            key: Key of the item to delete
            condition_expression: Condition expression (optional)
            expression_attribute_names: Expression attribute names (optional)
            expression_attribute_values: Expression attribute values (optional)
            return_values: Return values option
            return_consumed_capacity: Return consumed capacity option
            
        Returns:
            Response from DynamoDB
            
        Raises:
            DynamoDBValidationError: If input validation fails
            DynamoDBError: If an error occurs while deleting the item
        """
        # Validate inputs
        if not table_name:
            raise DynamoDBValidationError("table_name cannot be empty")
        
        if not key:
            raise DynamoDBValidationError("key cannot be empty")
        
        logger.info(f"Deleting item from DynamoDB table {table_name}")
        
        # Prepare delete parameters
        delete_params = {
            'TableName': table_name,
            'Key': key,
            'ReturnValues': return_values,
            'ReturnConsumedCapacity': return_consumed_capacity
        }
        
        if condition_expression:
            delete_params['ConditionExpression'] = condition_expression
        
        if expression_attribute_names:
            delete_params['ExpressionAttributeNames'] = expression_attribute_names
        
        if expression_attribute_values:
            delete_params['ExpressionAttributeValues'] = expression_attribute_values
        
        def delete_item_operation():
            return self.dynamodb_client.delete_item(**delete_params)
        
        return self._execute_with_retry("delete_item", delete_item_operation, table_name)
    
    def query(self, table_name: str, key_condition_expression: str,
             filter_expression: Optional[str] = None,
             projection_expression: Optional[str] = None,
             expression_attribute_names: Optional[Dict[str, str]] = None,
             expression_attribute_values: Optional[Dict[str, Any]] = None,
             index_name: Optional[str] = None,
             limit: Optional[int] = None,
             consistent_read: bool = False,
             scan_index_forward: bool = True,
             return_consumed_capacity: str = 'TOTAL') -> Dict[str, Any]:
        """
        Query a DynamoDB table.
        
        Args:
            table_name: Name of the table
            key_condition_expression: Key condition expression
            filter_expression: Filter expression (optional)
            projection_expression: Projection expression (optional)
            expression_attribute_names: Expression attribute names (optional)
            expression_attribute_values: Expression attribute values (optional)
            index_name: Name of the index to query (optional)
            limit: Maximum number of items to return (optional)
            consistent_read: Whether to use consistent read
            scan_index_forward: Whether to scan index forward
            return_consumed_capacity: Return consumed capacity option
            
        Returns:
            Response from DynamoDB
            
        Raises:
            DynamoDBValidationError: If input validation fails
            DynamoDBError: If an error occurs while querying the table
        """
        # Validate inputs
        if not table_name:
            raise DynamoDBValidationError("table_name cannot be empty")
        
        if not key_condition_expression:
            raise DynamoDBValidationError("key_condition_expression cannot be empty")
        
        logger.info(f"Querying DynamoDB table {table_name}")
        
        # Prepare query parameters
        query_params = {
            'TableName': table_name,
            'KeyConditionExpression': key_condition_expression,
            'ConsistentRead': consistent_read,
            'ScanIndexForward': scan_index_forward,
            'ReturnConsumedCapacity': return_consumed_capacity
        }
        
        if filter_expression:
            query_params['FilterExpression'] = filter_expression
        
        if projection_expression:
            query_params['ProjectionExpression'] = projection_expression
        
        if expression_attribute_names:
            query_params['ExpressionAttributeNames'] = expression_attribute_names
        
        if expression_attribute_values:
            query_params['ExpressionAttributeValues'] = expression_attribute_values
        
        if index_name:
            query_params['IndexName'] = index_name
        
        if limit is not None:
            query_params['Limit'] = limit
        
        def query_operation():
            return self.dynamodb_client.query(**query_params)
        
        return self._execute_with_retry("query", query_operation, table_name)
    
    def scan(self, table_name: str,
            filter_expression: Optional[str] = None,
            projection_expression: Optional[str] = None,
            expression_attribute_names: Optional[Dict[str, str]] = None,
            expression_attribute_values: Optional[Dict[str, Any]] = None,
            index_name: Optional[str] = None,
            limit: Optional[int] = None,
            consistent_read: bool = False,
            return_consumed_capacity: str = 'TOTAL') -> Dict[str, Any]:
        """
        Scan a DynamoDB table.
        
        Args:
            table_name: Name of the table
            filter_expression: Filter expression (optional)
            projection_expression: Projection expression (optional)
            expression_attribute_names: Expression attribute names (optional)
            expression_attribute_values: Expression attribute values (optional)
            index_name: Name of the index to scan (optional)
            limit: Maximum number of items to return (optional)
            consistent_read: Whether to use consistent read
            return_consumed_capacity: Return consumed capacity option
            
        Returns:
            Response from DynamoDB
            
        Raises:
            DynamoDBValidationError: If input validation fails
            DynamoDBError: If an error occurs while scanning the table
        """
        # Validate inputs
        if not table_name:
            raise DynamoDBValidationError("table_name cannot be empty")
        
        logger.info(f"Scanning DynamoDB table {table_name}")
        
        # Prepare scan parameters
        scan_params = {
            'TableName': table_name,
            'ConsistentRead': consistent_read,
            'ReturnConsumedCapacity': return_consumed_capacity
        }
        
        if filter_expression:
            scan_params['FilterExpression'] = filter_expression
        
        if projection_expression:
            scan_params['ProjectionExpression'] = projection_expression
        
        if expression_attribute_names:
            scan_params['ExpressionAttributeNames'] = expression_attribute_names
        
        if expression_attribute_values:
            scan_params['ExpressionAttributeValues'] = expression_attribute_values
        
        if index_name:
            scan_params['IndexName'] = index_name
        
        if limit is not None:
            scan_params['Limit'] = limit
        
        def scan_operation():
            return self.dynamodb_client.scan(**scan_params)
        
        return self._execute_with_retry("scan", scan_operation, table_name)
    
    def batch_get_item(self, request_items: Dict[str, Dict[str, Any]],
                      return_consumed_capacity: str = 'TOTAL') -> Dict[str, Any]:
        """
        Batch get items from DynamoDB tables.
        
        Args:
            request_items: Request items
            return_consumed_capacity: Return consumed capacity option
            
        Returns:
            Response from DynamoDB
            
        Raises:
            DynamoDBValidationError: If input validation fails
            DynamoDBError: If an error occurs while batch getting items
        """
        # Validate inputs
        if not request_items:
            raise DynamoDBValidationError("request_items cannot be empty")
        
        logger.info(f"Batch getting items from DynamoDB tables: {', '.join(request_items.keys())}")
        
        # Prepare batch get parameters
        batch_get_params = {
            'RequestItems': request_items,
            'ReturnConsumedCapacity': return_consumed_capacity
        }
        
        def batch_get_operation():
            return self.dynamodb_client.batch_get_item(**batch_get_params)
        
        return self._execute_with_retry("batch_get_item", batch_get_operation)
    
    def batch_write_item(self, request_items: Dict[str, List[Dict[str, Any]]],
                        return_consumed_capacity: str = 'TOTAL') -> Dict[str, Any]:
        """
        Batch write items to DynamoDB tables.
        
        Args:
            request_items: Request items
            return_consumed_capacity: Return consumed capacity option
            
        Returns:
            Response from DynamoDB
            
        Raises:
            DynamoDBValidationError: If input validation fails
            DynamoDBError: If an error occurs while batch writing items
        """
        # Validate inputs
        if not request_items:
            raise DynamoDBValidationError("request_items cannot be empty")
        
        logger.info(f"Batch writing items to DynamoDB tables: {', '.join(request_items.keys())}")
        
        # Prepare batch write parameters
        batch_write_params = {
            'RequestItems': request_items,
            'ReturnConsumedCapacity': return_consumed_capacity
        }
        
        def batch_write_operation():
            return self.dynamodb_client.batch_write_item(**batch_write_params)
        
        return self._execute_with_retry("batch_write_item", batch_write_operation)
    
    def transact_get_items(self, transact_items: List[Dict[str, Any]],
                          return_consumed_capacity: str = 'TOTAL') -> Dict[str, Any]:
        """
        Transactionally get items from DynamoDB tables.
        
        Args:
            transact_items: Transact items
            return_consumed_capacity: Return consumed capacity option
            
        Returns:
            Response from DynamoDB
            
        Raises:
            DynamoDBValidationError: If input validation fails
            DynamoDBError: If an error occurs while transactionally getting items
        """
        # Validate inputs
        if not transact_items:
            raise DynamoDBValidationError("transact_items cannot be empty")
        
        logger.info(f"Transactionally getting {len(transact_items)} items from DynamoDB")
        
        # Prepare transact get parameters
        transact_get_params = {
            'TransactItems': transact_items,
            'ReturnConsumedCapacity': return_consumed_capacity
        }
        
        def transact_get_operation():
            return self.dynamodb_client.transact_get_items(**transact_get_params)
        
        return self._execute_with_retry("transact_get_items", transact_get_operation)
    
    def transact_write_items(self, transact_items: List[Dict[str, Any]],
                            client_request_token: Optional[str] = None,
                            return_consumed_capacity: str = 'TOTAL',
                            return_item_collection_metrics: str = 'NONE') -> Dict[str, Any]:
        """
        Transactionally write items to DynamoDB tables.
        
        Args:
            transact_items: Transact items
            client_request_token: Client request token (optional)
            return_consumed_capacity: Return consumed capacity option
            return_item_collection_metrics: Return item collection metrics option
            
        Returns:
            Response from DynamoDB
            
        Raises:
            DynamoDBValidationError: If input validation fails
            DynamoDBError: If an error occurs while transactionally writing items
        """
        # Validate inputs
        if not transact_items:
            raise DynamoDBValidationError("transact_items cannot be empty")
        
        logger.info(f"Transactionally writing {len(transact_items)} items to DynamoDB")
        
        # Prepare transact write parameters
        transact_write_params = {
            'TransactItems': transact_items,
            'ReturnConsumedCapacity': return_consumed_capacity,
            'ReturnItemCollectionMetrics': return_item_collection_metrics
        }
        
        if client_request_token:
            transact_write_params['ClientRequestToken'] = client_request_token
        
        def transact_write_operation():
            return self.dynamodb_client.transact_write_items(**transact_write_params)
        
        return self._execute_with_retry("transact_write_items", transact_write_operation)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics for DynamoDB operations.
        
        Returns:
            Metrics for DynamoDB operations
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
            "read_capacity_consumed": self.metrics["read_capacity_consumed"],
            "write_capacity_consumed": self.metrics["write_capacity_consumed"],
            "throttling_count": self.metrics["throttling_count"],
            "table_operations": self.metrics["table_operations"]
        }
    
    def reset_metrics(self) -> None:
        """Reset metrics for DynamoDB operations."""
        self.metrics = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "retried_operations": 0,
            "operation_latencies": [],
            "read_capacity_consumed": 0,
            "write_capacity_consumed": 0,
            "throttling_count": 0,
            "table_operations": {}
        }
        
        logger.info("DynamoDB metrics reset")