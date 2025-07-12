"""
AWS Lambda Integration for MAIF
==============================

Provides integration with AWS Lambda for serverless function execution.
"""

import json
import time
import logging
import datetime
import base64
from typing import Dict, List, Optional, Any, Union, Tuple
import boto3
from botocore.exceptions import ClientError, ConnectionError, EndpointConnectionError

# Configure logger
logger = logging.getLogger(__name__)


class LambdaError(Exception):
    """Base exception for Lambda integration errors."""
    pass


class LambdaConnectionError(LambdaError):
    """Exception for Lambda connection errors."""
    pass


class LambdaThrottlingError(LambdaError):
    """Exception for Lambda throttling errors."""
    pass


class LambdaValidationError(LambdaError):
    """Exception for Lambda validation errors."""
    pass


class LambdaPermissionError(LambdaError):
    """Exception for Lambda permission errors."""
    pass


class LambdaInvocationError(LambdaError):
    """Exception for Lambda invocation errors."""
    pass


class LambdaClient:
    """Client for AWS Lambda service with production-ready features."""
    
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
        'ProvisionedConcurrencyConfigNotFoundException',
        'EC2ThrottledException',
        'EC2UnexpectedException',
        'SubnetIPAddressLimitReachedException',
        'ENILimitReachedException'
    }
    
    def __init__(self, region_name: str = "us-east-1", profile_name: Optional[str] = None,
                max_retries: int = 3, base_delay: float = 0.5, max_delay: float = 5.0):
        """
        Initialize Lambda client.
        
        Args:
            region_name: AWS region name
            profile_name: AWS profile name (optional)
            max_retries: Maximum number of retries for transient errors
            base_delay: Base delay for exponential backoff (seconds)
            max_delay: Maximum delay for exponential backoff (seconds)
            
        Raises:
            LambdaConnectionError: If unable to initialize the Lambda client
        """
        # Validate inputs
        if not region_name:
            raise LambdaValidationError("region_name cannot be empty")
        
        if max_retries < 0:
            raise LambdaValidationError("max_retries must be non-negative")
        
        if base_delay <= 0 or max_delay <= 0:
            raise LambdaValidationError("base_delay and max_delay must be positive")
        
        # Initialize AWS session
        try:
            logger.info(f"Initializing Lambda client in region {region_name}")
            session = boto3.Session(profile_name=profile_name, region_name=region_name)
            
            # Initialize Lambda client
            self.lambda_client = session.client('lambda')
            
            # Retry configuration
            self.max_retries = max_retries
            self.base_delay = base_delay
            self.max_delay = max_delay
            
            # Cache for function info
            self.function_info_cache: Dict[str, Dict[str, Any]] = {}
            self.function_cache_expiry: Dict[str, float] = {}
            self.function_cache_ttl = 3600  # 1 hour
            
            # Metrics for monitoring
            self.metrics = {
                "total_invocations": 0,
                "successful_invocations": 0,
                "failed_invocations": 0,
                "retried_invocations": 0,
                "invocation_latencies": [],
                "function_errors": {},
                "throttling_count": 0
            }
            
            logger.info("Lambda client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Lambda client: {e}", exc_info=True)
            raise LambdaConnectionError(f"Failed to initialize Lambda client: {e}")
    
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
            LambdaError: For non-retryable errors or if max retries exceeded
        """
        start_time = time.time()
        self.metrics["total_invocations"] += 1
        
        retries = 0
        last_exception = None
        
        while retries <= self.max_retries:
            try:
                logger.debug(f"Executing Lambda {operation_name} (attempt {retries + 1}/{self.max_retries + 1})")
                result = operation_func(*args, **kwargs)
                
                # Record success metrics
                self.metrics["successful_invocations"] += 1
                latency = time.time() - start_time
                self.metrics["invocation_latencies"].append(latency)
                
                # Trim latencies list if it gets too large
                if len(self.metrics["invocation_latencies"]) > 1000:
                    self.metrics["invocation_latencies"] = self.metrics["invocation_latencies"][-1000:]
                
                logger.debug(f"Lambda {operation_name} completed successfully in {latency:.2f}s")
                return result
                
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')
                error_message = e.response.get('Error', {}).get('Message', '')
                
                # Check if this is a retryable error
                if error_code in self.RETRYABLE_ERRORS and retries < self.max_retries:
                    retries += 1
                    self.metrics["retried_invocations"] += 1
                    
                    if error_code == 'Throttling' or error_code == 'ThrottlingException':
                        self.metrics["throttling_count"] += 1
                    
                    delay = min(self.max_delay, self.base_delay * (2 ** retries))
                    
                    logger.warning(
                        f"Lambda {operation_name} failed with retryable error: {error_code} - {error_message}. "
                        f"Retrying in {delay:.2f}s (attempt {retries}/{self.max_retries})"
                    )
                    
                    time.sleep(delay)
                    last_exception = e
                else:
                    # Non-retryable error or max retries exceeded
                    self.metrics["failed_invocations"] += 1
                    
                    # Track function-specific errors
                    if 'FunctionName' in kwargs:
                        function_name = kwargs['FunctionName']
                        if function_name not in self.metrics["function_errors"]:
                            self.metrics["function_errors"][function_name] = {}
                        
                        if error_code not in self.metrics["function_errors"][function_name]:
                            self.metrics["function_errors"][function_name][error_code] = 0
                        
                        self.metrics["function_errors"][function_name][error_code] += 1
                    
                    if error_code in self.RETRYABLE_ERRORS:
                        logger.error(
                            f"Lambda {operation_name} failed after {retries} retries: {error_code} - {error_message}",
                            exc_info=True
                        )
                        raise LambdaThrottlingError(f"{error_code}: {error_message}. Max retries exceeded.")
                    elif error_code in ('AccessDenied', 'ResourceNotFoundException'):
                        logger.error(
                            f"Lambda {operation_name} failed due to permission or resource error: {error_code} - {error_message}",
                            exc_info=True
                        )
                        raise LambdaPermissionError(f"{error_code}: {error_message}")
                    else:
                        logger.error(
                            f"Lambda {operation_name} failed with non-retryable error: {error_code} - {error_message}",
                            exc_info=True
                        )
                        raise LambdaError(f"{error_code}: {error_message}")
                        
            except (ConnectionError, EndpointConnectionError) as e:
                # Network-related errors
                if retries < self.max_retries:
                    retries += 1
                    self.metrics["retried_invocations"] += 1
                    delay = min(self.max_delay, self.base_delay * (2 ** retries))
                    
                    logger.warning(
                        f"Lambda {operation_name} failed with connection error: {str(e)}. "
                        f"Retrying in {delay:.2f}s (attempt {retries}/{self.max_retries})"
                    )
                    
                    time.sleep(delay)
                    last_exception = e
                else:
                    self.metrics["failed_invocations"] += 1
                    logger.error(f"Lambda {operation_name} failed after {retries} retries due to connection error", exc_info=True)
                    raise LambdaConnectionError(f"Connection error: {str(e)}. Max retries exceeded.")
            
            except Exception as e:
                # Unexpected errors
                self.metrics["failed_invocations"] += 1
                logger.error(f"Unexpected error in Lambda {operation_name}: {str(e)}", exc_info=True)
                raise LambdaError(f"Unexpected error: {str(e)}")
        
        # This should not be reached, but just in case
        if last_exception:
            raise LambdaError(f"Operation failed after {retries} retries: {str(last_exception)}")
        else:
            raise LambdaError(f"Operation failed after {retries} retries")
    
    def _clean_function_cache(self):
        """Clean expired entries from function cache."""
        current_time = time.time()
        expired_keys = [
            key for key, expiry in self.function_cache_expiry.items()
            if current_time > expiry
        ]
        
        for key in expired_keys:
            if key in self.function_info_cache:
                del self.function_info_cache[key]
            if key in self.function_cache_expiry:
                del self.function_cache_expiry[key]
        
        if expired_keys:
            logger.debug(f"Cleaned function cache: {len(expired_keys)} expired entries removed")
    
    def list_functions(self, max_items: int = 50) -> List[Dict[str, Any]]:
        """
        List Lambda functions.
        
        Args:
            max_items: Maximum number of functions to return
            
        Returns:
            List of function information dictionaries
            
        Raises:
            LambdaError: If an error occurs while listing functions
        """
        # Validate inputs
        if max_items <= 0:
            raise LambdaValidationError("max_items must be positive")
        
        logger.info(f"Listing Lambda functions (max: {max_items})")
        
        def list_functions_operation():
            paginator = self.lambda_client.get_paginator('list_functions')
            
            functions = []
            for page in paginator.paginate(MaxItems=max_items):
                functions.extend(page.get('Functions', []))
                
                # Stop if we've reached max_items
                if len(functions) >= max_items:
                    functions = functions[:max_items]
                    break
            
            return functions
        
        functions = self._execute_with_retry("list_functions", list_functions_operation)
        logger.info(f"Found {len(functions)} Lambda functions")
        return functions
    
    def get_function(self, function_name: str) -> Dict[str, Any]:
        """
        Get information about a Lambda function.
        
        Args:
            function_name: Name or ARN of the Lambda function
            
        Returns:
            Function information dictionary
            
        Raises:
            LambdaValidationError: If input validation fails
            LambdaError: If an error occurs while getting the function
        """
        # Validate inputs
        if not function_name:
            raise LambdaValidationError("function_name cannot be empty")
        
        # Clean cache periodically
        self._clean_function_cache()
        
        # Check cache
        if function_name in self.function_info_cache:
            logger.debug(f"Returning cached function info for {function_name}")
            return self.function_info_cache[function_name]
        
        logger.info(f"Getting information for Lambda function {function_name}")
        
        def get_function_operation():
            return self.lambda_client.get_function(FunctionName=function_name)
        
        function_info = self._execute_with_retry("get_function", get_function_operation)
        
        # Cache function info
        self.function_info_cache[function_name] = function_info
        self.function_cache_expiry[function_name] = time.time() + self.function_cache_ttl
        
        return function_info
    
    def invoke(self, function_name: str, payload: Optional[Dict[str, Any]] = None,
              invocation_type: str = 'RequestResponse',
              log_type: str = 'None',
              qualifier: Optional[str] = None) -> Dict[str, Any]:
        """
        Invoke a Lambda function.
        
        Args:
            function_name: Name or ARN of the Lambda function
            payload: Input payload for the Lambda function
            invocation_type: Invocation type ('RequestResponse', 'Event', or 'DryRun')
            log_type: Log type ('None' or 'Tail')
            qualifier: Function version or alias
            
        Returns:
            Response from Lambda
            
        Raises:
            LambdaValidationError: If input validation fails
            LambdaInvocationError: If the function returns an error
            LambdaError: If an error occurs during invocation
        """
        # Validate inputs
        if not function_name:
            raise LambdaValidationError("function_name cannot be empty")
        
        valid_invocation_types = ('RequestResponse', 'Event', 'DryRun')
        if invocation_type not in valid_invocation_types:
            raise LambdaValidationError(f"invocation_type must be one of {valid_invocation_types}")
        
        valid_log_types = ('None', 'Tail')
        if log_type not in valid_log_types:
            raise LambdaValidationError(f"log_type must be one of {valid_log_types}")
        
        logger.info(f"Invoking Lambda function {function_name} (type: {invocation_type})")
        
        # Prepare payload
        payload_json = json.dumps(payload) if payload is not None else ""
        
        # Prepare invoke parameters
        invoke_params = {
            'FunctionName': function_name,
            'InvocationType': invocation_type,
            'LogType': log_type,
            'Payload': payload_json
        }
        
        if qualifier:
            invoke_params['Qualifier'] = qualifier
        
        def invoke_operation():
            response = self.lambda_client.invoke(**invoke_params)
            
            # Check for function error
            if 'FunctionError' in response:
                # Parse error payload
                error_payload = json.loads(response['Payload'].read())
                
                # Track function error
                if function_name not in self.metrics["function_errors"]:
                    self.metrics["function_errors"][function_name] = {}
                
                error_type = error_payload.get('errorType', 'UnknownError')
                if error_type not in self.metrics["function_errors"][function_name]:
                    self.metrics["function_errors"][function_name][error_type] = 0
                
                self.metrics["function_errors"][function_name][error_type] += 1
                
                # Raise exception with error details
                error_message = error_payload.get('errorMessage', 'Unknown error')
                raise LambdaInvocationError(f"Function error: {error_type}: {error_message}")
            
            # Process response
            result = {k: v for k, v in response.items() if k != 'Payload'}
            
            # Read payload
            if 'Payload' in response:
                payload_bytes = response['Payload'].read()
                if payload_bytes:
                    try:
                        result['Payload'] = json.loads(payload_bytes)
                    except json.JSONDecodeError:
                        result['Payload'] = payload_bytes.decode('utf-8')
            
            # Decode logs if present
            if 'LogResult' in response and response['LogResult']:
                result['LogResult'] = base64.b64decode(response['LogResult']).decode('utf-8')
            
            return result
        
        return self._execute_with_retry("invoke", invoke_operation)
    
    def invoke_async(self, function_name: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Invoke a Lambda function asynchronously.
        
        Args:
            function_name: Name or ARN of the Lambda function
            payload: Input payload for the Lambda function
            
        Returns:
            Response from Lambda
            
        Raises:
            LambdaValidationError: If input validation fails
            LambdaError: If an error occurs during invocation
        """
        return self.invoke(function_name, payload, invocation_type='Event')
    
    def create_function(self, function_name: str, runtime: str, role: str, 
                       handler: str, code: Dict[str, Any],
                       description: Optional[str] = None,
                       timeout: int = 3,
                       memory_size: int = 128,
                       environment: Optional[Dict[str, str]] = None,
                       tags: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Create a Lambda function.
        
        Args:
            function_name: Name of the Lambda function
            runtime: Runtime identifier
            role: ARN of the execution role
            handler: Function handler
            code: Function code (S3 location or ZIP file)
            description: Function description
            timeout: Function timeout in seconds
            memory_size: Function memory size in MB
            environment: Environment variables
            tags: Function tags
            
        Returns:
            Function information
            
        Raises:
            LambdaValidationError: If input validation fails
            LambdaError: If an error occurs while creating the function
        """
        # Validate inputs
        if not function_name:
            raise LambdaValidationError("function_name cannot be empty")
        
        if not runtime:
            raise LambdaValidationError("runtime cannot be empty")
        
        if not role:
            raise LambdaValidationError("role cannot be empty")
        
        if not handler:
            raise LambdaValidationError("handler cannot be empty")
        
        if not code:
            raise LambdaValidationError("code cannot be empty")
        
        if timeout <= 0:
            raise LambdaValidationError("timeout must be positive")
        
        if memory_size <= 0:
            raise LambdaValidationError("memory_size must be positive")
        
        logger.info(f"Creating Lambda function {function_name} (runtime: {runtime})")
        
        # Prepare create parameters
        create_params = {
            'FunctionName': function_name,
            'Runtime': runtime,
            'Role': role,
            'Handler': handler,
            'Code': code,
            'Timeout': timeout,
            'MemorySize': memory_size
        }
        
        if description:
            create_params['Description'] = description
        
        if environment:
            create_params['Environment'] = {'Variables': environment}
        
        if tags:
            create_params['Tags'] = tags
        
        def create_function_operation():
            return self.lambda_client.create_function(**create_params)
        
        function_info = self._execute_with_retry("create_function", create_function_operation)
        
        # Cache function info
        self.function_info_cache[function_name] = function_info
        self.function_cache_expiry[function_name] = time.time() + self.function_cache_ttl
        
        return function_info
    
    def update_function_code(self, function_name: str, code: Dict[str, Any],
                            publish: bool = False) -> Dict[str, Any]:
        """
        Update a Lambda function's code.
        
        Args:
            function_name: Name or ARN of the Lambda function
            code: Function code (S3 location or ZIP file)
            publish: Whether to publish a new version
            
        Returns:
            Updated function information
            
        Raises:
            LambdaValidationError: If input validation fails
            LambdaError: If an error occurs while updating the function
        """
        # Validate inputs
        if not function_name:
            raise LambdaValidationError("function_name cannot be empty")
        
        if not code:
            raise LambdaValidationError("code cannot be empty")
        
        logger.info(f"Updating code for Lambda function {function_name}")
        
        # Prepare update parameters
        update_params = {
            'FunctionName': function_name,
            'Publish': publish
        }
        
        # Add code parameters
        if 'S3Bucket' in code and 'S3Key' in code:
            update_params['S3Bucket'] = code['S3Bucket']
            update_params['S3Key'] = code['S3Key']
            
            if 'S3ObjectVersion' in code:
                update_params['S3ObjectVersion'] = code['S3ObjectVersion']
        elif 'ZipFile' in code:
            update_params['ZipFile'] = code['ZipFile']
        else:
            raise LambdaValidationError("code must contain either S3 location or ZipFile")
        
        def update_function_code_operation():
            return self.lambda_client.update_function_code(**update_params)
        
        function_info = self._execute_with_retry("update_function_code", update_function_code_operation)
        
        # Update cache
        self.function_info_cache[function_name] = function_info
        self.function_cache_expiry[function_name] = time.time() + self.function_cache_ttl
        
        return function_info
    
    def delete_function(self, function_name: str, qualifier: Optional[str] = None) -> bool:
        """
        Delete a Lambda function.
        
        Args:
            function_name: Name or ARN of the Lambda function
            qualifier: Function version or alias
            
        Returns:
            True if successful
            
        Raises:
            LambdaValidationError: If input validation fails
            LambdaError: If an error occurs while deleting the function
        """
        # Validate inputs
        if not function_name:
            raise LambdaValidationError("function_name cannot be empty")
        
        logger.info(f"Deleting Lambda function {function_name}")
        
        # Prepare delete parameters
        delete_params = {
            'FunctionName': function_name
        }
        
        if qualifier:
            delete_params['Qualifier'] = qualifier
        
        def delete_function_operation():
            self.lambda_client.delete_function(**delete_params)
            return True
        
        result = self._execute_with_retry("delete_function", delete_function_operation)
        
        # Remove from cache
        if function_name in self.function_info_cache:
            del self.function_info_cache[function_name]
        
        if function_name in self.function_cache_expiry:
            del self.function_cache_expiry[function_name]
        
        return result


class MAIFLambdaIntegration:
    """Integration between MAIF and AWS Lambda for serverless function execution."""
    
    def __init__(self, lambda_client: LambdaClient):
        """
        Initialize MAIF Lambda integration.
        
        Args:
            lambda_client: Lambda client
            
        Raises:
            LambdaValidationError: If lambda_client is None
        """
        if lambda_client is None:
            raise LambdaValidationError("lambda_client cannot be None")
        
        logger.info("Initializing MAIF Lambda integration")
        self.lambda_client = lambda_client
        
        # Metrics
        self.metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0
        }
        
        logger.info("MAIF Lambda integration initialized successfully")
    
    def execute_function(self, function_name: str, input_data: Dict[str, Any],
                        async_execution: bool = False) -> Dict[str, Any]:
        """
        Execute a Lambda function.
        
        Args:
            function_name: Name or ARN of the Lambda function
            input_data: Input data for the function
            async_execution: Whether to execute asynchronously
            
        Returns:
            Function result
            
        Raises:
            LambdaValidationError: If input validation fails
            LambdaInvocationError: If the function returns an error
            LambdaError: If an error occurs during execution
        """
        # Validate inputs
        if not function_name:
            raise LambdaValidationError("function_name cannot be empty")
        
        logger.info(f"Executing Lambda function {function_name} (async: {async_execution})")
        self.metrics["total_executions"] += 1
        
        try:
            # Invoke function
            if async_execution:
                response = self.lambda_client.invoke_async(function_name, input_data)
                result = {
                    "status": "success",
                    "async": True,
                    "function_name": function_name,
                    "timestamp": time.time()
                }
            else:
                response = self.lambda_client.invoke(function_name, input_data)
                result = {
                    "status": "success",
                    "async": False,
                    "function_name": function_name,
                    "timestamp": time.time(),
                    "result": response.get('Payload')
                }
                
                # Include logs if available
                if 'LogResult' in response:
                    result["logs"] = response['LogResult']
            
            self.metrics["successful_executions"] += 1
            return result
            
        except LambdaInvocationError as e:
            self.metrics["failed_executions"] += 1
            logger.error(f"Function execution error: {e}")
            return {
                "status": "error",
                "function_name": function_name,
                "timestamp": time.time(),
                "error": str(e)
            }
        except LambdaError as e:
            self.metrics["failed_executions"] += 1
            logger.error(f"Lambda execution error: {e}")
            raise
    
    def execute_function_with_artifact(self, function_name: str, artifact, 
                                      async_execution: bool = False) -> Dict[str, Any]:
        """
        Execute a Lambda function with a MAIF artifact.
        
        Args:
            function_name: Name or ARN of the Lambda function
            artifact: MAIF artifact to process
            async_execution: Whether to execute asynchronously
            
        Returns:
            Function result
            
        Raises:
            LambdaValidationError: If input validation fails
            LambdaInvocationError: If the function returns an error
            LambdaError: If an error occurs during execution
        """
        # Validate inputs
        if not function_name:
            raise LambdaValidationError("function_name cannot be empty")
        
        if artifact is None:
            raise LambdaValidationError("artifact cannot be None")
        
        logger.info(f"Executing Lambda function {function_name} with artifact {artifact.name}")
        
        # Extract content from artifact
        content_list = []
        for content in artifact.get_content():
            content_item = {
                "content_type": content["content_type"],
                "metadata": content["metadata"]
            }
            
            # Convert binary data to base64
            if isinstance(content["data"], bytes):
                content_item["data"] = base64.b64encode(content["data"]).decode('utf-8')
                content_item["encoding"] = "base64"
            else:
                content_item["data"] = content["data"]
            
            content_list.append(content_item)
        
        # Prepare input data
        input_data = {
            "artifact_name": artifact.name,
            "content": content_list,
            "metadata": artifact.custom_metadata
        }
        
        # Execute function
        return self.execute_function(function_name, input_data, async_execution)


# Helper functions for easy integration
def create_lambda_integration(region_name: str = "us-east-1", 
                             profile_name: Optional[str] = None,
                             max_retries: int = 3) -> MAIFLambdaIntegration:
    """
    Create MAIF Lambda integration.
    
    Args:
        region_name: AWS region name
        profile_name: AWS profile name (optional)
        max_retries: Maximum number of retries for transient errors
        
    Returns:
        MAIFLambdaIntegration
        
    Raises:
        LambdaConnectionError: If unable to initialize the Lambda client
    """
    logger.info(f"Creating Lambda integration in region {region_name}")
    
    try:
        lambda_client = LambdaClient(
            region_name=region_name, 
            profile_name=profile_name,
            max_retries=max_retries
        )
        
        integration = MAIFLambdaIntegration(lambda_client)
        
        logger.info("Lambda integration created successfully")
        return integration
        
    except LambdaError as e:
        logger.error(f"Failed to create Lambda integration: {e}", exc_info=True)
        raise