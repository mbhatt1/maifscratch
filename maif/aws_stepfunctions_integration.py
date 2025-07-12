"""
AWS Step Functions Integration for MAIF
======================================

Provides integration with AWS Step Functions for workflow orchestration.
"""

import json
import time
import logging
import uuid
from typing import Dict, List, Optional, Any, Union, Tuple

import boto3
from botocore.exceptions import ClientError, ConnectionError, EndpointConnectionError

# Configure logger
logger = logging.getLogger(__name__)


class StepFunctionsError(Exception):
    """Base exception for Step Functions integration errors."""
    pass


class StepFunctionsConnectionError(StepFunctionsError):
    """Exception for Step Functions connection errors."""
    pass


class StepFunctionsThrottlingError(StepFunctionsError):
    """Exception for Step Functions throttling errors."""
    pass


class StepFunctionsValidationError(StepFunctionsError):
    """Exception for Step Functions validation errors."""
    pass


class StepFunctionsPermissionError(StepFunctionsError):
    """Exception for Step Functions permission errors."""
    pass


class StepFunctionsExecutionError(StepFunctionsError):
    """Exception for Step Functions execution errors."""
    pass


class StepFunctionsClient:
    """Client for AWS Step Functions service with production-ready features."""
    
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
                max_retries: int = 3, base_delay: float = 0.5, max_delay: float = 5.0,
                state_machine_cache_ttl: int = 3600):
        """
        Initialize Step Functions client.
        
        Args:
            region_name: AWS region name
            profile_name: AWS profile name (optional)
            max_retries: Maximum number of retries for transient errors
            base_delay: Base delay for exponential backoff (seconds)
            max_delay: Maximum delay for exponential backoff (seconds)
            state_machine_cache_ttl: Time-to-live for state machine cache entries (seconds)
            
        Raises:
            StepFunctionsConnectionError: If unable to initialize the Step Functions client
        """
        # Validate inputs
        if not region_name:
            raise StepFunctionsValidationError("region_name cannot be empty")
        
        if max_retries < 0:
            raise StepFunctionsValidationError("max_retries must be non-negative")
        
        if base_delay <= 0 or max_delay <= 0:
            raise StepFunctionsValidationError("base_delay and max_delay must be positive")
        
        if state_machine_cache_ttl <= 0:
            raise StepFunctionsValidationError("state_machine_cache_ttl must be positive")
        
        # Initialize AWS session
        try:
            logger.info(f"Initializing Step Functions client in region {region_name}")
            session = boto3.Session(profile_name=profile_name, region_name=region_name)
            
            # Initialize Step Functions client
            self.sfn_client = session.client('stepfunctions')
            
            # Retry configuration
            self.max_retries = max_retries
            self.base_delay = base_delay
            self.max_delay = max_delay
            
            # Cache for state machine info
            self.state_machine_cache: Dict[str, Any] = {}
            self.state_machine_cache_expiry: Dict[str, float] = {}
            self.state_machine_cache_ttl = state_machine_cache_ttl
            
            # Metrics for monitoring
            self.metrics = {
                "total_operations": 0,
                "successful_operations": 0,
                "failed_operations": 0,
                "retried_operations": 0,
                "operation_latencies": [],
                "throttling_count": 0,
                "state_machine_operations": {},
                "execution_operations": {}
            }
            
            logger.info("Step Functions client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Step Functions client: {e}", exc_info=True)
            raise StepFunctionsConnectionError(f"Failed to initialize Step Functions client: {e}")
    
    def _execute_with_retry(self, operation_name: str, operation_func, state_machine_arn: Optional[str] = None, *args, **kwargs):
        """
        Execute an operation with exponential backoff retry for transient errors.
        
        Args:
            operation_name: Name of the operation for logging
            operation_func: Function to execute
            state_machine_arn: ARN of the state machine (for metrics)
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Result of the operation
            
        Raises:
            StepFunctionsError: For non-retryable errors or if max retries exceeded
        """
        start_time = time.time()
        self.metrics["total_operations"] += 1
        
        # Track state machine-specific operations
        if state_machine_arn:
            if state_machine_arn not in self.metrics["state_machine_operations"]:
                self.metrics["state_machine_operations"][state_machine_arn] = {
                    "total": 0,
                    "successful": 0,
                    "failed": 0,
                    "retried": 0,
                    "throttled": 0
                }
            self.metrics["state_machine_operations"][state_machine_arn]["total"] += 1
        
        retries = 0
        last_exception = None
        
        while retries <= self.max_retries:
            try:
                logger.debug(f"Executing Step Functions {operation_name} (attempt {retries + 1}/{self.max_retries + 1})")
                result = operation_func(*args, **kwargs)
                
                # Record success metrics
                self.metrics["successful_operations"] += 1
                latency = time.time() - start_time
                self.metrics["operation_latencies"].append(latency)
                
                # Track state machine-specific success
                if state_machine_arn:
                    self.metrics["state_machine_operations"][state_machine_arn]["successful"] += 1
                
                # Trim latencies list if it gets too large
                if len(self.metrics["operation_latencies"]) > 1000:
                    self.metrics["operation_latencies"] = self.metrics["operation_latencies"][-1000:]
                
                logger.debug(f"Step Functions {operation_name} completed successfully in {latency:.2f}s")
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
                        if state_machine_arn:
                            self.metrics["state_machine_operations"][state_machine_arn]["throttled"] += 1
                    
                    # Track state machine-specific retries
                    if state_machine_arn:
                        self.metrics["state_machine_operations"][state_machine_arn]["retried"] += 1
                    
                    delay = min(self.max_delay, self.base_delay * (2 ** retries))
                    
                    logger.warning(
                        f"Step Functions {operation_name} failed with retryable error: {error_code} - {error_message}. "
                        f"Retrying in {delay:.2f}s (attempt {retries}/{self.max_retries})"
                    )
                    
                    time.sleep(delay)
                    last_exception = e
                else:
                    # Non-retryable error or max retries exceeded
                    self.metrics["failed_operations"] += 1
                    
                    # Track state machine-specific failures
                    if state_machine_arn:
                        self.metrics["state_machine_operations"][state_machine_arn]["failed"] += 1
                    
                    if error_code in self.RETRYABLE_ERRORS:
                        logger.error(
                            f"Step Functions {operation_name} failed after {retries} retries: {error_code} - {error_message}",
                            exc_info=True
                        )
                        raise StepFunctionsThrottlingError(f"{error_code}: {error_message}. Max retries exceeded.")
                    elif error_code in ('AccessDeniedException', 'AccessDenied', 'ResourceNotFoundException'):
                        logger.error(
                            f"Step Functions {operation_name} failed due to permission or resource error: {error_code} - {error_message}",
                            exc_info=True
                        )
                        raise StepFunctionsPermissionError(f"{error_code}: {error_message}")
                    elif error_code == 'ValidationException':
                        logger.error(
                            f"Step Functions {operation_name} failed due to validation error: {error_code} - {error_message}",
                            exc_info=True
                        )
                        raise StepFunctionsValidationError(f"{error_code}: {error_message}")
                    elif error_code == 'ExecutionAlreadyExists':
                        logger.error(
                            f"Step Functions {operation_name} failed due to execution already existing: {error_code} - {error_message}",
                            exc_info=True
                        )
                        raise StepFunctionsExecutionError(f"{error_code}: {error_message}")
                    else:
                        logger.error(
                            f"Step Functions {operation_name} failed with non-retryable error: {error_code} - {error_message}",
                            exc_info=True
                        )
                        raise StepFunctionsError(f"{error_code}: {error_message}")
                        
            except (ConnectionError, EndpointConnectionError) as e:
                # Network-related errors
                if retries < self.max_retries:
                    retries += 1
                    self.metrics["retried_operations"] += 1
                    
                    # Track state machine-specific retries
                    if state_machine_arn:
                        self.metrics["state_machine_operations"][state_machine_arn]["retried"] += 1
                    
                    delay = min(self.max_delay, self.base_delay * (2 ** retries))
                    
                    logger.warning(
                        f"Step Functions {operation_name} failed with connection error: {str(e)}. "
                        f"Retrying in {delay:.2f}s (attempt {retries}/{self.max_retries})"
                    )
                    
                    time.sleep(delay)
                    last_exception = e
                else:
                    self.metrics["failed_operations"] += 1
                    
                    # Track state machine-specific failures
                    if state_machine_arn:
                        self.metrics["state_machine_operations"][state_machine_arn]["failed"] += 1
                    
                    logger.error(f"Step Functions {operation_name} failed after {retries} retries due to connection error", exc_info=True)
                    raise StepFunctionsConnectionError(f"Connection error: {str(e)}. Max retries exceeded.")
            
            except Exception as e:
                # Unexpected errors
                self.metrics["failed_operations"] += 1
                
                # Track state machine-specific failures
                if state_machine_arn:
                    self.metrics["state_machine_operations"][state_machine_arn]["failed"] += 1
                
                logger.error(f"Unexpected error in Step Functions {operation_name}: {str(e)}", exc_info=True)
                raise StepFunctionsError(f"Unexpected error: {str(e)}")
        
        # This should not be reached, but just in case
        if last_exception:
            raise StepFunctionsError(f"Operation failed after {retries} retries: {str(last_exception)}")
        else:
            raise StepFunctionsError(f"Operation failed after {retries} retries")
    
    def _clean_state_machine_cache(self):
        """Clean expired entries from state machine cache."""
        current_time = time.time()
        expired_keys = [
            key for key, expiry in self.state_machine_cache_expiry.items()
            if current_time > expiry
        ]
        
        for key in expired_keys:
            if key in self.state_machine_cache:
                del self.state_machine_cache[key]
            if key in self.state_machine_cache_expiry:
                del self.state_machine_cache_expiry[key]
        
        if expired_keys:
            logger.debug(f"Cleaned state machine cache: {len(expired_keys)} expired entries removed")
    
    def list_state_machines(self, max_results: int = 100) -> List[Dict[str, Any]]:
        """
        List Step Functions state machines.
        
        Args:
            max_results: Maximum number of state machines to return
            
        Returns:
            List of state machines
            
        Raises:
            StepFunctionsError: If an error occurs while listing state machines
        """
        # Validate inputs
        if max_results <= 0:
            raise StepFunctionsValidationError("max_results must be positive")
        
        logger.info(f"Listing Step Functions state machines (max_results: {max_results})")
        
        def list_state_machines_operation():
            paginator = self.sfn_client.get_paginator('list_state_machines')
            
            state_machines = []
            for page in paginator.paginate(maxResults=min(max_results, 100), PaginationConfig={'MaxItems': max_results}):
                state_machines.extend(page.get('stateMachines', []))
                
                # Stop if we've reached the limit
                if len(state_machines) >= max_results:
                    state_machines = state_machines[:max_results]
                    break
            
            return state_machines
        
        state_machines = self._execute_with_retry("list_state_machines", list_state_machines_operation)
        logger.info(f"Found {len(state_machines)} Step Functions state machines")
        return state_machines
    
    def describe_state_machine(self, state_machine_arn: str) -> Dict[str, Any]:
        """
        Describe a Step Functions state machine.
        
        Args:
            state_machine_arn: ARN of the state machine
            
        Returns:
            State machine description
            
        Raises:
            StepFunctionsValidationError: If input validation fails
            StepFunctionsError: If an error occurs while describing the state machine
        """
        # Validate inputs
        if not state_machine_arn:
            raise StepFunctionsValidationError("state_machine_arn cannot be empty")
        
        # Clean cache periodically
        self._clean_state_machine_cache()
        
        # Check cache
        if state_machine_arn in self.state_machine_cache:
            logger.debug(f"Returning cached state machine info for {state_machine_arn}")
            return self.state_machine_cache[state_machine_arn]
        
        logger.info(f"Describing Step Functions state machine {state_machine_arn}")
        
        def describe_state_machine_operation():
            return self.sfn_client.describe_state_machine(stateMachineArn=state_machine_arn)
        
        state_machine_info = self._execute_with_retry("describe_state_machine", describe_state_machine_operation, state_machine_arn)
        
        # Cache state machine info
        self.state_machine_cache[state_machine_arn] = state_machine_info
        self.state_machine_cache_expiry[state_machine_arn] = time.time() + self.state_machine_cache_ttl
        
        return state_machine_info
    
    def create_state_machine(self, name: str, definition: str, role_arn: str,
                            type: str = 'STANDARD',
                            logging_configuration: Optional[Dict[str, Any]] = None,
                            tags: Optional[List[Dict[str, str]]] = None,
                            tracing_configuration: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Create a Step Functions state machine.
        
        Args:
            name: Name of the state machine
            definition: State machine definition (JSON string)
            role_arn: ARN of the IAM role to use
            type: Type of the state machine (STANDARD or EXPRESS)
            logging_configuration: Logging configuration (optional)
            tags: Tags to attach to the state machine (optional)
            tracing_configuration: Tracing configuration (optional)
            
        Returns:
            State machine creation response
            
        Raises:
            StepFunctionsValidationError: If input validation fails
            StepFunctionsError: If an error occurs while creating the state machine
        """
        # Validate inputs
        if not name:
            raise StepFunctionsValidationError("name cannot be empty")
        
        if not definition:
            raise StepFunctionsValidationError("definition cannot be empty")
        
        if not role_arn:
            raise StepFunctionsValidationError("role_arn cannot be empty")
        
        if type not in ('STANDARD', 'EXPRESS'):
            raise StepFunctionsValidationError("type must be either 'STANDARD' or 'EXPRESS'")
        
        # Validate definition is valid JSON
        try:
            json.loads(definition)
        except json.JSONDecodeError as e:
            raise StepFunctionsValidationError(f"definition is not valid JSON: {str(e)}")
        
        logger.info(f"Creating Step Functions state machine {name}")
        
        # Prepare create parameters
        create_params = {
            'name': name,
            'definition': definition,
            'roleArn': role_arn,
            'type': type
        }
        
        if logging_configuration:
            create_params['loggingConfiguration'] = logging_configuration
        
        if tags:
            create_params['tags'] = tags
        
        if tracing_configuration:
            create_params['tracingConfiguration'] = tracing_configuration
        
        def create_state_machine_operation():
            return self.sfn_client.create_state_machine(**create_params)
        
        return self._execute_with_retry("create_state_machine", create_state_machine_operation)
    
    def update_state_machine(self, state_machine_arn: str,
                            definition: Optional[str] = None,
                            role_arn: Optional[str] = None,
                            logging_configuration: Optional[Dict[str, Any]] = None,
                            tracing_configuration: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Update a Step Functions state machine.
        
        Args:
            state_machine_arn: ARN of the state machine
            definition: State machine definition (JSON string) (optional)
            role_arn: ARN of the IAM role to use (optional)
            logging_configuration: Logging configuration (optional)
            tracing_configuration: Tracing configuration (optional)
            
        Returns:
            State machine update response
            
        Raises:
            StepFunctionsValidationError: If input validation fails
            StepFunctionsError: If an error occurs while updating the state machine
        """
        # Validate inputs
        if not state_machine_arn:
            raise StepFunctionsValidationError("state_machine_arn cannot be empty")
        
        if not definition and not role_arn and not logging_configuration and not tracing_configuration:
            raise StepFunctionsValidationError("At least one of definition, role_arn, logging_configuration, or tracing_configuration must be provided")
        
        # Validate definition is valid JSON if provided
        if definition:
            try:
                json.loads(definition)
            except json.JSONDecodeError as e:
                raise StepFunctionsValidationError(f"definition is not valid JSON: {str(e)}")
        
        logger.info(f"Updating Step Functions state machine {state_machine_arn}")
        
        # Prepare update parameters
        update_params = {
            'stateMachineArn': state_machine_arn
        }
        
        if definition:
            update_params['definition'] = definition
        
        if role_arn:
            update_params['roleArn'] = role_arn
        
        if logging_configuration:
            update_params['loggingConfiguration'] = logging_configuration
        
        if tracing_configuration:
            update_params['tracingConfiguration'] = tracing_configuration
        
        def update_state_machine_operation():
            response = self.sfn_client.update_state_machine(**update_params)
            
            # Remove from cache if it exists
            if state_machine_arn in self.state_machine_cache:
                del self.state_machine_cache[state_machine_arn]
            
            if state_machine_arn in self.state_machine_cache_expiry:
                del self.state_machine_cache_expiry[state_machine_arn]
            
            return response
        
        return self._execute_with_retry("update_state_machine", update_state_machine_operation, state_machine_arn)
    
    def delete_state_machine(self, state_machine_arn: str) -> Dict[str, Any]:
        """
        Delete a Step Functions state machine.
        
        Args:
            state_machine_arn: ARN of the state machine
            
        Returns:
            State machine deletion response
            
        Raises:
            StepFunctionsValidationError: If input validation fails
            StepFunctionsError: If an error occurs while deleting the state machine
        """
        # Validate inputs
        if not state_machine_arn:
            raise StepFunctionsValidationError("state_machine_arn cannot be empty")
        
        logger.info(f"Deleting Step Functions state machine {state_machine_arn}")
        
        def delete_state_machine_operation():
            response = self.sfn_client.delete_state_machine(stateMachineArn=state_machine_arn)
            
            # Remove from cache if it exists
            if state_machine_arn in self.state_machine_cache:
                del self.state_machine_cache[state_machine_arn]
            
            if state_machine_arn in self.state_machine_cache_expiry:
                del self.state_machine_cache_expiry[state_machine_arn]
            
            return response
        
        return self._execute_with_retry("delete_state_machine", delete_state_machine_operation, state_machine_arn)
    
    def start_execution(self, state_machine_arn: str, 
                       input: Optional[str] = None,
                       name: Optional[str] = None,
                       trace_header: Optional[str] = None) -> Dict[str, Any]:
        """
        Start a Step Functions state machine execution.
        
        Args:
            state_machine_arn: ARN of the state machine
            input: Input data for the execution (JSON string) (optional)
            name: Name of the execution (optional)
            trace_header: Trace header for the execution (optional)
            
        Returns:
            Execution start response
            
        Raises:
            StepFunctionsValidationError: If input validation fails
            StepFunctionsError: If an error occurs while starting the execution
        """
        # Validate inputs
        if not state_machine_arn:
            raise StepFunctionsValidationError("state_machine_arn cannot be empty")
        
        # Validate input is valid JSON if provided
        if input:
            try:
                json.loads(input)
            except json.JSONDecodeError as e:
                raise StepFunctionsValidationError(f"input is not valid JSON: {str(e)}")
        
        # Generate a unique name if not provided
        if not name:
            name = f"execution-{uuid.uuid4()}"
        
        logger.info(f"Starting Step Functions state machine execution {name} for {state_machine_arn}")
        
        # Prepare start execution parameters
        start_params = {
            'stateMachineArn': state_machine_arn,
            'name': name
        }
        
        if input:
            start_params['input'] = input
        
        if trace_header:
            start_params['traceHeader'] = trace_header
        
        def start_execution_operation():
            return self.sfn_client.start_execution(**start_params)
        
        return self._execute_with_retry("start_execution", start_execution_operation, state_machine_arn)
    
    def stop_execution(self, execution_arn: str, error: Optional[str] = None, cause: Optional[str] = None) -> Dict[str, Any]:
        """
        Stop a Step Functions state machine execution.
        
        Args:
            execution_arn: ARN of the execution
            error: Error code for the stop reason (optional)
            cause: Cause for the stop reason (optional)
            
        Returns:
            Execution stop response
            
        Raises:
            StepFunctionsValidationError: If input validation fails
            StepFunctionsError: If an error occurs while stopping the execution
        """
        # Validate inputs
        if not execution_arn:
            raise StepFunctionsValidationError("execution_arn cannot be empty")
        
        logger.info(f"Stopping Step Functions execution {execution_arn}")
        
        # Prepare stop execution parameters
        stop_params = {
            'executionArn': execution_arn
        }
        
        if error:
            stop_params['error'] = error
        
        if cause:
            stop_params['cause'] = cause
        
        def stop_execution_operation():
            return self.sfn_client.stop_execution(**stop_params)
        
        return self._execute_with_retry("stop_execution", stop_execution_operation)
    
    def describe_execution(self, execution_arn: str) -> Dict[str, Any]:
        """
        Describe a Step Functions state machine execution.
        
        Args:
            execution_arn: ARN of the execution
            
        Returns:
            Execution description
            
        Raises:
            StepFunctionsValidationError: If input validation fails
            StepFunctionsError: If an error occurs while describing the execution
        """
        # Validate inputs
        if not execution_arn:
            raise StepFunctionsValidationError("execution_arn cannot be empty")
        
        logger.info(f"Describing Step Functions execution {execution_arn}")
        
        def describe_execution_operation():
            return self.sfn_client.describe_execution(executionArn=execution_arn)
        
        return self._execute_with_retry("describe_execution", describe_execution_operation)
    
    def list_executions(self, state_machine_arn: str, status_filter: Optional[str] = None, max_results: int = 100) -> List[Dict[str, Any]]:
        """
        List Step Functions state machine executions.
        
        Args:
            state_machine_arn: ARN of the state machine
            status_filter: Status filter (optional)
            max_results: Maximum number of executions to return
            
        Returns:
            List of executions
            
        Raises:
            StepFunctionsValidationError: If input validation fails
            StepFunctionsError: If an error occurs while listing executions
        """
        # Validate inputs
        if not state_machine_arn:
            raise StepFunctionsValidationError("state_machine_arn cannot be empty")
        
        if status_filter and status_filter not in ('RUNNING', 'SUCCEEDED', 'FAILED', 'TIMED_OUT', 'ABORTED'):
            raise StepFunctionsValidationError("status_filter must be one of 'RUNNING', 'SUCCEEDED', 'FAILED', 'TIMED_OUT', 'ABORTED'")
        
        if max_results <= 0:
            raise StepFunctionsValidationError("max_results must be positive")
        
        logger.info(f"Listing Step Functions executions for {state_machine_arn}")
        
        def list_executions_operation():
            paginator = self.sfn_client.get_paginator('list_executions')
            
            list_params = {
                'stateMachineArn': state_machine_arn,
                'maxResults': min(max_results, 100),
                'PaginationConfig': {'MaxItems': max_results}
            }
            
            if status_filter:
                list_params['statusFilter'] = status_filter
            
            executions = []
            for page in paginator.paginate(**list_params):
                executions.extend(page.get('executions', []))
                
                # Stop if we've reached the limit
                if len(executions) >= max_results:
                    executions = executions[:max_results]
                    break
            
            return executions
        
        executions = self._execute_with_retry("list_executions", list_executions_operation, state_machine_arn)
        logger.info(f"Found {len(executions)} Step Functions executions for {state_machine_arn}")
        return executions
    
    def get_execution_history(self, execution_arn: str, max_results: int = 100, reverse_order: bool = False) -> List[Dict[str, Any]]:
        """
        Get the history of a Step Functions state machine execution.
        
        Args:
            execution_arn: ARN of the execution
            max_results: Maximum number of history events to return
            reverse_order: Whether to return events in reverse order
            
        Returns:
            List of history events
            
        Raises:
            StepFunctionsValidationError: If input validation fails
            StepFunctionsError: If an error occurs while getting the execution history
        """
        # Validate inputs
        if not execution_arn:
            raise StepFunctionsValidationError("execution_arn cannot be empty")
        
        if max_results <= 0:
            raise StepFunctionsValidationError("max_results must be positive")
        
        logger.info(f"Getting Step Functions execution history for {execution_arn}")
        
        def get_execution_history_operation():
            paginator = self.sfn_client.get_paginator('get_execution_history')
            
            history_params = {
                'executionArn': execution_arn,
                'maxResults': min(max_results, 100),
                'reverseOrder': reverse_order,
                'PaginationConfig': {'MaxItems': max_results}
            }
            
            events = []
            for page in paginator.paginate(**history_params):
                events.extend(page.get('events', []))
                
                # Stop if we've reached the limit
                if len(events) >= max_results:
                    events = events[:max_results]
                    break
            
            return events
        
        events = self._execute_with_retry("get_execution_history", get_execution_history_operation)
        logger.info(f"Retrieved {len(events)} Step Functions execution history events for {execution_arn}")
        return events
    
    def send_task_success(self, task_token: str, output: str) -> Dict[str, Any]:
        """
        Send a task success signal to Step Functions.
        
        Args:
            task_token: Task token
            output: Output data (JSON string)
            
        Returns:
            Task success response
            
        Raises:
            StepFunctionsValidationError: If input validation fails
            StepFunctionsError: If an error occurs while sending the task success signal
        """
        # Validate inputs
        if not task_token:
            raise StepFunctionsValidationError("task_token cannot be empty")
        
        # Validate output is valid JSON
        try:
            json.loads(output)
        except json.JSONDecodeError as e:
            raise StepFunctionsValidationError(f"output is not valid JSON: {str(e)}")
        
        logger.info("Sending task success signal to Step Functions")
        
        def send_task_success_operation():
            return self.sfn_client.send_task_success(taskToken=task_token, output=output)
        
        return self._execute_with_retry("send_task_success", send_task_success_operation)
    
    def send_task_failure(self, task_token: str, error: Optional[str] = None, cause: Optional[str] = None) -> Dict[str, Any]:
        """
        Send a task failure signal to Step Functions.
        
        Args:
            task_token: Task token
            error: Error code (optional)
            cause: Cause of the failure (optional)
            
        Returns:
            Task failure response
            
        Raises:
            StepFunctionsValidationError: If input validation fails
            StepFunctionsError: If an error occurs while sending the task failure signal
        """
        # Validate inputs
        if not task_token:
            raise StepFunctionsValidationError("task_token cannot be empty")
        
        logger.info("Sending task failure signal to Step Functions")
        
        # Prepare send task failure parameters
        failure_params = {
            'taskToken': task_token
        }
        
        if error:
            failure_params['error'] = error
        
        if cause:
            failure_params['cause'] = cause
        
        def send_task_failure_operation():
            return self.sfn_client.send_task_failure(**failure_params)
        
        return self._execute_with_retry("send_task_failure", send_task_failure_operation)
    
    def send_task_heartbeat(self, task_token: str) -> Dict[str, Any]:
        """
        Send a task heartbeat signal to Step Functions.
        
        Args:
            task_token: Task token
            
        Returns:
            Task heartbeat response
            
        Raises:
            StepFunctionsValidationError: If input validation fails
            StepFunctionsError: If an error occurs while sending the task heartbeat signal
        """
        # Validate inputs
        if not task_token:
            raise StepFunctionsValidationError("task_token cannot be empty")
        
        logger.info("Sending task heartbeat signal to Step Functions")
        
        def send_task_heartbeat_operation():
            return self.sfn_client.send_task_heartbeat(taskToken=task_token)
        
        return self._execute_with_retry("send_task_heartbeat", send_task_heartbeat_operation)
    
    def tag_resource(self, resource_arn: str, tags: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Tag a Step Functions resource.
        
        Args:
            resource_arn: ARN of the resource
            tags: Tags to attach to the resource
            
        Returns:
            Tag resource response
            
        Raises:
            StepFunctionsValidationError: If input validation fails
            StepFunctionsError: If an error occurs while tagging the resource
        """
        # Validate inputs
        if not resource_arn:
            raise StepFunctionsValidationError("resource_arn cannot be empty")
        
        if not tags:
            raise StepFunctionsValidationError("tags cannot be empty")
        
        logger.info(f"Tagging Step Functions resource {resource_arn}")
        
        def tag_resource_operation():
            return self.sfn_client.tag_resource(resourceArn=resource_arn, tags=tags)
        
        return self._execute_with_retry("tag_resource", tag_resource_operation)
    
    def untag_resource(self, resource_arn: str, tag_keys: List[str]) -> Dict[str, Any]:
        """
        Untag a Step Functions resource.
        
        Args:
            resource_arn: ARN of the resource
            tag_keys: Keys of the tags to remove
            
        Returns:
            Untag resource response
            
        Raises:
            StepFunctionsValidationError: If input validation fails
            StepFunctionsError: If an error occurs while untagging the resource
        """
        # Validate inputs
        if not resource_arn:
            raise StepFunctionsValidationError("resource_arn cannot be empty")
        
        if not tag_keys:
            raise StepFunctionsValidationError("tag_keys cannot be empty")
        
        logger.info(f"Untagging Step Functions resource {resource_arn}")
        
        def untag_resource_operation():
            return self.sfn_client.untag_resource(resourceArn=resource_arn, tagKeys=tag_keys)
        
        return self._execute_with_retry("untag_resource", untag_resource_operation)
    
    def list_tags_for_resource(self, resource_arn: str) -> Dict[str, Any]:
        """
        List tags for a Step Functions resource.
        
        Args:
            resource_arn: ARN of the resource
            
        Returns:
            List tags response
            
        Raises:
            StepFunctionsValidationError: If input validation fails
            StepFunctionsError: If an error occurs while listing tags
        """
        # Validate inputs
        if not resource_arn:
            raise StepFunctionsValidationError("resource_arn cannot be empty")
        
        logger.info(f"Listing tags for Step Functions resource {resource_arn}")
        
        def list_tags_operation():
            return self.sfn_client.list_tags_for_resource(resourceArn=resource_arn)
        
        return self._execute_with_retry("list_tags_for_resource", list_tags_operation)
    
    def wait_for_execution_completion(self, execution_arn: str,
                                     max_wait_time: int = 300,
                                     poll_interval: int = 5) -> Dict[str, Any]:
        """
        Wait for a Step Functions execution to complete.
        
        Args:
            execution_arn: ARN of the execution
            max_wait_time: Maximum time to wait in seconds
            poll_interval: Time between polls in seconds
            
        Returns:
            Final execution description
            
        Raises:
            StepFunctionsValidationError: If input validation fails
            StepFunctionsError: If an error occurs while waiting for the execution
            TimeoutError: If the execution does not complete within max_wait_time
        """
        # Validate inputs
        if not execution_arn:
            raise StepFunctionsValidationError("execution_arn cannot be empty")
        
        if max_wait_time <= 0:
            raise StepFunctionsValidationError("max_wait_time must be positive")
        
        if poll_interval <= 0:
            raise StepFunctionsValidationError("poll_interval must be positive")
        
        logger.info(f"Waiting for Step Functions execution {execution_arn} to complete")
        
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            execution = self.describe_execution(execution_arn)
            status = execution.get('status')
            
            if status in ('SUCCEEDED', 'FAILED', 'TIMED_OUT', 'ABORTED'):
                logger.info(f"Step Functions execution {execution_arn} completed with status {status}")
                return execution
            
            logger.debug(f"Step Functions execution {execution_arn} still running, waiting {poll_interval}s")
            time.sleep(poll_interval)
        
        raise TimeoutError(f"Step Functions execution {execution_arn} did not complete within {max_wait_time}s")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics for Step Functions operations.
        
        Returns:
            Metrics for Step Functions operations
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
            "state_machine_operations": self.metrics["state_machine_operations"],
            "execution_operations": self.metrics["execution_operations"]
        }
    
    def reset_metrics(self) -> None:
        """Reset metrics for Step Functions operations."""
        self.metrics = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "retried_operations": 0,
            "operation_latencies": [],
            "throttling_count": 0,
            "state_machine_operations": {},
            "execution_operations": {}
        }
        
        logger.info("Step Functions metrics reset")