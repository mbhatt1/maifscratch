"""
AWS Step Functions Integration Demo
=================================

Demonstrates how to use the AWS Step Functions integration in MAIF.
"""

import json
import time
import logging
import sys
from maif.aws_stepfunctions_integration import StepFunctionsClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)


def main():
    """Main function to demonstrate AWS Step Functions integration."""
    
    # Initialize Step Functions client
    try:
        sfn_client = StepFunctionsClient(
            region_name='us-east-1',
            max_retries=3,
            base_delay=1.0,
            max_delay=10.0
        )
        logger.info("Step Functions client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Step Functions client: {e}")
        return
    
    # List state machines
    try:
        state_machines = sfn_client.list_state_machines(max_results=10)
        logger.info(f"Found {len(state_machines)} state machines")
        
        for sm in state_machines:
            logger.info(f"State Machine: {sm.get('name')} (ARN: {sm.get('stateMachineArn')})")
    except Exception as e:
        logger.error(f"Failed to list state machines: {e}")
    
    # Create a simple state machine
    state_machine_name = f"MAIFDemoStateMachine-{int(time.time())}"
    state_machine_definition = {
        "Comment": "A simple sequential workflow",
        "StartAt": "FirstState",
        "States": {
            "FirstState": {
                "Type": "Pass",
                "Result": "Hello",
                "Next": "SecondState"
            },
            "SecondState": {
                "Type": "Pass",
                "Result": "World",
                "End": True
            }
        }
    }
    
    role_arn = "arn:aws:iam::123456789012:role/StepFunctionsExecutionRole"  # Replace with actual role ARN
    
    try:
        # Create state machine
        create_response = sfn_client.create_state_machine(
            name=state_machine_name,
            definition=json.dumps(state_machine_definition),
            role_arn=role_arn,
            type='STANDARD'
        )
        
        state_machine_arn = create_response.get('stateMachineArn')
        logger.info(f"Created state machine: {state_machine_name} (ARN: {state_machine_arn})")
        
        # Describe state machine
        state_machine_info = sfn_client.describe_state_machine(state_machine_arn)
        logger.info(f"State machine details: {json.dumps(state_machine_info, indent=2)}")
        
        # Start execution
        execution_input = {"key": "value"}
        execution_response = sfn_client.start_execution(
            state_machine_arn=state_machine_arn,
            input=json.dumps(execution_input),
            name=f"Execution-{int(time.time())}"
        )
        
        execution_arn = execution_response.get('executionArn')
        logger.info(f"Started execution: {execution_arn}")
        
        # Wait for execution to complete
        try:
            execution_result = sfn_client.wait_for_execution_completion(
                execution_arn=execution_arn,
                max_wait_time=60,
                poll_interval=2
            )
            
            logger.info(f"Execution completed with status: {execution_result.get('status')}")
            
            if execution_result.get('status') == 'SUCCEEDED':
                output = json.loads(execution_result.get('output', '{}'))
                logger.info(f"Execution output: {json.dumps(output, indent=2)}")
            else:
                logger.warning(f"Execution failed: {execution_result.get('error')}, cause: {execution_result.get('cause')}")
        except TimeoutError as e:
            logger.error(f"Execution timed out: {e}")
        
        # Get execution history
        history_events = sfn_client.get_execution_history(execution_arn)
        logger.info(f"Execution history contains {len(history_events)} events")
        
        # List executions for the state machine
        executions = sfn_client.list_executions(state_machine_arn)
        logger.info(f"State machine has {len(executions)} executions")
        
        # Clean up - delete state machine
        delete_response = sfn_client.delete_state_machine(state_machine_arn)
        logger.info(f"Deleted state machine: {state_machine_name}")
        
    except Exception as e:
        logger.error(f"Error in Step Functions demo: {e}")
    
    # Display metrics
    metrics = sfn_client.get_metrics()
    logger.info(f"Step Functions metrics: {json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    main()