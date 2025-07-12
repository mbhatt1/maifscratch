"""
AWS Distributed Integration Demo
===============================

Demonstrates how to use the AWS distributed integration in MAIF.
"""

import json
import time
import logging
import sys
import os
import uuid
from pathlib import Path

from maif.aws_distributed_integration import AWSDistributedCoordinator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)


def main():
    """Main function to demonstrate AWS distributed integration."""
    
    # Create a unique node ID
    node_id = f"node-{uuid.uuid4().hex[:8]}"
    
    # Path for MAIF file
    maif_path = Path("./data/distributed_demo.maif")
    os.makedirs(maif_path.parent, exist_ok=True)
    
    # AWS configuration
    region_name = "us-east-1"
    dynamodb_table_prefix = "maif_distributed_demo_"
    s3_bucket = "maif-distributed-demo-bucket"  # Replace with your bucket name
    lambda_function_prefix = "maif_distributed"  # Replace with your Lambda function prefix
    step_functions_state_machine = None  # Replace with your state machine ARN if available
    
    # Initialize AWS Distributed Coordinator
    try:
        coordinator = AWSDistributedCoordinator(
            node_id=node_id,
            maif_path=str(maif_path),
            region_name=region_name,
            dynamodb_table_prefix=dynamodb_table_prefix,
            s3_bucket=s3_bucket,
            lambda_function_prefix=lambda_function_prefix,
            step_functions_state_machine=step_functions_state_machine
        )
        
        logger.info(f"AWS Distributed Coordinator initialized with node ID: {node_id}")
        
        # Create required infrastructure
        logger.info("Creating required DynamoDB tables...")
        coordinator.create_required_tables()
        
        logger.info("Creating S3 bucket if needed...")
        coordinator.create_s3_bucket_if_needed()
        
        # Demonstrate distributed lock
        resource_id = "demo-resource"
        logger.info(f"Acquiring distributed lock for resource: {resource_id}")
        
        if coordinator.acquire_lock(resource_id, timeout=10.0):
            logger.info(f"Successfully acquired lock for {resource_id}")
            
            try:
                # Demonstrate adding a block
                block_data = f"This is a test block from node {node_id}".encode()
                block_type = "text"
                metadata = {
                    "created_by": node_id,
                    "timestamp": time.time(),
                    "description": "Demo block for AWS distributed integration"
                }
                
                logger.info("Adding a block with distributed coordination...")
                block_id = coordinator.add_block_distributed(block_data, block_type, metadata)
                logger.info(f"Added block with ID: {block_id}")
                
                # Demonstrate retrieving a block
                logger.info(f"Retrieving block with ID: {block_id}")
                retrieved_block = coordinator.get_block_distributed(block_id)
                
                if retrieved_block:
                    logger.info(f"Successfully retrieved block: {retrieved_block.decode()}")
                else:
                    logger.warning("Failed to retrieve block")
                
                # Demonstrate starting a workflow (if state machine is configured)
                if coordinator.step_functions_state_machine:
                    logger.info("Starting a distributed workflow...")
                    
                    workflow_input = {
                        "node_id": node_id,
                        "block_id": block_id,
                        "action": "process",
                        "timestamp": time.time()
                    }
                    
                    execution_arn = coordinator.start_distributed_workflow(workflow_input)
                    
                    if execution_arn:
                        logger.info(f"Started workflow with execution ARN: {execution_arn}")
                        
                        # Wait for workflow to complete
                        logger.info("Waiting for workflow to complete...")
                        time.sleep(5)
                        
                        # Check workflow status
                        status = coordinator.check_workflow_status(execution_arn)
                        logger.info(f"Workflow status: {json.dumps(status, indent=2)}")
                
            finally:
                # Release the lock
                logger.info(f"Releasing lock for resource: {resource_id}")
                coordinator.release_lock(resource_id)
        else:
            logger.warning(f"Failed to acquire lock for {resource_id}")
        
        # Get cluster stats
        stats = coordinator.get_cluster_stats()
        logger.info(f"Cluster stats: {json.dumps(stats, indent=2)}")
        
    except Exception as e:
        logger.error(f"Error in AWS distributed demo: {e}", exc_info=True)


if __name__ == "__main__":
    main()