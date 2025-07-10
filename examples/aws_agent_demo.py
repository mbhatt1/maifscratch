#!/usr/bin/env python3
"""
AWS-Integrated MAIF Agent Demo
==============================

Demonstrates the use of decorator-based AWS integration with the MAIF agentic framework.
"""

import asyncio
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import MAIF components
from maif.agentic_framework import AgentState, MAIFAgent
from maif.aws_decorators import (
    maif_agent, aws_agent, aws_bedrock, aws_s3, aws_kms, aws_lambda, aws_dynamodb,
    enhance_perception_with_bedrock, enhance_reasoning_with_bedrock, enhance_execution_with_aws
)
from maif_sdk.artifact import Artifact as MAIFArtifact
from maif_sdk.types import SecurityLevel


# ===== Simple AWS-Enhanced Agent =====

@maif_agent(workspace="./demo_workspace/aws_simple")
class SimpleAWSAgent:
    """Simple agent with AWS Bedrock integration."""
    
    def __init__(self, name="SimpleAWSAgent"):
        self.name = name
        logger.info(f"Initialized {self.name}")
    
    @aws_bedrock()
    async def generate_text(self, prompt, bedrock=None):
        """Generate text using AWS Bedrock."""
        logger.info(f"Generating text for prompt: {prompt[:50]}...")
        text_block = bedrock.generate_text_block(prompt)
        return text_block["text"]
    
    @aws_s3()
    async def store_result(self, text, bucket_name, s3_client=None):
        """Store result in S3."""
        key = f"results/{int(time.time())}.txt"
        logger.info(f"Storing result in S3: {bucket_name}/{key}")
        
        try:
            s3_client.put_object(
                Bucket=bucket_name,
                Key=key,
                Body=text.encode('utf-8'),
                ContentType='text/plain'
            )
            return f"s3://{bucket_name}/{key}"
        except Exception as e:
            logger.error(f"Error storing in S3: {e}")
            return None
    
    async def process(self):
        """Main processing loop."""
        try:
            # Generate text with Bedrock
            prompt = "Explain the benefits of using a MAIF (Model Artifact Interchange Format) for AI agents."
            text = await self.generate_text(prompt)
            logger.info(f"Generated text: {text[:100]}...")
            
            # Store in MAIF artifact
            artifact = MAIFArtifact(name="bedrock_response", client=self.maif_client)
            artifact.add_text(text, title="Bedrock Response")
            artifact.save(self.workspace_path / "bedrock_response.maif")
            
            # In a real scenario, you might store in S3
            # s3_path = await self.store_result(text, "your-bucket-name")
            
            # Simulate work
            await asyncio.sleep(5.0)
            
        except Exception as e:
            logger.error(f"Process error: {e}")


# ===== Comprehensive AWS Agent =====

@aws_agent(workspace="./demo_workspace/aws_full")
class ComprehensiveAWSAgent:
    """Fully AWS-integrated agent with all systems enhanced."""
    
    def __init__(self, name="ComprehensiveAWSAgent"):
        self.name = name
        logger.info(f"Initialized {self.name}")
    
    @aws_dynamodb()
    async def fetch_task(self, dynamodb=None):
        """Fetch task from DynamoDB."""
        logger.info("Fetching task from DynamoDB")
        
        # In a real scenario, you would query an actual table
        # table = dynamodb.Table("agent_tasks")
        # response = table.scan(Limit=1)
        # return response.get('Items', [{}])[0]
        
        # Simulate a task
        return {
            "task_id": "task-123",
            "description": "Analyze customer feedback",
            "data": "Customer feedback is very positive about our new product.",
            "priority": "high"
        }
    
    @aws_lambda()
    async def process_task(self, task, lambda_client=None):
        """Process task using AWS Lambda."""
        logger.info(f"Processing task: {task['task_id']}")
        
        # In a real scenario, you would invoke an actual Lambda function
        # response = lambda_client.invoke(
        #     FunctionName="process-task",
        #     Payload=json.dumps(task)
        # )
        # return json.loads(response['Payload'].read())
        
        # Simulate Lambda processing
        return {
            "task_id": task["task_id"],
            "status": "completed",
            "result": "Analysis complete. Sentiment: Positive"
        }
    
    @aws_kms()
    async def sign_result(self, result, key_store=None, verifier=None):
        """Sign result using AWS KMS."""
        logger.info("Signing result with KMS")
        
        # In a real scenario, you would use an actual KMS key
        # key_id = key_store.list_kms_keys()[0]['KeyId']
        # signature = verifier.sign_data(json.dumps(result).encode(), key_id)
        
        # Simulate signing
        result["signed"] = True
        result["signature"] = "kms-signature-placeholder"
        return result
    
    async def process(self):
        """Main processing loop."""
        try:
            # Perception phase - perceive input data
            task = await self.fetch_task()
            perception = await self.perceive(task["data"], "text")
            
            # Reasoning phase - analyze the data
            reasoning = await self.reason([perception])
            
            # Planning phase - create a plan
            plan = await self.plan(f"analyze {task['description']}", [reasoning])
            
            # Execution phase - execute the plan
            execution_result = await self.execute(plan)
            
            # Process result with Lambda
            processed_result = await self.process_task(task)
            
            # Sign the result with KMS
            signed_result = await self.sign_result(processed_result)
            
            # Store the final result
            result_artifact = MAIFArtifact(name=f"result_{task['task_id']}", client=self.maif_client)
            result_artifact.add_data(
                json.dumps(signed_result, indent=2).encode(),
                title=f"Task Result: {task['task_id']}",
                data_type="json"
            )
            result_artifact.save(self.workspace_path / f"result_{task['task_id']}.maif")
            
            logger.info(f"Completed task: {task['task_id']}")
            
            # Simulate work interval
            await asyncio.sleep(10.0)
            
        except Exception as e:
            logger.error(f"Process error: {e}")


# ===== Custom AWS-Enhanced Agent =====

@maif_agent(workspace="./demo_workspace/aws_custom")
@enhance_perception_with_bedrock()
class CustomAWSAgent(MAIFAgent):
    """Custom agent with selective AWS enhancements."""
    
    async def run(self):
        """Custom run implementation."""
        logger.info(f"Starting custom AWS agent {self.agent_id}")
        
        while self.state != AgentState.TERMINATED:
            try:
                # Use the AWS-enhanced perception system
                perception = await self.perceive(
                    "Analyze the current market trends for AI technologies.",
                    "text"
                )
                
                # Use the standard reasoning system
                reasoning = await self.reason([perception])
                
                # Create a simple plan
                plan = await self.plan("summarize findings", [reasoning])
                
                # Execute the plan
                execution_result = await self.execute(plan)
                
                # Store results
                self.memory.store_episodic([perception, reasoning, plan, execution_result])
                
                logger.info("Completed analysis cycle")
                await asyncio.sleep(15.0)
                
            except Exception as e:
                logger.error(f"Agent error: {e}")
                await asyncio.sleep(5.0)


# ===== Demo Functions =====

async def demonstrate_simple_aws_agent():
    """Demonstrate simple AWS agent."""
    print("\n=== Simple AWS Agent Demo ===")
    
    workspace = Path("./demo_workspace/aws_simple")
    workspace.mkdir(parents=True, exist_ok=True)
    
    # Create and run agent
    agent = SimpleAWSAgent()
    
    # Run for a limited time
    task = asyncio.create_task(agent.run())
    await asyncio.sleep(20.0)
    
    # Shutdown
    agent.shutdown()
    task.cancel()
    
    print("Simple AWS agent demo completed")


async def demonstrate_comprehensive_aws_agent():
    """Demonstrate comprehensive AWS agent."""
    print("\n=== Comprehensive AWS Agent Demo ===")
    
    workspace = Path("./demo_workspace/aws_full")
    workspace.mkdir(parents=True, exist_ok=True)
    
    # Create and run agent
    agent = ComprehensiveAWSAgent()
    
    # Run for a limited time
    task = asyncio.create_task(agent.run())
    await asyncio.sleep(30.0)
    
    # Shutdown
    agent.shutdown()
    task.cancel()
    
    print("Comprehensive AWS agent demo completed")


async def demonstrate_custom_aws_agent():
    """Demonstrate custom AWS agent."""
    print("\n=== Custom AWS Agent Demo ===")
    
    workspace = Path("./demo_workspace/aws_custom")
    workspace.mkdir(parents=True, exist_ok=True)
    
    # Create and run agent
    agent = CustomAWSAgent("custom_aws_agent", str(workspace))
    
    # Run for a limited time
    task = asyncio.create_task(agent.run())
    await asyncio.sleep(25.0)
    
    # Shutdown
    agent.shutdown()
    task.cancel()
    
    print("Custom AWS agent demo completed")


async def main():
    """Run all demonstrations."""
    print("AWS-Integrated MAIF Agent Framework Demo")
    print("=======================================")
    
    # Run demonstrations
    await demonstrate_simple_aws_agent()
    await demonstrate_comprehensive_aws_agent()
    await demonstrate_custom_aws_agent()
    
    print("\n=== All Demos Completed ===")
    print("\nKey Features Demonstrated:")
    print("- Simple AWS agent with Bedrock and S3 integration")
    print("- Comprehensive AWS agent with all AWS services integrated")
    print("- Custom AWS agent with selective AWS enhancements")
    print("- Decorator-based AWS integration for easy implementation")
    print("- AWS-enhanced perception, reasoning, and execution systems")


if __name__ == "__main__":
    asyncio.run(main())