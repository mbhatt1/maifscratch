"""
AWS Decorators for MAIF Agentic Framework
=========================================

Provides simple decorator-based APIs for integrating the MAIF agentic framework
with AWS services like Bedrock, KMS, S3, Lambda, and more.
"""

import functools
import asyncio
import boto3
from typing import Optional, Dict, Any, Callable, List, Type, Union
from pathlib import Path
import json
import time

from .agentic_framework import (
    MAIFAgent, AgentState, PerceptionSystem,
    ReasoningSystem, ExecutionSystem
)
from .aws_bedrock_integration import BedrockClient, MAIFBedrockIntegration
from .aws_kms_integration import create_kms_verifier, sign_block_data_with_kms

from maif_sdk.artifact import Artifact as MAIFArtifact
from maif_sdk.types import SecurityLevel


# ===== Enhanced AWS System Implementations =====

class AWSEnhancedPerceptionSystem(PerceptionSystem):
    """Perception system enhanced with AWS Bedrock capabilities."""
    
    def __init__(self, agent: MAIFAgent, region_name: str = "us-east-1"):
        super().__init__(agent)
        
        # Initialize Bedrock integration
        bedrock_client = BedrockClient(region_name=region_name)
        self.bedrock_integration = MAIFBedrockIntegration(bedrock_client)
    
    async def _process_text(self, text: str, artifact: MAIFArtifact):
        """Process text using AWS Bedrock."""
        # Generate embeddings using Bedrock
        embedding = self.bedrock_integration.embed_text(text)
        
        # Add to artifact
        artifact.add_text(text, title="Text Perception", language="en")
        
        if embedding:
            artifact.custom_metadata.update({
                "embedding": embedding,
                "perception_type": "text",
                "source": "aws_bedrock"
            })
    
    async def _process_image(self, image_data: bytes, artifact: MAIFArtifact):
        """Process image using AWS Bedrock."""
        # Analyze image using Bedrock
        description = self.bedrock_integration.analyze_image(image_data)
        
        # Add to artifact
        artifact.add_image(image_data, title="Image Perception", format="unknown")
        artifact.add_text(description, title="Image Description", language="en")
        
        artifact.custom_metadata["perception_type"] = "image"
        artifact.custom_metadata["source"] = "aws_bedrock"


class AWSEnhancedReasoningSystem(ReasoningSystem):
    """Reasoning system enhanced with AWS Bedrock capabilities."""
    
    def __init__(self, agent: MAIFAgent, region_name: str = "us-east-1"):
        super().__init__(agent)
        
        # Initialize Bedrock integration
        bedrock_client = BedrockClient(region_name=region_name)
        self.bedrock_integration = MAIFBedrockIntegration(bedrock_client)
    
    async def process(self, context: List[MAIFArtifact]) -> MAIFArtifact:
        """Apply reasoning using AWS Bedrock."""
        # Extract content from context
        texts = []
        for artifact in context:
            for content in artifact.get_content():
                if content['content_type'] == 'text':
                    texts.append(content['data'].decode('utf-8'))
        
        # Generate reasoning using Bedrock
        combined_text = "\n\n".join(texts)
        prompt = f"Analyze the following information and provide insights:\n\n{combined_text}"
        
        reasoning_block = self.bedrock_integration.generate_text_block(prompt)
        
        # Create reasoning artifact
        artifact = MAIFArtifact(
            name=f"reasoning_{int(time.time() * 1000000)}",
            client=self.agent.maif_client,
            security_level=SecurityLevel.CONFIDENTIAL
        )
        
        artifact.add_text(
            reasoning_block["text"],
            title="Reasoning Result",
            language="en"
        )
        
        # Add metadata
        artifact.custom_metadata.update({
            "source": "aws_bedrock",
            "model": reasoning_block["metadata"]["model"],
            "timestamp": time.time()
        })
        
        # Save to knowledge base
        artifact.save(self.agent.knowledge_path)
        
        return artifact


class AWSExecutionSystem(ExecutionSystem):
    """Execution system enhanced with AWS service capabilities."""
    
    def __init__(self, agent: MAIFAgent, region_name: str = "us-east-1"):
        super().__init__(agent)
        
        # Add AWS-specific executors
        self.executors.update({
            "invoke_lambda": self._invoke_lambda,
            "store_in_s3": self._store_in_s3,
            "query_dynamodb": self._query_dynamodb,
            "generate_image": self._generate_image_with_bedrock
        })
        
        # Initialize AWS clients
        session = boto3.Session(region_name=region_name)
        self.lambda_client = session.client('lambda')
        self.s3_client = session.client('s3')
        self.dynamodb = session.resource('dynamodb')
        
        # Initialize Bedrock
        bedrock_client = BedrockClient(region_name=region_name)
        self.bedrock_integration = MAIFBedrockIntegration(bedrock_client)
    
    async def _invoke_lambda(self, parameters: Dict) -> Dict:
        """Invoke AWS Lambda function."""
        function_name = parameters.get('function_name')
        payload = parameters.get('payload', {})
        
        try:
            response = self.lambda_client.invoke(
                FunctionName=function_name,
                Payload=json.dumps(payload)
            )
            
            return {
                "status": "success",
                "response": json.loads(response['Payload'].read())
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def _store_in_s3(self, parameters: Dict) -> Dict:
        """Store data in S3."""
        bucket = parameters.get('bucket')
        key = parameters.get('key')
        data = parameters.get('data')
        content_type = parameters.get('content_type', 'application/octet-stream')
        
        try:
            import io
            data_bytes = data.encode('utf-8') if isinstance(data, str) else data
            self.s3_client.upload_fileobj(
                io.BytesIO(data_bytes),
                bucket,
                key,
                ExtraArgs={'ContentType': content_type}
            )
            return {
                "status": "success",
                "location": f"s3://{bucket}/{key}"
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def _query_dynamodb(self, parameters: Dict) -> Dict:
        """Query DynamoDB table."""
        table_name = parameters.get('table_name')
        query_type = parameters.get('query_type', 'scan')
        query_params = parameters.get('query_params', {})
        
        try:
            table = self.dynamodb.Table(table_name)
            
            if query_type == 'get_item':
                response = table.get_item(**query_params)
                result = response.get('Item')
            elif query_type == 'query':
                response = table.query(**query_params)
                result = response.get('Items', [])
            else:  # scan
                response = table.scan(**query_params)
                result = response.get('Items', [])
                
            return {
                "status": "success",
                "result": result
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def _generate_image_with_bedrock(self, parameters: Dict) -> Dict:
        """Generate image using Bedrock."""
        prompt = parameters.get('prompt', '')
        width = parameters.get('width', 1024)
        height = parameters.get('height', 1024)
        
        image_block = self.bedrock_integration.generate_image_block(
            prompt, width=width, height=height
        )
        
        return {
            "status": "success",
            "image_data": image_block["image_data"]
        }


# ===== Agent Creation Decorators =====

def maif_agent(agent_class: Type = None, **config):
    """
    Decorator to easily create a MAIF agent class.
    
    @maif_agent(workspace="./agent_data")
    class MyAgent:
        def process(self, data):
            # Your logic here
    """
    def decorator(cls):
        # Create a new class that inherits from both the original class and MAIFAgent
        class MixedAgent(cls, MAIFAgent):
            def __init__(self, agent_id="default_agent", workspace_path="./workspace", **kwargs):
                # Initialize MAIFAgent
                MAIFAgent.__init__(self, agent_id, workspace_path, config)
                
                # Initialize the original class
                cls.__init__(self, **kwargs)
                
            async def run(self):
                # Default implementation that can be overridden
                while self.state != AgentState.TERMINATED:
                    try:
                        # Call the process method from the original class if it exists
                        if hasattr(self, 'process'):
                            await self.process()
                        await asyncio.sleep(5.0)
                    except Exception as e:
                        print(f"Agent error: {e}")
                        await asyncio.sleep(10.0)
        
        return MixedAgent
    
    # Handle case where decorator is used without parentheses
    if agent_class is not None:
        return decorator(agent_class)
    return decorator


# ===== AWS Integration Decorators =====

def aws_bedrock(region_name: str = "us-east-1", profile_name: Optional[str] = None):
    """
    Decorator to add AWS Bedrock capabilities to a method.
    
    @aws_bedrock()
    def generate_text(self, prompt):
        # 'bedrock' is injected into the method
        return bedrock.invoke_text_model("anthropic.claude-v2", prompt)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Create Bedrock client if not already available
            if not hasattr(self, '_bedrock_client'):
                self._bedrock_client = BedrockClient(region_name=region_name, profile_name=profile_name)
                self._bedrock_integration = MAIFBedrockIntegration(self._bedrock_client)
            
            # Inject bedrock client into the function call
            kwargs['bedrock'] = self._bedrock_integration
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


def aws_kms(region_name: str = "us-east-1", profile_name: Optional[str] = None):
    """
    Decorator to add AWS KMS capabilities to a method.
    
    @aws_kms()
    def sign_data(self, data):
        # 'key_store' and 'verifier' are injected into the method
        return verifier.sign_data(data, key_id)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Create KMS verifier if not already available
            if not hasattr(self, '_kms_key_store') or not hasattr(self, '_kms_verifier'):
                self._kms_key_store, self._kms_verifier = create_kms_verifier(
                    region_name=region_name, 
                    profile_name=profile_name
                )
            
            # Inject KMS components into the function call
            kwargs['key_store'] = self._kms_key_store
            kwargs['verifier'] = self._kms_verifier
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


def aws_s3(region_name: str = "us-east-1", profile_name: Optional[str] = None):
    """
    Decorator to add AWS S3 capabilities to a method.
    
    @aws_s3()
    def store_artifact(self, artifact, bucket_name):
        # 's3_client' is injected into the method
        s3_client.upload_fileobj(...)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Create S3 client if not already available
            if not hasattr(self, '_s3_client'):
                session = boto3.Session(region_name=region_name, profile_name=profile_name)
                self._s3_client = session.client('s3')
            
            # Inject S3 client into the function call
            kwargs['s3_client'] = self._s3_client
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


def aws_lambda(region_name: str = "us-east-1", profile_name: Optional[str] = None):
    """
    Decorator to add AWS Lambda capabilities to a method.
    
    @aws_lambda()
    def invoke_function(self, function_name, payload):
        # 'lambda_client' is injected into the method
        lambda_client.invoke(...)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Create Lambda client if not already available
            if not hasattr(self, '_lambda_client'):
                session = boto3.Session(region_name=region_name, profile_name=profile_name)
                self._lambda_client = session.client('lambda')
            
            # Inject Lambda client into the function call
            kwargs['lambda_client'] = self._lambda_client
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


def aws_dynamodb(region_name: str = "us-east-1", profile_name: Optional[str] = None):
    """
    Decorator to add AWS DynamoDB capabilities to a method.
    
    @aws_dynamodb()
    def query_table(self, table_name, key):
        # 'dynamodb' is injected into the method
        table = dynamodb.Table(table_name)
        return table.get_item(...)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Create DynamoDB resource if not already available
            if not hasattr(self, '_dynamodb'):
                session = boto3.Session(region_name=region_name, profile_name=profile_name)
                self._dynamodb = session.resource('dynamodb')
            
            # Inject DynamoDB resource into the function call
            kwargs['dynamodb'] = self._dynamodb
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


def aws_step_functions(region_name: str = "us-east-1", profile_name: Optional[str] = None):
    """
    Decorator to add AWS Step Functions capabilities to a method.
    
    @aws_step_functions()
    def execute_workflow(self, state_machine_arn, input_data):
        # 'sfn_client' is injected into the method
        execution = sfn_client.start_execution(
            stateMachineArn=state_machine_arn,
            input=json.dumps(input_data)
        )
        return execution['executionArn']
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Create Step Functions client if not already available
            if not hasattr(self, '_sfn_client'):
                session = boto3.Session(region_name=region_name, profile_name=profile_name)
                self._sfn_client = session.client('stepfunctions')
            
            # Inject Step Functions client into the function call
            kwargs['sfn_client'] = self._sfn_client
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


# ===== Agent System Enhancement Decorators =====

def enhance_perception_with_bedrock(region_name: str = "us-east-1"):
    """
    Decorator to enhance an agent class with Bedrock-powered perception.
    
    @enhance_perception_with_bedrock()
    class MyAgent(MAIFAgent):
        # Agent implementation
    """
    def decorator(cls):
        original_init = cls.__init__
        
        def new_init(self, *args, **kwargs):
            # Call original init
            original_init(self, *args, **kwargs)
            
            # Replace perception system with enhanced version
            self.perception = AWSEnhancedPerceptionSystem(self, region_name)
        
        cls.__init__ = new_init
        return cls
    
    return decorator


def enhance_reasoning_with_bedrock(region_name: str = "us-east-1"):
    """
    Decorator to enhance an agent class with Bedrock-powered reasoning.
    
    @enhance_reasoning_with_bedrock()
    class MyAgent(MAIFAgent):
        # Agent implementation
    """
    def decorator(cls):
        original_init = cls.__init__
        
        def new_init(self, *args, **kwargs):
            # Call original init
            original_init(self, *args, **kwargs)
            
            # Replace reasoning system with enhanced version
            self.reasoning = AWSEnhancedReasoningSystem(self, region_name)
        
        cls.__init__ = new_init
        return cls
    
    return decorator


def enhance_execution_with_aws(region_name: str = "us-east-1"):
    """
    Decorator to enhance an agent class with AWS-powered execution.
    
    @enhance_execution_with_aws()
    class MyAgent(MAIFAgent):
        # Agent implementation
    """
    def decorator(cls):
        original_init = cls.__init__
        
        def new_init(self, *args, **kwargs):
            # Call original init
            original_init(self, *args, **kwargs)
            
            # Replace execution system with enhanced version
            self.execution = AWSExecutionSystem(self, region_name)
        
        cls.__init__ = new_init
        return cls
    
    return decorator


# ===== Step Functions Workflow System =====

class StepFunctionsWorkflowSystem:
    """Workflow system that uses AWS Step Functions for orchestration."""
    
    def __init__(self, agent: MAIFAgent, region_name: str = "us-east-1"):
        """Initialize Step Functions workflow system."""
        self.agent = agent
        
        # Initialize AWS Step Functions client
        session = boto3.Session(region_name=region_name)
        self.sfn_client = session.client('stepfunctions')
        
        # Store workflow definitions and executions
        self.workflows = {}
        self.executions = {}
    
    def register_workflow(self, name: str, state_machine_arn: str, description: str = ""):
        """Register a Step Functions workflow."""
        self.workflows[name] = {
            "state_machine_arn": state_machine_arn,
            "description": description,
            "registered_at": time.time()
        }
    
    async def execute_workflow(self, workflow_name: str, input_data: Dict[str, Any]) -> str:
        """Execute a Step Functions workflow."""
        if workflow_name not in self.workflows:
            raise ValueError(f"Workflow '{workflow_name}' not registered")
        
        state_machine_arn = self.workflows[workflow_name]["state_machine_arn"]
        
        # Start execution
        response = self.sfn_client.start_execution(
            stateMachineArn=state_machine_arn,
            input=json.dumps(input_data)
        )
        
        execution_arn = response["executionArn"]
        
        # Store execution details
        self.executions[execution_arn] = {
            "workflow_name": workflow_name,
            "started_at": time.time(),
            "status": "RUNNING",
            "input": input_data
        }
        
        return execution_arn
    
    async def check_execution_status(self, execution_arn: str) -> Dict[str, Any]:
        """Check the status of a workflow execution."""
        response = self.sfn_client.describe_execution(
            executionArn=execution_arn
        )
        
        # Update execution status
        if execution_arn in self.executions:
            self.executions[execution_arn]["status"] = response["status"]
            
            if "output" in response and response["status"] == "SUCCEEDED":
                self.executions[execution_arn]["output"] = json.loads(response["output"])
        
        return {
            "execution_arn": execution_arn,
            "status": response["status"],
            "started_at": response["startDate"].timestamp(),
            "output": json.loads(response["output"]) if "output" in response else None
        }
    
    async def wait_for_execution(self, execution_arn: str, timeout: float = 300.0) -> Dict[str, Any]:
        """Wait for a workflow execution to complete."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = await self.check_execution_status(execution_arn)
            
            if status["status"] in ["SUCCEEDED", "FAILED", "TIMED_OUT", "ABORTED"]:
                return status
            
            # Wait before checking again
            await asyncio.sleep(2.0)
        
        # Timeout reached
        return {
            "execution_arn": execution_arn,
            "status": "TIMEOUT",
            "error": f"Execution did not complete within {timeout} seconds"
        }


# ===== Step Functions Enhancement Decorator =====

def enhance_with_step_functions(region_name: str = "us-east-1"):
    """
    Decorator to enhance an agent class with Step Functions workflow capabilities.
    
    @enhance_with_step_functions()
    class MyAgent(MAIFAgent):
        # Agent implementation
    """
    def decorator(cls):
        original_init = cls.__init__
        
        def new_init(self, *args, **kwargs):
            # Call original init
            original_init(self, *args, **kwargs)
            
            # Add workflow system
            self.workflow = StepFunctionsWorkflowSystem(self, region_name)
        
        cls.__init__ = new_init
        return cls
    
    return decorator


# ===== Full AWS Agent Decorator =====

def aws_agent(region_name: str = "us-east-1", profile_name: Optional[str] = None, **config):
    """
    Comprehensive decorator to create a fully AWS-integrated MAIF agent.
    
    @aws_agent(workspace="./agent_data")
    class MyAgent:
        def process(self):
            # Your logic here
    """
    def decorator(cls):
        # First apply the maif_agent decorator
        agent_cls = maif_agent(**config)(cls)
        
        # Then apply all AWS enhancement decorators
        agent_cls = enhance_perception_with_bedrock(region_name)(agent_cls)
        agent_cls = enhance_reasoning_with_bedrock(region_name)(agent_cls)
        agent_cls = enhance_execution_with_aws(region_name)(agent_cls)
        agent_cls = enhance_with_step_functions(region_name)(agent_cls)
        
        return agent_cls
    
    return decorator