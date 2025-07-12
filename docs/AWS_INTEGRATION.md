# AWS Integration with MAIF Agentic Framework

This comprehensive guide covers all AWS integrations available in the MAIF agentic framework for production use.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [Available Decorators](#available-decorators)
5. [Core Features](#core-features)
6. [AWS Service Integrations](#aws-service-integrations)
7. [Agent Lifecycle Management](#agent-lifecycle-management)
8. [Deployment](#deployment)
9. [Monitoring & Observability](#monitoring--observability)
10. [Security Best Practices](#security-best-practices)
11. [Error Handling](#error-handling)
12. [Cost Optimization](#cost-optimization)
13. [Troubleshooting](#troubleshooting)

## Overview

The MAIF agentic framework provides production-ready AWS integrations with:

- **Automatic AWS Service Integration**: Agents automatically use AWS services when `use_aws=True`
- **Production Features**: Retry logic, connection pooling, error handling, monitoring
- **State Management**: Automatic state dumps and restoration from S3
- **Deployment Tools**: CloudFormation, CDK, Lambda, ECS/Fargate support
- **Observability**: X-Ray tracing, CloudWatch logging, custom metrics
- **Security**: KMS encryption, IAM roles, Secrets Manager integration

## Quick Start

```python
from maif.agentic_framework import MAIFAgent
from maif_sdk.aws_backend import AWSConfig

# Create an AWS-enabled agent
class MyAgent(MAIFAgent):
    async def run(self):
        # Your agent logic here
        pass

# Initialize with AWS
agent = MyAgent(
    agent_id="my-agent",
    workspace_path="./workspace",
    use_aws=True,  # Enable AWS integration
    aws_config=AWSConfig(
        region_name="us-east-1",
        s3_bucket="my-maif-bucket"
    )
)

# AWS services automatically start up
await agent.initialize()

# Run agent
await agent.run()

# Shutdown dumps state to S3
agent.shutdown()
```

## Configuration

### AWS Configuration Options

```python
from maif_sdk.aws_backend import AWSConfig

config = AWSConfig(
    # Required
    region_name="us-east-1",              # AWS region
    
    # Storage
    s3_bucket="maif-artifacts",           # S3 bucket for artifacts
    s3_prefix="agents/",                  # S3 key prefix
    
    # Security
    kms_key_id="alias/maif-key",         # KMS key for encryption
    use_encryption=True,                  # Enable KMS encryption
    
    # Compliance
    use_compliance_logging=True,          # Enable CloudWatch compliance logs
    compliance_log_group="/maif/compliance",
    
    # Monitoring
    enable_xray=True,                     # Enable X-Ray tracing
    xray_sampling_rate=0.1,              # Sample 10% of requests
    
    # Credentials (optional - uses IAM role by default)
    profile_name=None,                    # AWS profile name
    access_key_id=None,                   # Explicit credentials
    secret_access_key=None
)
```

### Environment Variables

```bash
# AWS credentials (if not using IAM roles)
export AWS_DEFAULT_REGION=us-east-1
export AWS_PROFILE=myprofile

# MAIF-specific
export MAIF_S3_BUCKET=my-maif-bucket
export MAIF_USE_AWS=true
export MAIF_KMS_KEY_ID=alias/maif-key
```

## Available Decorators

### Agent Creation Decorators

| Decorator | Description |
|-----------|-------------|
| `@maif_agent(**config)` | Create a MAIF agent from any class. Supports `use_aws=True` to enable AWS backends |
| `@aws_agent(**config)` | Create a fully AWS-integrated MAIF agent with AWS backends enabled by default |

### AWS Service Decorators

| Decorator | Description |
|-----------|-------------|
| `@aws_bedrock()` | Add AWS Bedrock capabilities to a method |
| `@aws_kms()` | Add AWS KMS capabilities to a method |
| `@aws_s3()` | Add AWS S3 capabilities to a method |
| `@aws_lambda()` | Add AWS Lambda capabilities to a method |
| `@aws_dynamodb()` | Add AWS DynamoDB capabilities to a method |
| `@aws_step_functions()` | Add AWS Step Functions capabilities to a method |

### AWS X-Ray Tracing Decorators

| Decorator | Description |
|-----------|-------------|
| `@aws_xray()` | Add X-Ray tracing to a class or method |
| `@xray_trace()` | Create a custom X-Ray trace segment |
| `@xray_subsegment()` | Create an X-Ray subsegment |
| `@trace_aws_call()` | Trace AWS service calls specifically |

### Agent System Enhancement Decorators

| Decorator | Description |
|-----------|-------------|
| `@enhance_perception_with_bedrock()` | Enhance agent perception with AWS Bedrock |
| `@enhance_reasoning_with_bedrock()` | Enhance agent reasoning with AWS Bedrock |
| `@enhance_execution_with_aws()` | Enhance agent execution with AWS services |
| `@enhance_with_step_functions()` | Enhance agent with Step Functions workflow capabilities |

## Usage Examples

### Simple AWS-Enhanced Agent

```python
from maif.aws_decorators import maif_agent, aws_bedrock, aws_s3

@maif_agent(workspace="./agent_workspace")
class SimpleAgent:
    def __init__(self, name="SimpleAgent"):
        self.name = name
    
    @aws_bedrock()
    def generate_response(self, prompt, bedrock=None):
        # bedrock is automatically injected
        return bedrock.generate_text_block(prompt)
    
    @aws_s3()
    def store_result(self, text, bucket_name, s3_client=None):
        # s3_client is automatically injected
        key = f"results/{int(time.time())}.txt"
        s3_client.put_object(
            Bucket=bucket_name,
            Key=key,
            Body=text.encode('utf-8')
        )
        return f"s3://{bucket_name}/{key}"
    
    async def process(self):
        # Your agent logic here
        response = self.generate_response("What's the weather today?")
        s3_path = self.store_result(response["text"], "my-bucket")
```

### Agent with AWS Backends Enabled

```python
from maif.aws_decorators import maif_agent
from maif_sdk.types import SecurityLevel

@maif_agent(workspace="./agent_workspace", use_aws=True)
class AWSBackendAgent:
    async def process(self, data: str):
        # When use_aws=True, the agent automatically uses:
        # - S3 for artifact storage
        # - KMS for encryption
        # - Secrets Manager for key management
        # - CloudWatch for compliance logging
        # - Macie for privacy classification
        
        client = self.maif_client  # Pre-configured with AWS backends
        
        # Create artifact - stored in S3, encrypted with KMS
        artifact = await client.create_artifact("analysis", SecurityLevel.L4_REGULATED)
        artifact.add_text(data, title="Input Data")
        
        # Save to S3 (happens automatically)
        artifact_id = await client.write_content(artifact)
        
        # Read from S3 (happens automatically)
        retrieved = await client.read_content(artifact_id)
        
        return artifact_id
```

### Fully AWS-Integrated Agent

```python
from maif.aws_decorators import aws_agent
from maif_sdk.aws_backend import AWSConfig

# aws_agent has use_aws=True by default
@aws_agent(workspace="./agent_workspace")
class FullAWSAgent:
    async def process(self):
        # Uses AWS-enhanced systems automatically:
        # - Bedrock for perception and reasoning
        # - S3 for artifact storage
        # - KMS for encryption
        # - Step Functions for workflows
        
        perception = await self.perceive("Analyze this data", "text")
        reasoning = await self.reason([perception])
        plan = await self.plan("create summary", [reasoning])
        result = await self.execute(plan)
        
        # All operations use AWS services under the hood

# With custom AWS configuration
custom_config = AWSConfig(
    region_name="us-west-2",
    s3_bucket="my-custom-bucket",
    kms_key_id="alias/my-custom-key"
)

@aws_agent(workspace="./agent_workspace", aws_config=custom_config)
class CustomConfigAWSAgent:
    async def process(self):
        # Uses custom AWS configuration
        pass
```

### Agent with X-Ray Tracing

```python
from maif.aws_decorators import aws_agent, xray_trace, xray_subsegment

# X-Ray tracing enabled by default for aws_agent
@aws_agent(workspace="./agent_workspace", xray_service_name="MyTracedAgent")
class TracedAgent:
    @xray_subsegment("data_validation")
    async def validate(self, data):
        # This operation appears as a subsegment in X-Ray
        return validate_data(data)
    
    @xray_trace("complex_operation")
    async def process(self):
        # Creates a new trace segment
        data = await self.validate({"test": "data"})
        
        # Add custom annotations
        if hasattr(self, '_xray_integration'):
            self._xray_integration.add_annotation('operation_type', 'process')
            self._xray_integration.add_metadata('input_data', data)
        
        return await self.execute(data)
```

### Custom AWS-Enhanced Agent

```python
from maif.agentic_framework import MAIFAgent
from maif.aws_decorators import enhance_perception_with_bedrock, enhance_reasoning_with_bedrock

@enhance_perception_with_bedrock()
@enhance_reasoning_with_bedrock()
class CustomAWSAgent(MAIFAgent):
    async def run(self):
        while self.state != AgentState.TERMINATED:
            try:
                # Use AWS-enhanced perception and reasoning
                perception = await self.perceive("Input data", "text")
                reasoning = await self.reason([perception])
                
                # Use standard planning and execution
                plan = await self.plan("analyze data", [reasoning])
                result = await self.execute(plan)
                
                await asyncio.sleep(5.0)
            except Exception as e:
                print(f"Error: {e}")
```

## AWS Service Integration Details

### AWS Bedrock Integration

The `@aws_bedrock()` decorator provides access to AWS Bedrock for:

- Text generation with Claude, Titan, and other models
- Embedding generation for semantic search
- Image generation with Stable Diffusion
- Image analysis and understanding

Example:

```python
@aws_bedrock()
def analyze_image(self, image_data, bedrock=None):
    # Generate description of image
    description = bedrock.analyze_image(image_data)
    
    # Generate embeddings for the description
    embedding = bedrock.embed_text(description)
    
    return {
        "description": description,
        "embedding": embedding
    }
```

### AWS KMS Integration

The `@aws_kms()` decorator provides access to AWS KMS for:

- Secure key management
- Digital signatures
- Verification operations

Example:

```python
@aws_kms()
def sign_data(self, data, key_store=None, verifier=None):
    # List available keys
    keys = key_store.list_kms_keys()
    key_id = keys[0]['KeyId']
    
    # Sign data
    signature_metadata = sign_block_data_with_kms(
        verifier, data.encode(), key_id
    )
    
    return signature_metadata
```

### AWS S3 Integration

The `@aws_s3()` decorator provides access to AWS S3 for:

- Storing artifacts
- Retrieving artifacts
- Managing S3 buckets

Example:

```python
@aws_s3()
def store_artifact(self, artifact, bucket_name, s3_client=None):
    # Save artifact to file
    filepath = artifact.save("temp.maif")
    
    # Upload to S3
    s3_client.upload_file(
        filepath, 
        bucket_name, 
        f"artifacts/{artifact.name}.maif"
    )
    
    return f"s3://{bucket_name}/artifacts/{artifact.name}.maif"
```

## Enhanced Agent Systems

### AWS-Enhanced Perception System

The `AWSEnhancedPerceptionSystem` uses AWS Bedrock to:

- Generate embeddings for text
- Analyze images
- Process audio
- Handle multimodal inputs

### AWS-Enhanced Reasoning System

The `AWSEnhancedReasoningSystem` uses AWS Bedrock to:

- Analyze context
- Generate insights
- Create reasoning artifacts

### AWS-Enhanced Execution System

The `AWSExecutionSystem` provides executors for:

- Invoking Lambda functions
- Storing data in S3
- Querying DynamoDB
- Generating images with Bedrock

## Configuration

All decorators accept AWS configuration parameters:

```python
@aws_bedrock(
    region_name="us-west-2",
    profile_name="my-profile"
)
def my_method(self, bedrock=None):
    # Use bedrock with custom configuration
```

## AWS Step Functions Integration

The AWS Step Functions integration allows you to orchestrate complex agent workflows using AWS Step Functions.

### Using the Step Functions Decorator

```python
@aws_step_functions()
def execute_workflow(self, state_machine_arn, input_data, sfn_client=None):
    # Start execution
    execution = sfn_client.start_execution(
        stateMachineArn=state_machine_arn,
        input=json.dumps(input_data)
    )
    return execution['executionArn']
```

### Using the Workflow System

```python
@enhance_with_step_functions()
class WorkflowAgent(MAIFAgent):
    async def run(self):
        # Register workflows
        self.workflow.register_workflow(
            "data_processing",
            "arn:aws:states:us-east-1:123456789012:stateMachine:DataProcessing",
            "Process data with Step Functions"
        )
        
        # Execute workflow
        execution_arn = await self.workflow.execute_workflow(
            "data_processing",
            {"input": "data to process"}
        )
        
        # Wait for completion
        result = await self.workflow.wait_for_execution(execution_arn)
        
        # Process result
        if result["status"] == "SUCCEEDED":
            print(f"Workflow succeeded: {result['output']}")
```

## Complete Example

See the [AWS Agent Demo](../examples/aws_agent_demo.py) for a complete example of using the AWS decorators with the MAIF agentic framework.

## AWS Credentials

The decorators use the standard AWS credential chain:

1. Environment variables
2. Shared credential file (~/.aws/credentials)
3. AWS IAM role for EC2/ECS

Make sure your AWS credentials are properly configured before using these decorators.

## Bedrock Agent Swarm

The MAIF framework provides support for creating a swarm of agents using different AWS Bedrock models that share the same MAIF storage.

### Overview

The Bedrock Agent Swarm allows you to:

1. Create multiple agents with different Bedrock models
2. Share MAIF storage across all agents
3. Distribute tasks to appropriate agents
4. Aggregate results from multiple models
5. Create specialized agent swarms for complex tasks

### Using the Bedrock Agent Swarm

```python
from maif.bedrock_swarm import BedrockAgentSwarm
from maif.aws_bedrock_integration import BedrockModelProvider

# Create agent swarm
swarm = BedrockAgentSwarm("./workspace")

# Add agents with different models
swarm.add_agent_with_model(
    "claude_agent",
    BedrockModelProvider.ANTHROPIC,
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "us-east-1"
)

swarm.add_agent_with_model(
    "titan_agent",
    BedrockModelProvider.AMAZON,
    "amazon.titan-text-express-v1",
    "us-east-1"
)

# Start swarm
swarm_task = asyncio.create_task(swarm.run())

# Submit task to swarm
task_id = await swarm.submit_task({
    "task_id": "task_1",
    "type": "all",
    "data": "Analyze the benefits of using multiple AI models.",
    "input_type": "text",
    "goal": "provide analysis",
    "aggregation": "all"
})

# Get result
result = await swarm.get_result(task_id, timeout=120.0)
```

### Task Types

The Bedrock Agent Swarm supports different task types:

Task Type | Description |
|-----------|-------------|
`all` | Send task to all agents in the swarm |
`provider` | Send task to agents from a specific provider |
`specific` | Send task to specific agents by ID |

### Result Aggregation

Results can be aggregated in different ways:

Aggregation Method | Description |
|-------------------|-------------|
`all` | Return all individual agent results |
`vote` | Use a voting mechanism to determine consensus |
`weighted_vote` | Use weighted voting based on model provider and confidence |
`ensemble` | Combine insights from all models with attribution |
`semantic_merge` | Group and merge results by provider with semantic organization |

### Example

See the [Bedrock Swarm Demo](../examples/bedrock_swarm_demo.py) for a complete example of using the Bedrock Agent Swarm.

### Benefits of Agent Swarms

Using a swarm of agents with different models provides several benefits:

1. **Diverse Perspectives**: Different models have different strengths and biases
2. **Consensus Building**: Multiple models can vote on the best answer
3. **Specialized Roles**: Models can be assigned to tasks that match their strengths
4. **Redundancy**: If one model fails, others can still complete the task
5. **Shared Knowledge**: All agents share the same MAIF storage

### Implementation Details

The Bedrock Agent Swarm is implemented using:

- `BedrockAgentSwarm`: Extends `MAIFAgentConsortium` for specialized functionality
- `BedrockAgentFactory`: Creates agents with different Bedrock models
- Shared MAIF storage for knowledge exchange between agents
- Task distribution and result aggregation mechanisms

## AWS X-Ray Integration

The MAIF framework provides comprehensive AWS X-Ray integration for distributed tracing of agent operations.

### Overview

AWS X-Ray integration enables:

1. Distributed tracing across agent operations
2. Performance monitoring and optimization
3. Error tracking and debugging
4. Service map visualization
5. Custom annotations and metadata

### Enabling X-Ray Tracing

#### For Individual Agents

```python
from maif.aws_decorators import maif_agent, aws_agent

# Enable X-Ray for maif_agent
@maif_agent(
    workspace="./agent_data",
    use_aws=True,
    enable_xray=True,
    xray_service_name="MyTracedAgent"
)
class TracedAgent:
    async def process(self, data):
        # All operations are automatically traced
        pass

# AWS agent has X-Ray enabled by default
@aws_agent(
    workspace="./agent_data",
    xray_service_name="MyAWSAgent"
)
class AWSTracedAgent:
    async def process(self):
        # Full AWS integration with X-Ray tracing
        pass
```

#### Using X-Ray Decorators

```python
from maif.aws_decorators import aws_xray, xray_trace, xray_subsegment, trace_aws_call

# Trace an entire class
@aws_xray(service_name="DataProcessor", sampling_rate=0.5)
class DataProcessor:
    def process(self, data):
        # All methods are traced
        pass

# Trace specific methods
class MyAgent:
    @xray_trace("custom_operation")
    async def complex_operation(self, data):
        # Creates a new trace segment
        result = await self.step1(data)
        return await self.step2(result)
    
    @xray_subsegment("validation")
    async def validate_data(self, data):
        # Creates a subsegment under current trace
        return validate(data)
    
    @trace_aws_call("s3")
    async def upload_to_s3(self, bucket, key, data):
        # Specifically traces AWS service calls
        s3_client.put_object(Bucket=bucket, Key=key, Body=data)
```

### Adding Custom Metadata

```python
from maif.aws_xray_integration import xray

class TracedAgent:
    async def process(self, data):
        # Add annotations (indexed, searchable)
        xray.current_segment().put_annotation('user_id', data['user_id'])
        xray.current_segment().put_annotation('operation_type', 'process')
        
        # Add metadata (not indexed, for debugging)
        xray.current_segment().put_metadata('input_data', data)
        xray.current_segment().put_metadata('config', self.config)
        
        # Process data
        result = await self.analyze(data)
        
        # Add result metadata
        xray.current_segment().put_metadata('result', result)
        
        return result
```

### Distributed Tracing Across Agents

```python
from maif.aws_decorators import aws_agent
from maif.aws_xray_integration import MAIFXRayIntegration

@aws_agent(workspace="./collector", xray_service_name="Collector")
class CollectorAgent:
    async def collect(self, source):
        # Get trace header for propagation
        xray_integration = self._xray_integration
        trace_header = xray_integration.get_trace_header()
        
        # Collect data with trace context
        data = await self.fetch_data(source)
        data['trace_id'] = trace_header
        
        return data

@aws_agent(workspace="./processor", xray_service_name="Processor")
class ProcessorAgent:
    async def process(self, data):
        # Continue the distributed trace
        if 'trace_id' in data:
            xray.begin_segment(
                name="process_data",
                trace_id=data['trace_id']
            )
        
        # Process continues the same trace
        result = await self.analyze(data)
        
        xray.end_segment()
        return result
```

### X-Ray Configuration

```python
from maif.aws_xray_integration import MAIFXRayIntegration

# Configure X-Ray integration
xray_integration = MAIFXRayIntegration(
    service_name="MyService",
    region_name="us-east-1",
    sampling_rate=0.1,  # Trace 10% of requests
    daemon_address="127.0.0.1:2000"  # X-Ray daemon address
)

# Use with decorators
@xray_integration.trace_agent_operation("custom_op")
async def my_operation(data):
    # Operation is traced
    pass
```

### X-Ray Best Practices

1. **Sampling Rate**: Use appropriate sampling rates for production (typically 0.1-0.5)
2. **Annotations**: Use annotations for searchable fields (user_id, operation_type)
3. **Metadata**: Use metadata for debugging information (input data, configs)
4. **Subsegments**: Create subsegments for logical operations within traces
5. **Error Handling**: X-Ray automatically captures exceptions with stack traces

### Viewing Traces

Access the AWS X-Ray console to:

1. View the service map showing agent interactions
2. Analyze trace timelines with detailed segments
3. Search traces by annotations
4. Monitor performance metrics
5. Debug errors with captured exceptions

### Running X-Ray Daemon

For local development, run the X-Ray daemon:

```bash
# Using Docker
docker run -p 2000:2000/udp amazon/aws-xray-daemon -o

# Or download and run directly
wget https://s3.us-east-2.amazonaws.com/aws-xray-assets.us-east-2/xray-daemon/aws-xray-daemon-3.x.zip
unzip aws-xray-daemon-3.x.zip
./xray -o -n us-east-1
```

For production, X-Ray daemon is pre-installed on:
- AWS EC2 (with X-Ray integration)
- AWS ECS/Fargate
- AWS Lambda (automatic)

### Complete X-Ray Example

See the [AWS X-Ray Agent Demo](../examples/aws_xray_agent_demo.py) for a comprehensive example of using X-Ray tracing with MAIF agents.

## Agent Lifecycle Management

### Automatic State Preservation

Agents automatically dump their complete state to MAIF when shutting down:

```python
# State is automatically saved on shutdown
agent.shutdown()  # Creates comprehensive MAIF dump in S3
```

### State Restoration

Resume agents from previous state dumps:

```python
# Method 1: Restore on initialization
restored_agent = MyAgent(
    agent_id="my-agent",
    workspace_path="./workspace",
    use_aws=True,
    restore_from="./dumps/agent_state.maif"  # or S3 artifact ID
)

# Method 2: Create from dump
agent = MyAgent.from_dump(
    dump_path="s3://bucket/agent_dump.maif",
    use_aws=True,
    aws_config=aws_config
)
```

### Checkpointing

Create periodic checkpoints for long-running agents:

```python
class CheckpointAgent(MAIFAgent):
    def create_checkpoint(self):
        return self.dump_complete_state()
    
    async def run(self):
        for i in range(100):
            # Process work
            if i % 10 == 0:
                checkpoint = self.create_checkpoint()
                print(f"Checkpoint saved: {checkpoint}")
```

## Deployment

### Lambda Deployment

Deploy agents to AWS Lambda for event-driven processing:

```python
from maif.aws_deployment import deploy_agent_to_lambda

# Deploy agent
deployment = deploy_agent_to_lambda(
    agent_name="my-processing-agent",
    agent_class="ProcessingAgent",
    agent_module="myapp.agents",
    s3_bucket="maif-deployments",
    memory_mb=1024,
    timeout_seconds=300,
    environment_vars={
        "LOG_LEVEL": "INFO"
    }
)

# Deployment creates:
# - Lambda function with handler
# - CloudFormation template
# - IAM roles with required permissions
# - CloudWatch log groups
```

### ECS/Fargate Deployment

Deploy long-running agents to ECS:

```python
from maif.aws_deployment import deploy_agent_to_ecs

deployment = deploy_agent_to_ecs(
    agent_name="continuous-agent",
    agent_class="ContinuousAgent",
    agent_module="myapp.agents",
    ecr_repository="123456789012.dkr.ecr.us-east-1.amazonaws.com/agents",
    ecs_cluster="maif-agents",
    memory_mb=2048
)

# Creates:
# - Dockerfile
# - ECS task definition
# - ECS service
# - CloudFormation template
```

### CDK Deployment

Generate AWS CDK projects for infrastructure as code:

```python
from maif.aws_deployment import DeploymentConfig, DeploymentManager

config = DeploymentConfig(
    agent_name="my-agent",
    agent_class="MyAgent",
    memory_mb=512,
    s3_bucket="maif-data"
)

manager = DeploymentManager(config)
cdk_project = manager.generate_cdk_project(output_dir="./cdk")
```

## Monitoring & Observability

### CloudWatch Metrics

Agents automatically publish metrics to CloudWatch:

```python
# Automatic metrics include:
# - Perception count
# - Reasoning operations
# - Execution results
# - Error rates
# - Processing latency
```

### Custom Metrics

Add custom metrics:

```python
if self.use_aws and self.cloudwatch_logger:
    self.cloudwatch_logger.log_metric(
        metric_name="ItemsProcessed",
        value=self.items_processed,
        unit="Count",
        dimensions={
            "AgentId": self.agent_id,
            "Environment": "production"
        }
    )
```

### Alarms

Set up CloudWatch alarms:

```python
# CloudFormation template includes alarm definitions
alarms:
  - ErrorRate:
      threshold: 5
      evaluationPeriods: 2
      period: 300
  - MemoryUtilization:
      threshold: 80
      evaluationPeriods: 1
```

## Security Best Practices

### IAM Roles

Always use IAM roles instead of credentials:

```python
# Good - uses IAM role
agent = MyAgent(use_aws=True)

# Avoid - explicit credentials
agent = MyAgent(
    use_aws=True,
    aws_config=AWSConfig(
        access_key_id="AKIAIOSFODNN7EXAMPLE",  # Don't do this
        secret_access_key="wJalrXUtnFEMI..."   # Don't do this
    )
)
```

### Encryption

All data is encrypted by default:

- **At Rest**: S3 with KMS encryption
- **In Transit**: TLS 1.2+
- **Secrets**: AWS Secrets Manager

### Network Security

Use VPC for network isolation:

```python
config = DeploymentConfig(
    agent_name="secure-agent",
    vpc_config={
        "SubnetIds": ["subnet-12345", "subnet-67890"],
        "SecurityGroupIds": ["sg-12345"]
    }
)
```

## Error Handling

### Automatic Retry Logic

All AWS operations include exponential backoff:

```python
# Built-in retry configuration
retry_config = {
    "max_attempts": 3,
    "initial_backoff": 1.0,
    "max_backoff": 60.0,
    "backoff_multiplier": 2.0
}
```

### Error Classification

Errors are automatically classified:

```python
try:
    result = await agent.process()
except AWSServiceError as e:
    if e.is_retryable:
        # Automatic retry
        pass
    elif e.is_throttling:
        # Backoff and retry
        pass
    else:
        # Non-retryable error
        logger.error(f"Fatal error: {e}")
```

### Circuit Breaker

Automatic circuit breaker for failing services:

```python
# Services automatically disabled after repeated failures
if self.s3_circuit_breaker.is_open():
    # Fallback to local storage
    artifact.save(local_path)
```

## Cost Optimization

### S3 Lifecycle Policies

Automatic lifecycle management for artifacts:

```python
# Old artifacts moved to cheaper storage
lifecycle_rules = [
    {
        "id": "archive-old-artifacts",
        "transitions": [
            {"days": 30, "storage_class": "STANDARD_IA"},
            {"days": 90, "storage_class": "GLACIER"}
        ]
    }
]
```

### Lambda Cost Optimization

- Use appropriate memory settings
- Enable X-Ray sampling (not 100%)
- Use Step Functions for orchestration

### Monitoring Costs

Track costs with CloudWatch:

```python
# Cost metrics automatically tracked
cost_metrics = {
    "S3Storage": "USD/GB",
    "LambdaInvocations": "USD/million",
    "BedrockTokens": "USD/1k tokens"
}
```

## Troubleshooting

### Common Issues

1. **Permission Denied**
   ```python
   # Check IAM role has required permissions
   # Use AWS Policy Simulator to test
   ```

2. **Timeout Errors**
   ```python
   # Increase timeout settings
   config.timeout_seconds = 900  # 15 minutes
   ```

3. **Out of Memory**
   ```python
   # Increase memory allocation
   config.memory_mb = 3008  # Maximum for Lambda
   ```

### Debug Mode

Enable debug logging:

```python
import logging

# Enable debug logs
logging.basicConfig(level=logging.DEBUG)

# AWS SDK debug logs
boto3.set_stream_logger('boto3', logging.DEBUG)
```

### X-Ray Trace Analysis

Use X-Ray to debug performance issues:

1. Open X-Ray console
2. Filter by service name
3. Analyze service map
4. Drill into slow traces

## Production Checklist

Before deploying to production:

- [ ] IAM roles configured with least privilege
- [ ] Encryption enabled (KMS, S3, transit)
- [ ] VPC and security groups configured
- [ ] CloudWatch alarms set up
- [ ] X-Ray tracing enabled (with sampling)
- [ ] Cost budgets and alerts configured
- [ ] Backup and recovery tested
- [ ] Load testing completed
- [ ] Monitoring dashboards created
- [ ] Runbook documentation prepared

## Examples

- [Basic AWS Agent](../examples/aws_agent_demo.py)
- [AWS Agent with Backends](../examples/aws_agent_with_backends_demo.py)
- [AWS X-Ray Integration](../examples/aws_xray_agent_demo.py)
- [Agent State Restoration](../examples/agent_state_restoration_demo.py)
- [AWS Deployment](../examples/aws_deployment_demo.py)
- [Integrated Agent Demo](../examples/aws_integrated_agent_demo.py)

## API Reference

For detailed API documentation, see:
- [AWS Decorators API](../api/aws_decorators.md)
- [AWS Backend API](../api/aws_backend.md)
- [AWS Deployment API](../api/aws_deployment.md)