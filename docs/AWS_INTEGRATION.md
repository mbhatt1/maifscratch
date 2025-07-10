# AWS Integration with MAIF Agentic Framework

This guide explains how to use the decorator-based AWS integration with the MAIF agentic framework.

## Overview

The MAIF agentic framework provides simple decorator-based APIs for integrating with AWS services. These decorators make it easy to:

1. Create agents with MAIF that leverage AWS services
2. Enhance agent systems with AWS capabilities
3. Add AWS service access to specific methods

## Available Decorators

### Agent Creation Decorators

| Decorator | Description |
|-----------|-------------|
| `@maif_agent(**config)` | Create a MAIF agent from any class |
| `@aws_agent(**config)` | Create a fully AWS-integrated MAIF agent |

### AWS Service Decorators

| Decorator | Description |
|-----------|-------------|
| `@aws_bedrock()` | Add AWS Bedrock capabilities to a method |
| `@aws_kms()` | Add AWS KMS capabilities to a method |
| `@aws_s3()` | Add AWS S3 capabilities to a method |
| `@aws_lambda()` | Add AWS Lambda capabilities to a method |
| `@aws_dynamodb()` | Add AWS DynamoDB capabilities to a method |
| `@aws_step_functions()` | Add AWS Step Functions capabilities to a method |

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

### Fully AWS-Integrated Agent

```python
from maif.aws_decorators import aws_agent

@aws_agent(workspace="./agent_workspace")
class FullAWSAgent:
    async def process(self):
        # Uses AWS-enhanced systems automatically
        perception = await self.perceive("Analyze this data", "text")
        reasoning = await self.reason([perception])
        plan = await self.plan("create summary", [reasoning])
        result = await self.execute(plan)
        
        # All operations use AWS services under the hood
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